"""Deploy a SHA-tagged image to staging and roll back automatically on failure.

Runs **on the staging host** (the CI job copies it there over SSH) using nothing
but the standard library, so the VM needs docker + compose + python3 and no
project virtualenv. The contract is deliberately narrow:

    pull the immutable SHA tag -> up -d -> health -> 3 fixed Arabic reviews
    -> schema + label assertions

Anything that fails sends staging back to the last SHA that passed, and the
process still exits non-zero so the release is visibly failed in GitHub.

Three design points are load-bearing:

* **Only a smoke-tested SHA is ever promoted.** ``last_known_good`` is written
  after the assertions pass, never before, so a bad release can never become
  somebody's rollback target.
* **Health alone is not a passing deployment.** An image can serve ``/health``
  while ``/predict`` 500s -- that is exactly what happened once already, when
  the graph path was wrong. So the gate asserts real predictions with real
  expected labels, and compares the version the service reports against the
  OCI label baked into the image.
* **Rollback state lives outside the checkout**, in its own directory, so
  removing the copied tool files cannot destroy the record of what is known
  good.

The three fixtures are not guesses. Their labels and score floors come from
running the shipped int8 graph: positive 0.990 / negative 0.979 / neutral 0.660
in a single 3-review batch, and 0.990 / 0.981 / 0.693 one review at a time, so
the labels are stable across batch shapes. Note this model has no reliable
``neutral`` for most phrasings -- "وصل المنتج" scores 0.97 *positive* -- which
is why the neutral fixture is a sentence with a measured 0.45 margin rather
than a more obvious-sounding one.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_COMPOSE_FILE = "/opt/raay-staging/docker-compose.staging.yml"
DEFAULT_STATE_FILE = "/var/lib/raay-staging/deploy-state.json"
DEFAULT_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_HEALTH_TIMEOUT_S = 180
SERVICE = "raay-sentiment"
VALID_LABELS = ("positive", "negative", "neutral")
RESPONSE_KEYS = {"predictions", "model_version"}
PREDICTION_KEYS = {"label", "score"}
REVISION_LABEL = "org.opencontainers.image.revision"
VERSION_LABEL = "org.opencontainers.image.version"
HISTORY_LIMIT = 10
LOG_TAIL_LINES = 100
HEX = frozenset("0123456789abcdef")


class DeployError(RuntimeError):
    """A deploy step failed. The message is meant to be readable in a job log."""


@dataclass(frozen=True)
class SmokeCase:
    """One fixed review and the label it must produce."""

    text: str
    label: str
    min_score: float


#: Verified against the shipped int8 graph (see the module docstring). The
#: floors sit well below the observed scores so quantisation or CPU differences
#: cannot fail a healthy release, while a wrong label or a swapped graph still
#: does.
SMOKE_CASES = (
    SmokeCase("المنتج ممتاز وسريع التوصيل", "positive", 0.85),
    SmokeCase("الطلبية وصلت متأخرة جدا وتلفت البضاعة", "negative", 0.85),
    SmokeCase("المنتج بحجم متوسط", "neutral", 0.50),
)

Runner = Callable[..., subprocess.CompletedProcess[str]]
HttpCall = Callable[[str, str, dict[str, Any] | None, int], tuple[int, Any]]


def default_runner(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    stdin: str | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        cmd, env=env, input=stdin, capture_output=True, text=True, check=False
    )
    if check and proc.returncode != 0:
        raise DeployError(
            f"`{' '.join(cmd)}` exited {proc.returncode}\n"
            f"stdout: {proc.stdout.strip()}\nstderr: {proc.stderr.strip()}"
        )
    return proc


def default_http(
    url: str, method: str, payload: dict[str, Any] | None, timeout: int
) -> tuple[int, Any]:
    """Return ``(status, parsed_body)``. A non-2xx is a value, not an exception."""
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read() or b"null")
    except urllib.error.HTTPError as exc:
        body = exc.read()
        try:
            return exc.code, json.loads(body or b"null")
        except ValueError:
            return exc.code, None


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class Config:
    """Everything a deploy needs to talk to one staging host."""

    registry: str
    repository: str
    compose_file: str = DEFAULT_COMPOSE_FILE
    state_file: Path = Path(DEFAULT_STATE_FILE)
    base_url: str = DEFAULT_BASE_URL
    health_timeout_s: int = DEFAULT_HEALTH_TIMEOUT_S
    expect_model_version: str | None = None
    runner: Runner = default_runner
    http: HttpCall = default_http

    def image_for(self, sha: str) -> str:
        """Build the immutable ``registry/repository:sha`` reference.

        Refusing anything that is not a lowercase hex SHA is deliberate: the
        whole point of pinning is that staging cannot be retargeted by a moving
        tag, and ``latest`` would quietly break that. The length check matters
        as much as the alphabet -- ``abc`` is hex but is not a git SHA.
        """
        if len(sha) < 7 or not HEX.issuperset(sha.lower()):
            raise DeployError(
                f"target {sha!r} is not a git SHA; staging deploys immutable tags only"
            )
        if not self.registry or not self.repository:
            raise DeployError("a registry and repository are required")
        return f"{self.registry.rstrip('/')}/{self.repository.strip('/')}:{sha}"

    def compose_env(self, image: str) -> dict[str, str]:
        """The compose file refuses to interpolate an empty ``RAAY_IMAGE``."""
        return {**os.environ, "RAAY_IMAGE": image}

    def compose(self, *args: str) -> list[str]:
        return ["docker", "compose", "-f", self.compose_file, *args]

    def container_id(self, image: str) -> str | None:
        proc = self.runner(
            self.compose("ps", "-q", SERVICE), env=self.compose_env(image)
        )
        return proc.stdout.strip() or None

    def label(self, image: str, label: str) -> str | None:
        """Read an OCI label off the *running container*.

        Reading the container rather than the tag answers the only question a
        rollback cares about: what is actually running right now.
        """
        cid = self.container_id(image)
        if cid is None:
            return None
        proc = self.runner(
            [
                "docker",
                "inspect",
                "--format",
                f'{{{{ index .Config.Labels "{label}" }}}}',
                cid,
            ],
            check=False,
        )
        if proc.returncode != 0:
            return None
        value = proc.stdout.strip()
        return value if value and value != "<no value>" else None

    def logs(self, image: str) -> str:
        proc = self.runner(
            self.compose("logs", "--no-color", "--tail", str(LOG_TAIL_LINES)),
            env=self.compose_env(image),
            check=False,
        )
        return (proc.stdout + proc.stderr).strip()


@dataclass
class Report:
    """Machine-readable outcome: printed as JSON and folded into the summary."""

    target_sha: str
    image: str = ""
    previous_sha: str | None = None
    outcome: str = "failed"
    rolled_back_to: str | None = None
    checks: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    logs: str = ""

    @property
    def ok(self) -> bool:
        return self.outcome in {"deployed", "already_deployed"}

    def as_dict(self) -> dict[str, Any]:
        return {
            "target_sha": self.target_sha,
            "image": self.image,
            "previous_sha": self.previous_sha,
            "outcome": self.outcome,
            "rolled_back_to": self.rolled_back_to,
            "checks": self.checks,
            "errors": self.errors,
        }


def read_state(path: Path) -> dict[str, Any]:
    """Load the deploy state, tolerating a missing or corrupt file.

    A corrupt state file must not block a deploy; it only costs the remembered
    rollback target, which :meth:`Config.label` can recover from the live
    container.
    """
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def write_state(path: Path, state: dict[str, Any]) -> None:
    """Write atomically: a host that dies mid-write must not lose the record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def login(cfg: Config, user: str, token: str) -> None:
    """``docker login`` over stdin, so the token never reaches argv or a log."""
    if not token:
        raise DeployError(
            "no registry token: set RAAY_REGISTRY_TOKEN or pass --registry-token-file"
        )
    cfg.runner(
        ["docker", "login", cfg.registry, "-u", user, "--password-stdin"], stdin=token
    )


def pull_and_up(cfg: Config, image: str, report: Report) -> None:
    env = cfg.compose_env(image)
    report.checks.append(f"docker compose pull {image}")
    cfg.runner(cfg.compose("pull"), env=env)
    cfg.runner(cfg.compose("up", "-d", "--remove-orphans"), env=env)


def wait_for_health(
    cfg: Config,
    image: str,
    report: Report,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Poll ``/health`` until it answers, the container dies, or time runs out.

    The container-died check matters: without it a crash-looping release burns
    the whole timeout before anyone learns why.
    """
    deadline = monotonic() + cfg.health_timeout_s
    last = "no response yet"
    while monotonic() < deadline:
        try:
            status, body = cfg.http(f"{cfg.base_url}/health", "GET", None, 5)
            if status == 200 and isinstance(body, dict):
                report.checks.append("/health answered 200")
                return body
            last = f"status={status} body={body!r}"
        except (urllib.error.URLError, OSError, ValueError) as exc:
            last = f"{type(exc).__name__}: {exc}"
        if cfg.container_id(image) is None:
            raise DeployError(f"the {SERVICE} container is gone; last probe: {last}")
        running = cfg.runner(
            [
                "docker",
                "inspect",
                "--format",
                "{{.State.Running}}",
                cfg.container_id(image),
            ],
            check=False,
        )
        if running.stdout.strip() != "true":
            raise DeployError(
                f"the {SERVICE} container is not running; last probe: {last}"
            )
        sleep(2.0)
    raise DeployError(
        f"{SERVICE} was not healthy within {cfg.health_timeout_s}s; last probe: {last}"
    )


def check_health_body(
    body: dict[str, Any], expected_version: str | None, report: Report
) -> None:
    if body.get("status") != "healthy":
        raise DeployError(
            f"/health reported status={body.get('status')!r}, expected 'healthy'"
        )
    report.checks.append("/health status is healthy")
    reported = body.get("model_version")
    if expected_version is None:
        report.checks.append(f"/health model_version={reported!r} (unverified)")
        return
    if reported != expected_version:
        raise DeployError(
            f"/health reports model_version={reported!r}, but the image is "
            f"labelled {expected_version!r}"
        )
    report.checks.append(
        f"/health model_version matches the image label ({expected_version})"
    )


def check_one_prediction(case: SmokeCase, prediction: Any) -> None:
    if not isinstance(prediction, dict):
        raise DeployError(f"prediction for {case.text!r} is not an object")
    keys = set(prediction)
    if keys != PREDICTION_KEYS:
        raise DeployError(
            f"prediction keys are {sorted(keys)}, expected {sorted(PREDICTION_KEYS)}"
        )
    label = prediction["label"]
    if label not in VALID_LABELS:
        raise DeployError(f"label {label!r} is not one of {list(VALID_LABELS)}")
    if label != case.label:
        raise DeployError(
            f"{case.text!r} was labelled {label!r}, expected {case.label!r}"
        )
    score = prediction["score"]
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise DeployError(f"score {score!r} is not a number")
    if not 0.0 <= float(score) <= 1.0:
        raise DeployError(f"score {score!r} is outside [0, 1]")
    if float(score) < case.min_score:
        raise DeployError(
            f"{case.text!r} scored {float(score):.4f}, below the "
            f"{case.min_score:.2f} floor for {case.label!r}"
        )


def check_predict_payload(
    body: Any, expected_version: str | None, report: Report
) -> None:
    if not isinstance(body, dict):
        raise DeployError(
            f"/predict returned {type(body).__name__}, expected an object"
        )
    keys = set(body)
    if keys != RESPONSE_KEYS:
        raise DeployError(
            f"/predict response keys are {sorted(keys)}, expected {sorted(RESPONSE_KEYS)}"
        )
    predictions = body["predictions"]
    if not isinstance(predictions, list):
        raise DeployError("/predict 'predictions' is not a list")
    if len(predictions) != len(SMOKE_CASES):
        raise DeployError(
            f"/predict returned {len(predictions)} predictions, "
            f"expected {len(SMOKE_CASES)}"
        )
    for case, prediction in zip(SMOKE_CASES, predictions, strict=True):
        check_one_prediction(case, prediction)
        report.checks.append(f"{case.label}: {case.text[:24]}...")
    if expected_version is not None and body["model_version"] != expected_version:
        raise DeployError(
            f"/predict reports model_version={body['model_version']!r}, "
            f"expected {expected_version!r}"
        )
    report.checks.append("/predict labels, scores and schema are correct")


def check_validation_rejects_non_strings(cfg: Config, report: Report) -> None:
    """A 422 here means the pydantic wiring survived the image build."""
    status, _ = cfg.http(f"{cfg.base_url}/predict", "POST", {"texts": [1]}, 10)
    if status != 422:
        raise DeployError(f"a non-string review returned HTTP {status}, expected 422")
    report.checks.append("non-string review is rejected with 422")


def run_smoke(
    cfg: Config,
    image: str,
    report: Report,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> None:
    expected = cfg.expect_model_version
    check_health_body(
        wait_for_health(cfg, image, report, sleep, monotonic), expected, report
    )
    status, payload = cfg.http(
        f"{cfg.base_url}/predict",
        "POST",
        {"texts": [case.text for case in SMOKE_CASES]},
        60,
    )
    if status != 200:
        raise DeployError(f"/predict returned HTTP {status}, expected 200")
    check_predict_payload(payload, expected, report)
    check_validation_rejects_non_strings(cfg, report)


def _record(state: dict[str, Any], sha: str, outcome: str) -> None:
    history = [entry for entry in state.get("history", []) if isinstance(entry, dict)]
    history.append({"sha": sha, "outcome": outcome, "at": _now()})
    state["history"] = history[-HISTORY_LIMIT:]
    state["updated_at"] = _now()


def rollback(
    cfg: Config,
    report: Report,
    state: dict[str, Any],
    state_file: Path,
    sleep: Callable[[float], None],
    monotonic: Callable[[], float],
) -> Report:
    """Put the last smoke-tested SHA back. Never promotes anything new."""
    target = report.previous_sha
    if not target or target == report.target_sha:
        report.outcome = "failed"
        report.errors.append(
            "no rollback target: nothing has ever passed the smoke test on this "
            "host, so there is no previous good version to return to"
        )
        return report

    image = cfg.image_for(target)
    report.checks.append(f"rolling back to {target}")
    try:
        pull_and_up(cfg, image, report)
        if cfg.label(image, REVISION_LABEL) != target:
            raise DeployError(
                f"the rolled-back container reports revision "
                f"{cfg.label(image, REVISION_LABEL)!r}, expected {target!r}"
            )
        wait_for_health(cfg, image, report, sleep, monotonic)
    except (DeployError, urllib.error.URLError, OSError, ValueError) as exc:
        report.outcome = "failed"
        report.errors.append(f"ROLLBACK FAILED: {exc}")
        return report

    report.outcome = "rolled_back"
    report.rolled_back_to = target
    report.errors.append(f"smoke test failed; staging is back on {target}")
    # last_known_good stays exactly as it was: the previous release is still the
    # newest thing known to work, and therefore still the rollback target.
    _record(state, report.target_sha, "rolled_back")
    write_state(state_file, state)
    return report


def deploy(
    cfg: Config,
    target_sha: str,
    *,
    registry_user: str = "github-actions",
    registry_token: str = "",
    skip_login: bool = False,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> Report:
    image = cfg.image_for(target_sha)
    report = Report(target_sha=target_sha, image=image)
    state = read_state(cfg.state_file)
    remembered = state.get("last_known_good")
    report.previous_sha = remembered if isinstance(remembered, str) else None

    if not skip_login:
        login(cfg, registry_user, registry_token)
        report.checks.append(f"docker login {cfg.registry}")

    running = cfg.label(image, REVISION_LABEL)
    if report.previous_sha is None and running and running != target_sha:
        # Bootstrap the state file from the container that is already serving.
        report.previous_sha = running
    report.checks.append(f"previously running revision: {running or 'none'}")

    if not skip_login:
        state["last_attempt"] = target_sha
        _record(state, target_sha, "attempted")
        write_state(cfg.state_file, state)

    try:
        if running == target_sha:
            report.checks.append("already on the target SHA; re-verifying in place")
        else:
            pull_and_up(cfg, image, report)
        revision = cfg.label(image, REVISION_LABEL)
        if revision != target_sha:
            raise DeployError(
                f"the deployed container reports revision {revision!r}, "
                f"expected {target_sha!r}"
            )
        report.checks.append("deployed image label matches the target SHA")
        if cfg.expect_model_version is None:
            cfg.expect_model_version = cfg.label(image, VERSION_LABEL)
        run_smoke(cfg, image, report, sleep, monotonic)
    except (DeployError, urllib.error.URLError, OSError, ValueError) as exc:
        report.errors.append(str(exc))
        report.logs = cfg.logs(image)
        return rollback(cfg, report, state, cfg.state_file, sleep, monotonic)

    state["last_known_good"] = target_sha
    state["last_attempt"] = target_sha
    _record(state, target_sha, "deployed")
    write_state(cfg.state_file, state)
    report.outcome = (
        "deployed" if report.previous_sha != target_sha else "already_deployed"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target-sha", required=True)
    parser.add_argument("--registry", default="ghcr.io")
    parser.add_argument("--repository", required=True)
    parser.add_argument("--compose-file", default=DEFAULT_COMPOSE_FILE)
    parser.add_argument("--state-file", default=DEFAULT_STATE_FILE)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--health-timeout", type=int, default=DEFAULT_HEALTH_TIMEOUT_S)
    parser.add_argument(
        "--expect-model-version",
        default=None,
        help="cross-check the reported model_version against this value",
    )
    parser.add_argument("--registry-user", default="github-actions")
    parser.add_argument(
        "--registry-token-file",
        default=None,
        help="read the registry token from this file instead of RAAY_REGISTRY_TOKEN",
    )
    parser.add_argument(
        "--skip-login",
        action="store_true",
        help="assume the host is already logged in (local rehearsals)",
    )
    args = parser.parse_args(argv)

    token = os.environ.get("RAAY_REGISTRY_TOKEN", "")
    if args.registry_token_file:
        token = Path(args.registry_token_file).read_text().strip()

    cfg = Config(
        registry=args.registry,
        repository=args.repository,
        compose_file=args.compose_file,
        state_file=Path(args.state_file),
        base_url=args.base_url,
        health_timeout_s=args.health_timeout,
        expect_model_version=args.expect_model_version,
    )
    try:
        report = deploy(
            cfg,
            args.target_sha,
            registry_user=args.registry_user,
            registry_token=token,
            skip_login=args.skip_login,
        )
    except DeployError as exc:
        print(json.dumps({"outcome": "failed", "errors": [str(exc)]}, indent=2))
        return 1

    print(json.dumps(report.as_dict(), indent=2, ensure_ascii=False))
    if report.logs:
        print("\n--- staging container logs (tail) ---", file=sys.stderr)
        print(report.logs, file=sys.stderr)
    if report.ok:
        return 0
    print(
        f"\nDeploy FAILED: staging is now on "
        f"{report.rolled_back_to or 'nothing working'}.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
