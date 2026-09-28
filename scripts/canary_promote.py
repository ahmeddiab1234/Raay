"""Shadow-then-canary rollout for the Raay service (Phase 5 step 5).

The rollout owns the *traffic split*, not just the registry alias. It drives
the standalone ``docker-compose.canary.yml`` project (``name: canary``) where
two workers -- ``raay-sentiment`` (the current Production graph) and
``raay-sentiment-canary`` (the declared candidate) -- sit behind an nginx
front that publishes :8000 (live ingress) and :8081 (operator front). nginx
owns :8000 only while this project is up, so the rollback is a conf change,
not a compose swap.

Stage progression (one conf render + ``nginx -s reload`` per stage):

    shadow       stable 100 / candidate 0; every /predict is mirrored to the
                 candidate (nginx ``mirror /shadow``). Both hops carry
                 ``X-Raay-Shadow: 1`` so the agent expects a pair.
    canary-5     95/5 weighted split, no mirror.
    canary-25    75/25.
    canary-50    50/50.
    full         candidate 100, no mirror, then the Phase 6 offline gate.

A stage only advances on gates fed by the compose-local Prometheus (which
scrapes the canary-agent that the workers POST events to). The shadow stage
gates on label agreement + candidate error rate + p95 ratio vs the stable
worker; the weighted stages gate on error rate + p95 ratio only (agreement
needs the mirror). ``--to full`` *additionally* runs the Phase 6 offline
promotion gate (``scripts/promote_model.py``) -- the only code allowed to
move the ``Production`` alias -- so the alias can never flip without both the
online and offline gates being clean.

Rollback re-renders the stable-only conf and flips the alias back to the
version recorded when the shadow started. The state file
``reports/canary_state.json`` (git-ignored) tracks the phase, the candidate,
the pre-rollout Production version and the last gate result; a promotion and
its rollback are one atomic story rooted in that file.

Modes (run from repo root):

    uv run python scripts/canary_promote.py --mode declare                 # idempotent register of the candidate
    uv run python scripts/canary_promote.py --mode shadow                  # enter shadow (mirror, 100/0)
    uv run python scripts/canary_promote.py --mode advance --to canary-5   # gate + widen; also 25 / 50 / full
    uv run python scripts/canary_promote.py --mode rollback                # stable-only conf + alias back to pre-rollout
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import urllib.parse
import urllib.request
import warnings
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import URLError

import mlflow
import onnx
import yaml
from loguru import logger

from raay.config.env import load_environment, mlflow_tracking_uri
from raay.enums.constants import DefaultPaths, Experiments, Models

warnings.filterwarnings("ignore", category=SyntaxWarning)

_TOOL_TAG = "scripts.canary_promote"
_MODEL = Models.REGISTERED_BASELINE.value  # both variants live under ArabicSentiment
_STAGE_CANARY = "Canary"
_STAGE_PRODUCTION = "Production"

_PHASES = ("shadow", "canary-5", "canary-25", "canary-50", "full")
# The only legal transitions. ``advance --to X`` must name exactly the edge
# out of the current phase.
_EDGES = {
    "shadow": "canary-5",
    "canary-5": "canary-25",
    "canary-25": "canary-50",
    "canary-50": "full",
}

# Health gates. nginx owns :8000 through the rollout, so each worker is also
# published on its own direct port so the operator can gate it independently of
# any weighted routing.
_STABLE_URL = "http://127.0.0.1:8002/health"
_CANDIDATE_URL = "http://127.0.0.1:8001/health"
_FRONT_URL = "http://127.0.0.1:8081/health"
_INGRESS_URL = "http://127.0.0.1:8000/health"
_AGENT_URL = "http://127.0.0.1:9100/health"
_PROMETHEUS_HEALTH_URL = "http://127.0.0.1:9090/-/healthy"

_STABLE_UPSTREAM = "server raay-sentiment:3000"
_CANDIDATE_UPSTREAM = "server raay-sentiment-canary:3000"
_DEFAULT_COMPOSE_FILE = "docker-compose.canary.yml"
_DEFAULT_CONF_PATH = "deploy/nginx_canary.conf"
_DEFAULT_STATE_PATH = "reports/canary_state.json"

# The Phase 6 offline gate that ``--to full`` shells out to. It is the only
# code allowed to move the Production alias; a passing single invocation both
# verifies and flips (the 14 offline gates are re-checks of a frozen split, so
# no second pass is needed the way the human-approval workflow requires one).
_PROMOTE_SCRIPT_ARGV = ("uv", "run", "python", "scripts/promote_model.py")


def _weights_for(phase: str) -> tuple[int, int]:
    return {
        "shadow": (100, 0),
        "canary-5": (95, 5),
        "canary-25": (75, 25),
        "canary-50": (50, 50),
        "full": (0, 100),
        "rollback": (100, 0),
    }[phase]


def render_canary_conf(phase: str) -> str:
    """Render the nginx conf for a rollout stage.

    ``shadow`` mirrors every ``/predict`` to the candidate via ``/shadow`` and
    marks **both** hops ``X-Raay-Shadow: 1`` so the agent expects a pair.
    Every later phase drops the mirror entirely -- agreement needs the mirror
    by construction. The two locations that proxy to the candidate never touch
    ``$request_id``: a mirror subrequest is a separate request object, so its
    ``$request_id`` differs from the main request's, and ``proxy_set_header``
    on the main location is invisible to the mirror (nginx clones only the
    client-sent headers). Correlation therefore happens in the agent on the
    request content, which the mirror does clone.
    """

    stable_weight, candidate_weight = _weights_for(phase)
    mirrored = phase == "shadow"

    predict_hops = [
        "        proxy_pass http://raay_sentiment_backend/predict;",
        "        proxy_set_header Host $host;",
        "        proxy_set_header X-Real-IP $remote_addr;",
        "        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;",
    ]
    if mirrored:
        predict_hops.append("        proxy_set_header X-Raay-Shadow 1;")
        predict_hops.append("        mirror /shadow;")

    shadow_block = ""
    if mirrored:
        shadow_block = (
            "    location /shadow {\n"
            "        proxy_pass http://raay-sentiment-canary:3000/predict;\n"
            "        proxy_set_header Host $host;\n"
            "        proxy_set_header X-Raay-Shadow 1;\n"
            "    }\n\n"
        )

    return (
        "# Phase 5 step 5 -- nginx front for the shadow/canary rollout.\n"
        "# Generated by scripts/canary_promote.py -- do not hand-edit.\n"
        f"# Stage: {phase} (stable={stable_weight}, candidate={candidate_weight}, "
        f"mirror={'yes' if mirrored else 'no'})\n\n"
        "upstream raay_sentiment_backend {\n"
        f"    {_STABLE_UPSTREAM} weight={stable_weight};\n"
        f"    {_CANDIDATE_UPSTREAM} weight={candidate_weight};\n"
        "}\n\n"
        "server {\n"
        "    listen 8000;\n"
        "    listen 8081;\n\n"
        "    location /predict {\n"
        + "\n".join(predict_hops)
        + "\n    }\n\n"
        + shadow_block
        + "    location /health {\n"
        "        proxy_pass http://raay_sentiment_backend/health;\n"
        "        proxy_set_header Host $host;\n"
        "    }\n"
        "}\n"
    )


# ---------------------------------------------------------------------- state


@dataclass
class CanaryState:
    """Owned by the rollout: phase, candidate, pre-rollout Production, gates."""

    phase: str
    candidate_version: int
    previous_production_version: int | None
    entered_at: str
    prometheus_url: str
    last_gate: dict[str, Any] | None = None
    edges: list[dict[str, Any]] = field(default_factory=list)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(content)
    tmp.replace(path)


def _load_state(path: Path) -> CanaryState:
    if not path.exists():
        raise SystemExit(f"No rollout state at {path}. Start with `--mode shadow`.")
    return CanaryState(**json.loads(path.read_text()))


def _save_state(state: CanaryState, path: Path) -> None:
    _write_text(path, json.dumps(asdict(state), indent=2) + "\n")


# ------------------------------------------------------------------ registry


def _production_version(client: mlflow.tracking.MlflowClient) -> int | None:
    """The version the Production alias points at right now (pre-rollout)."""
    try:
        version = client.get_model_version_by_alias(_MODEL, _STAGE_PRODUCTION)
    except Exception:  # noqa: BLE001 - no alias yet means first promotion, nothing to roll back to
        return None
    if not version:
        return None
    try:
        return int(version)
    except (TypeError, ValueError):
        return None


def _int8_version(client: mlflow.tracking.MlflowClient) -> int:
    versions = [
        v
        for v in client.search_model_versions(f"name = '{_MODEL}'")
        if v.tags.get("variant") == "onnx-int8"
    ]
    if not versions:
        raise SystemExit(
            f"No onnx-int8 version under {_MODEL} to roll back to. "
            f"Run `scripts/benchmark.py` + `scripts/log_variants_mlflow.py` first."
        )
    return int(versions[0].version)


def _flip_production(client: mlflow.tracking.MlflowClient, version: int) -> None:
    client.transition_model_version_stage(
        name=_MODEL,
        version=version,
        stage=_STAGE_PRODUCTION,
        archive_existing_versions=True,
    )
    client.set_registered_model_alias(_MODEL, _STAGE_PRODUCTION, str(version))
    logger.info(f"'{_STAGE_PRODUCTION}' alias -> v{version}")


# --------------------------------------------------------------------- gates


def _health_gate(url: str) -> None:
    """Fail loudly if a worker healthcheck is not 200."""
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            body = resp.read().decode("utf-8", "replace")
            status = resp.status
    except URLError as exc:
        raise SystemExit(f"Health gate FAILED ({url}): {exc}") from exc
    if status != 200:
        raise SystemExit(f"Health gate FAILED ({url}): HTTP {status}")
    logger.info(f"Health gate OK ({url}): {body.strip()}")


def _health_gates(urls: Iterable[str]) -> None:
    for url in urls:
        _health_gate(url)


def _promql_query(base_url: str, promql: str) -> float | None:
    """Instant query against the Prometheus HTTP API; None = no samples/error.

    A missing series must be distinguishable from a failed scrape: the caller
    treats ``None`` as INCONCLUSIVE so an empty canary never reads as a pass.
    """
    url = (
        f"{base_url.rstrip('/')}/api/v1/query?"
        f"{urllib.parse.urlencode({'query': promql})}"
    )
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            payload = json.loads(resp.read().decode("utf-8", "replace"))
    except Exception as exc:  # noqa: BLE001 - a dead Prometheus is INCONCLUSIVE, not a crash
        logger.warning(f"Prometheus query not answered: {exc}")
        return None
    if payload.get("status") != "success":
        logger.warning(f"Prometheus query error: {payload.get('error')}")
        return None
    result = (payload.get("data") or {}).get("result") or []
    if not result:
        return None
    try:
        return float(result[0]["value"][1])
    except (KeyError, IndexError, TypeError, ValueError):
        return None


@dataclass
class GateOutcome:
    passed: bool
    inconclusive: bool
    reason: str
    observed: dict[str, float | int | None]


def evaluate_gate(
    phase: str,
    *,
    query: Callable[[str], float | None],
    entered_at: str | None = None,
    window: str = "5m",
    min_requests: int = 500,
    hold_seconds: int = 600,
    agreement_min: float = 0.99,
    error_rate_max: float = 0.005,
    p95_ratio_max: float = 1.10,
    now: Callable[[], datetime] | None = None,
) -> GateOutcome:
    """Run the stage gates over the accumulated window.

    Semantics (all over the trailing ``window``):

    * candidate must have seen a real request stream (``--min-requests``) and
      the current stage must have been held for ``--hold-seconds`` -- a wait,
      not a fail-if-data-missing, so a short window exits 1 (operator waits).
    * candidate error rate <= ``--error-rate-max``.
    * candidate p95 <= ``--p95-ratio-max`` x stable p95 (same box, so it is a
      fair A/B).
    * shadow stage only: label agreement >= ``--agreement-min``.

    ``None`` from any query (missing series, Prometheus down, zero traffic in a
    window) is INCONCLUSIVE, never a pass: an empty canary must not advance.
    """

    observed: dict[str, float | int | None] = {}

    def _q(name: str, promql: str) -> float | None:
        value = query(promql)
        observed[name] = value
        return value

    def inconclusive(reason: str) -> GateOutcome:
        return GateOutcome(
            passed=False, inconclusive=True, reason=reason, observed=observed
        )

    def fail(reason: str) -> GateOutcome:
        return GateOutcome(
            passed=False, inconclusive=False, reason=reason, observed=observed
        )

    requests_c = _q(
        "candidate_requests",
        f'sum(increase(raay_ingest_total{{worker="candidate"}}[{window}]))',
    )
    errors_c = _q(
        "candidate_errors",
        f'sum(increase(raay_errors_total{{worker="candidate"}}[{window}]))',
    )
    p95_c = _q(
        "candidate_p95_ms",
        f'histogram_quantile(0.95, sum(rate(raay_latency_seconds_bucket{{worker="candidate"}}[{window}])) by (le))',
    )
    p95_s = _q(
        "stable_p95_ms",
        f'histogram_quantile(0.95, sum(rate(raay_latency_seconds_bucket{{worker="stable"}}[{window}])) by (le))',
    )

    if requests_c is None or requests_c <= 0:
        return inconclusive(
            "no candidate requests in the window (metric missing or scrape broken)"
        )
    error_rate = (errors_c or 0.0) / requests_c
    if error_rate > error_rate_max:
        return fail(f"candidate error rate {error_rate:.4f} > max {error_rate_max}")

    if p95_c is None or p95_s is None or p95_c <= 0 or p95_s <= 0:
        return inconclusive("latency percentile missing (no samples in window)")

    if entered_at is not None:
        held_for = (
            now() if now is not None else datetime.now(UTC)
        ) - datetime.fromisoformat(entered_at)
        held_s = held_for.total_seconds()
        if held_s < hold_seconds:
            return fail(
                f"stage held only {held_s:.0f}s < {hold_seconds}s",
            )

    if requests_c < min_requests:
        return fail(
            f"candidate saw {requests_c:.0f} requests < --min-requests {min_requests}"
        )

    ratio = p95_c / p95_s
    if ratio > p95_ratio_max:
        return fail(
            f"candidate p95 {p95_c:.1f}ms is {ratio:.2f}x stable p95 {p95_s:.1f}ms "
            f"(> {p95_ratio_max}x)"
        )

    if phase == "shadow":
        agree = _q(
            "agreement_agree",
            f'sum(increase(raay_agreement_total{{status="agree"}}[{window}]))',
        )
        disagree = _q(
            "agreement_disagree",
            f'sum(increase(raay_agreement_total{{status="disagree"}}[{window}]))',
        )
        total = (agree or 0.0) + (disagree or 0.0)
        if total <= 0:
            return inconclusive("no paired shadow traffic in the window")
        rate = (agree or 0.0) / total
        if rate < agreement_min:
            return fail(
                f"shadow agreement {rate:.4f} < --agreement-min {agreement_min}"
            )

    return GateOutcome(
        passed=True,
        inconclusive=False,
        reason="all gates passed",
        observed=observed,
    )


def _reload_nginx(compose_file: str) -> None:
    cmd = [
        "docker",
        "compose",
        "-f",
        compose_file,
        "exec",
        "-T",
        "raay-nginx",
        "nginx",
        "-s",
        "reload",
    ]
    logger.info(f"Reloading nginx: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _run_offline_gate(candidate_version: int) -> None:
    """Run the Phase 6 offline gate; the only code that flips the alias."""
    argv = [*_PROMOTE_SCRIPT_ARGV, "--candidate-version", str(candidate_version)]
    logger.info(f"Running Phase 6 offline gate: {' '.join(argv)}")
    subprocess.run(argv, check=True)


def _gate_report(
    phase: str, target: str, outcome: GateOutcome, thresholds: dict
) -> dict:
    return {
        "decision": "PASS"
        if outcome.passed
        else ("INCONCLUSIVE" if outcome.inconclusive else "FAIL"),
        "phase": phase,
        "target": target,
        "reason": outcome.reason,
        "observed": outcome.observed,
        "thresholds": thresholds,
        "evaluated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def _write_gate_report(report: dict[str, Any], phase: str) -> Path:
    path = Path("reports") / f"canary_gate_{phase}.json"
    _write_text(path, json.dumps(report, indent=2) + "\n")
    return path


# -------------------------------------------------------------------- modes


def _materialize_canary_model_dir(run_id: str, onnx_path: str, out_dir: Path) -> str:
    """Build a self-contained mlflow Model dir (onnx flavor) for the distilled fp32 graph.

    Mirrors ``_materialize_model_dir`` in ``scripts/log_variants_mlflow.py``:
    mlflow 3.15 + the relative artifact_location here stages ``mlflow.onnx``
    models into internal ``mlruns/<exp>/models/m-*`` copies that make
    ``runs:/<run_id>/model`` unresolvable, so we assemble the Model dir
    ourselves and register straight from that path.
    """

    out_dir = out_dir.resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # The graph uses external weights (distilled.onnx.data) next to it -- copy
    # both so the registered artifact is self-contained and locally loadable.
    shutil.copy(onnx_path, out_dir / "model.onnx")
    external = Path(onnx_path).with_suffix(".onnx.data")
    if external.exists():
        shutil.copy(external, out_dir / "model.onnx.data")

    (out_dir / "requirements.txt").write_text(
        f"mlflow=={mlflow.__version__}\nonnxruntime>=1.18.0\nnumpy\n"
    )
    (out_dir / "python_env.yaml").write_text(
        "python: 3.12.3\n"
        "build_dependencies:\n"
        "- pip\n"
        "- setuptools==84.0.0\n"
        "- wheel\n"
        "dependencies:\n"
        "- -r requirements.txt\n"
    )
    (out_dir / "conda.yaml").write_text(
        "channels:\n"
        "- conda-forge\n"
        "dependencies:\n"
        "- python=3.12.3\n"
        "- pip\n"
        "- pip:\n"
        f"  - mlflow=={mlflow.__version__}\n"
        "  - onnxruntime>=1.18.0\n"
        "  - numpy\n"
        "name: mlflow-env\n"
    )
    mlmodel = {
        "artifact_path": str(out_dir),
        "flavors": {
            "onnx": {
                "code": None,
                "data": "model.onnx",
                "onnx_session_options": None,
                "onnx_version": onnx.__version__,
                "providers": ["CPUExecutionProvider"],
            },
            "python_function": {
                "data": "model.onnx",
                "env": {"conda": "conda.yaml", "virtualenv": "python_env.yaml"},
                "loader_module": "mlflow.onnx",
                "python_version": platform.python_version(),
            },
        },
        "mlflow_version": mlflow.__version__,
        "run_id": run_id,
        "utc_time_created": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S.%f"),
    }
    (out_dir / "MLmodel").write_text(yaml.safe_dump(mlmodel, sort_keys=False))
    return str(out_dir)


def declare(
    client: mlflow.tracking.MlflowClient, experiment: str
) -> mlflow.entities.ModelVersion:
    """Register the distilled fp32 graph as the ``Canary`` alias. Idempotent."""

    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        raise SystemExit(f"Experiment {experiment!r} not found")
    experiment_id = exp.experiment_id

    referenced = {v.run_id for v in client.search_model_versions() if v.run_id}
    prior = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.tool = '{_TOOL_TAG}'",
    )
    to_delete = [r for r in prior if r.info.run_id not in referenced]
    for run in to_delete:
        client.delete_run(run.info.run_id)
    if to_delete:
        logger.info(f"Deleted {len(to_delete)} previous canary run(s)")
    if len(prior) - len(to_delete):
        logger.info(
            f"Kept {len(prior) - len(to_delete)} canary run(s) referenced by the registry"
        )

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name="register-canary-distilled-fp32",
        tags={
            "tool": _TOOL_TAG,
            "stage": "canary",
            "variant": "distilled-fp32",
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        },
    ) as run:
        run_id = run.info.run_id
        logger.info(f"Logged canary run {run_id}")

    model_uri = _materialize_canary_model_dir(
        run_id=run_id,
        onnx_path=DefaultPaths.ONNX_DISTILLED.value,
        out_dir=Path("reports") / "models" / "canary_distilled_fp32_mlflow",
    )
    logger.info(f"Materialized canary MLmodel dir -> {model_uri}")

    registered = mlflow.register_model(model_uri=model_uri, name=_MODEL)
    client.set_registered_model_alias(_MODEL, _STAGE_CANARY, str(registered.version))
    client.set_model_version_tag(_MODEL, registered.version, "mlflow.run_id", run_id)
    client.set_model_version_tag(
        _MODEL, registered.version, "variant", "distilled-fp32"
    )
    client.set_model_version_tag(_MODEL, registered.version, "stage", "canary")
    logger.info(f"Registered {_MODEL} v{registered.version} -> '{_STAGE_CANARY}' alias")
    return registered


def _canary_version(
    client: mlflow.tracking.MlflowClient,
) -> mlflow.entities.ModelVersion:
    versions = [
        v
        for v in client.search_model_versions(f"name = '{_MODEL}'")
        if v.tags.get("variant") == "distilled-fp32"
    ]
    if not versions:
        raise SystemExit(
            f"No distilled-fp32 version under {_MODEL}. Run `--mode declare` first."
        )
    return versions[0]


def cmd_shadow(
    client: mlflow.tracking.MlflowClient,
    *,
    conf_path: Path,
    compose_file: str,
    state_path: Path,
    prometheus_url: str,
) -> None:
    """Health-gate everything, then start mirroring every /predict at the candidate."""
    _health_gates(
        (
            _STABLE_URL,
            _CANDIDATE_URL,
            _FRONT_URL,
            _INGRESS_URL,
            _AGENT_URL,
            _PROMETHEUS_HEALTH_URL,
        )
    )
    candidate = _canary_version(client)
    _write_text(conf_path, render_canary_conf("shadow"))
    _reload_nginx(compose_file)
    state = CanaryState(
        phase="shadow",
        candidate_version=int(candidate.version),
        previous_production_version=_production_version(client),
        entered_at=datetime.now(UTC).isoformat(timespec="seconds"),
        prometheus_url=prometheus_url,
    )
    _save_state(state, state_path)
    logger.info(
        f"shadow started: {_MODEL} v{state.candidate_version} mirrored behind "
        f"Production v{state.previous_production_version}"
    )


def cmd_advance(
    client: mlflow.tracking.MlflowClient,
    *,
    to: str,
    conf_path: Path,
    compose_file: str,
    state_path: Path,
    window: str,
    min_requests: int,
    hold_seconds: int,
    agreement_min: float,
    error_rate_max: float,
    p95_ratio_max: float,
) -> None:
    """Gate the current stage, then widen the candidate slice (or go full)."""
    state = _load_state(state_path)
    if _EDGES.get(state.phase) != to:
        raise SystemExit(
            f"cannot advance {state.phase!r} -> {to!r}; "
            f"expected {_EDGES.get(state.phase)!r}"
        )

    _health_gates((_STABLE_URL, _CANDIDATE_URL, _AGENT_URL))

    def query(promql: str) -> float | None:
        return _promql_query(state.prometheus_url, promql)

    outcome = evaluate_gate(
        state.phase,
        query=query,
        entered_at=state.entered_at,
        window=window,
        min_requests=min_requests,
        hold_seconds=hold_seconds,
        agreement_min=agreement_min,
        error_rate_max=error_rate_max,
        p95_ratio_max=p95_ratio_max,
    )
    thresholds = {
        "window": window,
        "min_requests": min_requests,
        "hold_seconds": hold_seconds,
        "agreement_min": agreement_min,
        "error_rate_max": error_rate_max,
        "p95_ratio_max": p95_ratio_max,
    }
    report = _gate_report(state.phase, to, outcome, thresholds)
    gate_report = _write_gate_report(report, state.phase)
    print(json.dumps(report, indent=2))

    if not outcome.passed:
        print(f"Gate report: {gate_report}")
        raise SystemExit(2 if outcome.inconclusive else 1)

    # ``full`` moves real user traffic to the candidate (they have survived
    # 50/50 already), so the Phase 6 offline gate must be clean first. It is
    # the only code that flips the Production alias; on a non-zero exit the
    # nginx conf is left untouched and the alias is still the old version.
    if to == "full":
        _run_offline_gate(state.candidate_version)

    _write_text(conf_path, render_canary_conf(to))
    _reload_nginx(compose_file)

    state.last_gate = report
    state.edges.append({"from": state.phase, "to": to, "at": report["evaluated_at"]})
    state.phase = to
    state.entered_at = datetime.now(UTC).isoformat(timespec="seconds")
    _save_state(state, state_path)
    logger.info(
        f"advanced {_MODEL} to {to} (roadmap: {[e['to'] for e in state.edges]})"
    )


def cmd_rollback(
    client: mlflow.tracking.MlflowClient,
    *,
    conf_path: Path,
    compose_file: str,
    state_path: Path,
) -> None:
    """Stable-only weights instantly, then return the alias to the pre-rollout version."""
    state = _load_state(state_path)
    if state.phase == "rolled-back":
        raise SystemExit(f"rollback already completed (state: {state_path})")
    _health_gates((_STABLE_URL, _CANDIDATE_URL, _FRONT_URL, _INGRESS_URL, _AGENT_URL))

    # Traffic first, registry after: the old graph is the one the service has
    # served all along, so a conf change alone is a working rollback; the alias
    # flip only matters to a freshly-started worker.
    _write_text(conf_path, render_canary_conf("rollback"))
    _reload_nginx(compose_file)

    target = state.previous_production_version
    if target is None:
        logger.warning(
            "no pre-rollout Production version on record; falling back to the int8 variant"
        )
        target = _int8_version(client)
    _flip_production(client, target)

    state.phase = "rolled-back"
    _save_state(state, state_path)
    logger.info(
        f"rolled back: stable-only weights live, '{_STAGE_PRODUCTION}' alias -> v{target}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["declare", "shadow", "advance", "rollback"],
        required=True,
    )
    parser.add_argument(
        "--to",
        choices=list(_PHASES[1:]),
        default=None,
        help="target stage for --mode advance",
    )
    parser.add_argument("--experiment", default=Experiments.TRAINING.value)
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--conf-path", default=_DEFAULT_CONF_PATH)
    parser.add_argument("--compose-file", default=_DEFAULT_COMPOSE_FILE)
    parser.add_argument("--state-file", default=_DEFAULT_STATE_PATH)
    parser.add_argument("--prometheus-url", default="http://127.0.0.1:9090")
    parser.add_argument("--window", default="5m")
    parser.add_argument("--min-requests", type=int, default=500)
    parser.add_argument("--hold-seconds", type=int, default=600)
    parser.add_argument("--agreement-min", type=float, default=0.99)
    parser.add_argument("--error-rate-max", type=float, default=0.005)
    parser.add_argument("--p95-ratio-max", type=float, default=1.10)
    args = parser.parse_args()

    if args.mode == "advance" and not args.to:
        parser.error("--mode advance requires --to canary-5|canary-25|canary-50|full")

    load_environment()
    tracking_uri = (
        args.tracking_uri
        if args.tracking_uri
        else mlflow_tracking_uri(default="file:./mlruns")
    )
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    conf_path = Path(args.conf_path)
    state_path = Path(args.state_file)

    if args.mode == "declare":
        registered = declare(client, args.experiment)
        print(f"Declared canary: {_MODEL} v{registered.version} (alias 'Canary')")
    elif args.mode == "shadow":
        cmd_shadow(
            client,
            conf_path=conf_path,
            compose_file=args.compose_file,
            state_path=state_path,
            prometheus_url=args.prometheus_url,
        )
    elif args.mode == "advance":
        cmd_advance(
            client,
            to=args.to,
            conf_path=conf_path,
            compose_file=args.compose_file,
            state_path=state_path,
            window=args.window,
            min_requests=args.min_requests,
            hold_seconds=args.hold_seconds,
            agreement_min=args.agreement_min,
            error_rate_max=args.error_rate_max,
            p95_ratio_max=args.p95_ratio_max,
        )
    elif args.mode == "rollback":
        cmd_rollback(
            client,
            conf_path=conf_path,
            compose_file=args.compose_file,
            state_path=state_path,
        )


if __name__ == "__main__":
    main()
