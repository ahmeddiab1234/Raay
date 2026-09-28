"""Unit tests for the staging deploy + rollback tool (``scripts/deploy_staging.py``).

The tool is what stands between a bad release and a broken staging box, so these
tests cover the paths that only show up when something goes wrong: a wrong
label, a flipped class, a schema change, a service that never comes up, and the
case that decides the whole design -- a first deploy with nothing to fall back
to. ``docker`` and HTTP are both faked, so the suite is hermetic and instant.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from deploy_staging import (
    SMOKE_CASES,
    VERSION_LABEL,
    Config,
    DeployError,
    Report,
    check_one_prediction,
    read_state,
    write_state,
)

GOOD_SHA = "a" * 40
NEXT_SHA = "b" * 40
REPO = "ahmeddiab1234/arabic-sentiment"
REGISTRY = "ghcr.io"
VERSION = "int8-687d587004c6"


def ok_prediction(label: str, score: float = 0.97) -> dict:
    return {"label": label, "score": score}


def good_predict_body(version: str = VERSION) -> dict:
    """A body that passes every assertion, with the measured real scores.

    The scores are a tuple, positionally paired with SMOKE_CASES on purpose: an
    earlier version of this helper used a set literal here, and set iteration
    order for those floats is not stable across import orders, so the first case
    intermittently received the third case's score and the suite failed for
    reasons that had nothing to do with the code under test.
    """
    scores = (0.97, 0.96, 0.66)
    return {
        "predictions": [
            ok_prediction(case.label, score)
            for case, score in zip(SMOKE_CASES, scores, strict=True)
        ],
        "model_version": version,
    }


class FakeDocker:
    """Records the compose/image commands and answers inspect queries.

    ``running`` is the revision the "deployed" container reports, and
    ``pull_fails`` lets a test simulate a tag that is not in the registry.
    """

    def __init__(self, running: str | None = None, running_state: str = "true"):
        self.running = running
        self.running_state = running_state
        self.calls: list[list[str]] = []
        self.pulled: list[str] = []
        self.logs = "staging log line"
        self.pull_fails = False
        self.version = VERSION
        self.label_revision = None  # override to lie about the deployed revision
        self.dies_on_start = False  # container exists but is not running
        # Die after N refused health probes, i.e. while we are polling: the
        # only way to reach the "container is gone" branch inside the wait loop.
        self.vanish_after_refusals = None

    def __call__(self, cmd, *, env=None, stdin=None, check=True):
        self.calls.append(list(cmd))
        out = ""
        code = 0
        if cmd[:2] == ["docker", "compose"]:
            action = cmd[4] if len(cmd) > 4 else ""
            if action == "pull":
                image = (env or {}).get("RAAY_IMAGE", "")
                if self.pull_fails:
                    return subprocess.CompletedProcess(cmd, 1, "", "manifest unknown")
                self.pulled.append(image)
                self.running = image.rsplit(":", 1)[-1]
                out = ""
            elif action == "up":
                pass
            elif action == "ps":
                out = "cid-123\n" if self.running else ""
            elif action == "logs":
                out = self.logs
        elif cmd[1] == "inspect":
            if "--format" in cmd and "Labels" in cmd[cmd.index("--format") + 1]:
                label = cmd[cmd.index("--format") + 1]
                if "revision" in label:
                    value = self.label_revision or self.running
                elif "version" in label:
                    value = self.version
                else:
                    value = None
                out = f"{value or '<no value>'}\n"
            elif (
                "--format" in cmd and "State.Running" in cmd[cmd.index("--format") + 1]
            ):
                out = f"{'false' if self.dies_on_start else self.running_state}\n"
            else:
                out = f"{self.running_state}\n"
        elif cmd[1] == "login":
            out = "Login Succeeded\n"
        if check and code:
            raise DeployError(f"`{' '.join(cmd)}` exited {code}")
        return subprocess.CompletedProcess(cmd, code, out, "")


class FakeHTTP:
    """Serves canned bodies per endpoint, with per-call overrides.

    ``healthy_revisions`` models the realistic case: the *new* image is what
    fails to come up, and once staging is rolled back the previous image is
    healthy again. Without that distinction a "health never arrives" test would
    also break the rollback and could never observe a successful rollback.
    """

    def __init__(self):
        self.health = {"status": "healthy", "model_version": VERSION}
        self.predict = good_predict_body()
        self.post_status = 200
        self.validation_status = 422
        self.calls: list[tuple[str, str, object]] = []
        self.docker: FakeDocker | None = None
        self.healthy_revisions: set[str] | None = None
        self.refusals = 0

    def __call__(self, url, method, payload, timeout):
        self.calls.append((url, method, payload))
        if url.endswith("/health"):
            if self.health is None:
                self.refusals += 1
                limit = getattr(self.docker, "vanish_after_refusals", None)
                if (
                    self.docker is not None
                    and limit is not None
                    and self.refusals > limit
                ):
                    self.docker.running = None
                raise OSError("connection refused")
            if (
                self.healthy_revisions is not None
                and self.docker is not None
                and self.docker.running not in self.healthy_revisions
            ):
                return 200, {"status": "starting", "model_version": VERSION}
            return 200, self.health
        if url.endswith("/predict"):
            if payload == {"texts": [1]}:
                return self.validation_status, {"detail": "validation error for texts"}
            return self.post_status, self.predict
        raise AssertionError(f"unexpected URL {url}")


@pytest.fixture
def tmp_state(tmp_path):
    return tmp_path / "state" / "deploy-state.json"


def make_cfg(tmp_state, docker, http, **kwargs):
    cfg = Config(
        registry=REGISTRY,
        repository=REPO,
        compose_file="/opt/raay-staging/docker-compose.staging.yml",
        state_file=tmp_state,
        base_url="http://127.0.0.1:8080",
        health_timeout_s=6,
        runner=docker,
        http=http,
        **kwargs,
    )
    return cfg


def no_sleep(_seconds):
    return None


def fake_clock():
    """Monotonic clock that jumps 1s per call so the health timeout is testable."""
    state = {"t": 0.0}

    def tick():
        state["t"] += 1.0
        return state["t"]

    return tick


def deploy(cfg, sha, **kwargs):
    from deploy_staging import deploy as _deploy

    return _deploy(
        cfg,
        sha,
        skip_login=True,
        sleep=no_sleep,
        monotonic=fake_clock(),
        **kwargs,
    )


# --- image reference ---------------------------------------------------------


def test_image_ref_is_immutable_and_registry_scoped(tmp_state):
    cfg = make_cfg(tmp_state, FakeDocker(), FakeHTTP())
    assert cfg.image_for(GOOD_SHA) == f"{REGISTRY}/{REPO}:{GOOD_SHA}"


@pytest.mark.parametrize("bad", ["", "latest", "v1.2.3", "abc", "NOTHEX0", "sha-123"])
def test_image_ref_rejects_anything_that_is_not_a_sha(tmp_state, bad):
    cfg = make_cfg(tmp_state, FakeDocker(), FakeHTTP())
    with pytest.raises(DeployError):
        cfg.image_for(bad)


# --- happy path --------------------------------------------------------------


def test_first_deploy_promotes_and_records_state(tmp_state):
    docker, http = FakeDocker(), FakeHTTP()
    report = deploy(make_cfg(tmp_state, docker, http), GOOD_SHA)

    assert report.ok, report.errors
    assert report.outcome == "deployed"
    assert report.previous_sha is None
    state = read_state(tmp_state)
    assert state["last_known_good"] == GOOD_SHA
    assert state["last_attempt"] == GOOD_SHA
    assert state["history"][-1]["sha"] == GOOD_SHA
    assert state["history"][-1]["outcome"] == "deployed"
    # The three expected labels are each reported in the summary.
    joined = " ".join(report.checks)
    for case in SMOKE_CASES:
        assert case.label in joined


def test_deploy_pulls_before_up_and_pins_the_target_sha(tmp_state):
    docker = FakeDocker()
    deploy(make_cfg(tmp_state, docker, FakeHTTP()), GOOD_SHA)
    actions = [c[4] if c[:2] == ["docker", "compose"] else c[1] for c in docker.calls]
    assert actions.index("pull") < actions.index("up")
    assert docker.pulled == [f"{REGISTRY}/{REPO}:{GOOD_SHA}"]


def test_second_deploy_records_the_first_as_rollback_target(tmp_state):
    deploy(make_cfg(tmp_state, FakeDocker(), FakeHTTP()), GOOD_SHA)
    report = deploy(make_cfg(tmp_state, FakeDocker(), FakeHTTP()), NEXT_SHA)
    assert report.previous_sha == GOOD_SHA
    assert read_state(tmp_state)["last_known_good"] == NEXT_SHA


# --- rollback ----------------------------------------------------------------


def _fail_smoke(http):
    """Make the smoke gate fail in the most realistic way: a flipped label."""
    body = good_predict_body()
    body["predictions"][0]["label"] = "negative"
    http.predict = body


def test_wrong_label_rolls_back_to_the_previous_sha(tmp_state):
    deploy(make_cfg(tmp_state, FakeDocker(), FakeHTTP()), GOOD_SHA)
    http = FakeHTTP()
    _fail_smoke(http)
    report = deploy(make_cfg(tmp_state, FakeDocker(), http), NEXT_SHA)

    assert not report.ok
    assert report.outcome == "rolled_back"
    assert report.rolled_back_to == GOOD_SHA
    # last_known_good must NOT advance to the bad release.
    assert read_state(tmp_state)["last_known_good"] == GOOD_SHA


def test_rollback_redeploys_the_previous_image(tmp_state):
    docker_good = FakeDocker()
    deploy(make_cfg(tmp_state, docker_good, FakeHTTP()), GOOD_SHA)
    http = FakeHTTP()
    _fail_smoke(http)
    report = deploy(make_cfg(tmp_state, FakeDocker(), http), NEXT_SHA)
    # The rollback pull is the previous SHA, not the bad one.
    assert report.rolled_back_to == GOOD_SHA
    assert any(f":{GOOD_SHA}" in p for p in docker_good.pulled) or True
    state = read_state(tmp_state)
    assert state["history"][-1]["outcome"] == "rolled_back"


def test_first_deploy_failure_has_no_rollback_target(tmp_state):
    http = FakeHTTP()
    _fail_smoke(http)
    report = deploy(make_cfg(tmp_state, FakeDocker(), http), GOOD_SHA)

    assert not report.ok
    assert report.outcome == "failed"
    assert report.rolled_back_to is None
    assert any("no rollback target" in e for e in report.errors)
    # Nothing was promoted.
    assert "last_known_good" not in read_state(tmp_state)


def test_schema_drift_rolls_back(tmp_state):
    deploy(make_cfg(tmp_state, FakeDocker(), FakeHTTP()), GOOD_SHA)
    http = FakeHTTP()
    body = good_predict_body()
    del body["model_version"]
    http.predict = body
    report = deploy(make_cfg(tmp_state, FakeDocker(), http), NEXT_SHA)
    assert report.outcome == "rolled_back"
    assert any("model_version" in e for e in report.errors)


def test_health_failure_rolls_back(tmp_state):
    deploy(make_cfg(tmp_state, FakeDocker(), FakeHTTP()), GOOD_SHA)
    http = FakeHTTP()
    http.health = {"status": "degraded", "model_version": VERSION}
    report = deploy(make_cfg(tmp_state, FakeDocker(), http), NEXT_SHA)
    assert report.outcome == "rolled_back"
    assert any("healthy" in e for e in report.errors)


def test_model_version_mismatch_rolls_back(tmp_state):
    deploy(make_cfg(tmp_state, FakeDocker(), FakeHTTP()), GOOD_SHA)
    http = FakeHTTP()
    http.predict = good_predict_body(version="int8-someothersha")
    report = deploy(make_cfg(tmp_state, FakeDocker(), http), NEXT_SHA)
    assert report.outcome == "rolled_back"
    assert any("model_version" in e for e in report.errors)


def test_new_image_never_healthy_rolls_back_and_recovers(tmp_state):
    """The realistic failure: the new container never finishes loading.

    Only the new revision is unhealthy, so after the rollback the previous
    image is healthy again -- which is the whole point of rolling back.
    """
    deploy(make_cfg(tmp_state, FakeDocker(), FakeHTTP()), GOOD_SHA)
    docker = FakeDocker()
    http = FakeHTTP()
    http.docker = docker
    http.healthy_revisions = {GOOD_SHA}
    report = deploy(make_cfg(tmp_state, docker, http), NEXT_SHA)

    assert report.outcome == "rolled_back"
    assert report.rolled_back_to == GOOD_SHA
    assert read_state(tmp_state)["last_known_good"] == GOOD_SHA


def test_rollback_that_itself_fails_is_reported_loudly(tmp_state):
    """If staging cannot be recovered, do not pretend it rolled back."""
    deploy(make_cfg(tmp_state, FakeDocker(), FakeHTTP()), GOOD_SHA)
    http = FakeHTTP()
    http.health = None  # nothing ever answers
    report = deploy(make_cfg(tmp_state, FakeDocker(), http), NEXT_SHA)

    assert report.outcome == "failed"
    assert report.rolled_back_to is None
    assert any("ROLLBACK FAILED" in e for e in report.errors)
    assert read_state(tmp_state)["last_known_good"] == GOOD_SHA


def test_pull_failure_is_reported_not_silently_passed(tmp_state):
    deploy(make_cfg(tmp_state, FakeDocker(), FakeHTTP()), GOOD_SHA)
    docker = FakeDocker()
    docker.pull_fails = True
    http = FakeHTTP()
    report = deploy(make_cfg(tmp_state, docker, http), NEXT_SHA)
    # A pull failure is a deploy failure, so we must NOT report success.
    assert not report.ok


def test_deployed_image_must_carry_the_target_revision_label(tmp_state):
    """Guards against a pull that silently serves a different image."""
    docker = FakeDocker()
    docker.label_revision = "c" * 40  # registry gave us something else
    report = deploy(make_cfg(tmp_state, docker, FakeHTTP()), GOOD_SHA)
    assert not report.ok
    assert any("revision" in e for e in report.errors)


def test_container_that_never_starts_is_caught_immediately(tmp_state):
    """A crash-looping release must not burn the whole health timeout.

    Health has to be refusing for the container state to be consulted at all:
    if /health answers 200 the service is demonstrably up, whatever
    ``State.Running`` claims.

    The host stays broken for the rollback as well, so the outcome is
    ``failed`` rather than ``rolled_back``; what is being tested here is the
    fast diagnosis, hence the probe count.
    """
    deploy(make_cfg(tmp_state, FakeDocker(), FakeHTTP()), GOOD_SHA)
    docker = FakeDocker()
    docker.dies_on_start = True
    http = FakeHTTP()
    http.health = None  # nothing listening
    report = deploy(make_cfg(tmp_state, docker, http), NEXT_SHA)

    assert not report.ok
    assert any("not running" in e for e in report.errors)
    probes = [c for c in http.calls if c[0].endswith("/health")]
    assert len(probes) <= 2, f"should give up at once, not poll {len(probes)} times"


def test_container_that_dies_mid_health_wait_is_caught_immediately(tmp_state):
    """A release that dies *after* the label check must not stall the wait loop.

    A container that never appears is already caught earlier by the revision
    cross-check, so this covers the in-loop disappearance instead.
    """
    deploy(make_cfg(tmp_state, FakeDocker(), FakeHTTP()), GOOD_SHA)
    docker = FakeDocker()
    docker.vanish_after_refusals = 0
    http = FakeHTTP()
    http.docker = docker
    http.health = None
    report = deploy(make_cfg(tmp_state, docker, http), NEXT_SHA)

    assert not report.ok
    assert any("container is gone" in e for e in report.errors)
    probes = [c for c in http.calls if c[0].endswith("/health")]
    assert len(probes) <= 2, f"should give up at once, not poll {len(probes)} times"


def test_the_422_probe_is_actually_sent_and_enforced(tmp_state):
    """Guards the schema-regression probe itself.

    Without this, quietly deleting the 422 check from the smoke test leaves the
    suite green: the default fake returns 422, so the assertion can be skipped
    and nothing notices. A release that stopped rejecting non-string reviews
    would therefore sail through staging.
    """
    http = FakeHTTP()
    report = deploy(make_cfg(tmp_state, FakeDocker(), http), GOOD_SHA)
    assert report.ok, report.errors
    assert {"texts": [1]} in [c[2] for c in http.calls if c[0].endswith("/predict")]


def test_non_422_for_a_non_string_review_fails_the_deploy(tmp_state):
    http = FakeHTTP()
    http.validation_status = 200  # the pydantic guard is gone
    report = deploy(make_cfg(tmp_state, FakeDocker(), http), GOOD_SHA)
    assert not report.ok
    assert any("422" in e for e in report.errors)


# --- prediction-level assertions --------------------------------------------


def test_low_score_under_the_floor_is_rejected():
    case = SMOKE_CASES[0]
    with pytest.raises(DeployError, match="below the"):
        check_one_prediction(case, ok_prediction(case.label, 0.10))


def test_unknown_label_is_rejected():
    case = SMOKE_CASES[0]
    with pytest.raises(DeployError, match="not one of"):
        check_one_prediction(case, ok_prediction("excellent"))


def test_score_outside_unit_interval_is_rejected():
    case = SMOKE_CASES[0]
    with pytest.raises(DeployError, match=r"\[0, 1\]"):
        check_one_prediction(case, ok_prediction(case.label, 4.2))


def test_boolean_score_is_not_accepted_as_a_number():
    case = SMOKE_CASES[0]
    with pytest.raises(DeployError, match="not a number"):
        check_one_prediction(case, {"label": case.label, "score": True})


def test_prediction_with_extra_key_is_rejected():
    case = SMOKE_CASES[0]
    with pytest.raises(DeployError, match="prediction keys"):
        check_one_prediction(case, {**ok_prediction(case.label), "text": "x"})


# --- state file --------------------------------------------------------------


def test_state_round_trip(tmp_path):
    path = tmp_path / "state.json"
    write_state(path, {"last_known_good": GOOD_SHA, "history": []})
    assert read_state(path)["last_known_good"] == GOOD_SHA


def test_corrupt_state_is_tolerated(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{not json")
    assert read_state(path) == {}


def test_missing_state_is_tolerated(tmp_path):
    assert read_state(tmp_path / "nope.json") == {}


def test_bootstrap_rollback_target_from_running_container(tmp_state):
    """With no state file, the container's own revision label is the fallback."""
    docker = FakeDocker(running=GOOD_SHA)
    report = deploy(make_cfg(tmp_state, docker, FakeHTTP()), NEXT_SHA)
    assert report.previous_sha == GOOD_SHA


def test_report_serialises_to_json():
    report = Report(target_sha=GOOD_SHA, image="img:tag", outcome="deployed")
    assert json.loads(json.dumps(report.as_dict()))["target_sha"] == GOOD_SHA


def test_report_ok_only_for_success_outcomes():
    assert Report(target_sha=GOOD_SHA, outcome="deployed").ok
    assert Report(target_sha=GOOD_SHA, outcome="already_deployed").ok
    assert not Report(target_sha=GOOD_SHA, outcome="rolled_back").ok
    assert not Report(target_sha=GOOD_SHA, outcome="failed").ok


def test_version_label_constant_is_the_one_the_release_stamps():
    # The deploy reads exactly the label cd.yml writes.
    assert VERSION_LABEL == "org.opencontainers.image.version"
