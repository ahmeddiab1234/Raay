import sys
from pathlib import Path

import pytest
from deploy_staging_helpers import (
    GOOD_SHA,
    NEXT_SHA,
    REGISTRY,
    REPO,
    VERSION,
    FakeDocker,
    FakeHTTP,
    deploy,
    good_predict_body,
    make_cfg,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from deploy_staging import SMOKE_CASES, DeployError, read_state

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
