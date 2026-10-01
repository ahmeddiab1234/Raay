import json
import sys
from pathlib import Path

import pytest
from deploy_staging_helpers import (
    GOOD_SHA,
    NEXT_SHA,
    FakeDocker,
    FakeHTTP,
    deploy,
    make_cfg,
    ok_prediction,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from deploy_staging import (
    SMOKE_CASES,
    VERSION_LABEL,
    DeployError,
    Report,
    check_one_prediction,
    read_state,
    write_state,
)

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
