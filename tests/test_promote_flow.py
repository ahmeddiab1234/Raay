"""Promotion sweep and registry flow tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import promote_model as pm
import promotion_graph
import promotion_split
import pytest
from promote_model import (
    Config,
    evaluate_all_gates,
    format_table,
    promote,
    trigger_report_path,
    trigger_version_tags,
)
from promotion_helpers import (
    CAND_ONNX,
    PROD_ONNX,
    FakeRegistry,
    baseline_report,
    make_cfg,
    make_repo,
    metrics,
    patch_eval,
    restore_eval,
)


@pytest.fixture
def repo(tmp_path):
    return make_repo(tmp_path)


@pytest.fixture
def cfg(repo):
    return make_cfg(repo)


@pytest.fixture(autouse=True)
def _restore_eval():
    with restore_eval():
        yield


def _write_trigger(cfg: Config, day: str = "2026-10-01", **over: object) -> None:
    payload = {
        "date": day,
        "reason": "psi_breach",
        "triggered": True,
        "psi": {
            "fired": True,
            "worst": {"drift_score": 0.44, "column": "positive"},
        },
        "calendar": {"fired": False},
    }
    payload.update(over)
    directory = cfg.report_dir / "retrain_trigger"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{day}.json").write_text(json.dumps(payload))


def _cli_args(repo: Path, *extra: str) -> list[str]:
    return [
        "--candidate-version",
        "7",
        "--candidate-onnx",
        str(repo / "cand.onnx"),
        "--production-onnx",
        str(repo / "prod.onnx"),
        "--test-split",
        str(repo / "test.csv"),
        "--floor-report",
        str(repo / "floor.json"),
        "--parity-report",
        str(repo / "parity.json"),
        "--report-dir",
        str(repo / "reports"),
        "--dvc-lock",
        str(repo / "dvc.lock"),
        "--skip-registry",
        *extra,
    ]


def test_all_gates_pass_for_a_clean_candidate(cfg: Config) -> None:
    cand = metrics(f1_macro=0.6450, accuracy=0.8520)
    gates = evaluate_all_gates(cfg, cand, metrics(), baseline_report())
    assert [gate.name for gate in gates if not gate.passed] == []


def test_all_gates_name_the_failure(cfg: Config) -> None:
    gates = evaluate_all_gates(
        cfg, metrics(f1_macro=0.30), metrics(), baseline_report()
    )
    failed = {gate.name for gate in gates if not gate.passed}
    assert "f1_macro_not_regressed" in failed
    assert "f1_macro_above_floor" in failed


def test_format_table_lists_every_gate(cfg: Config) -> None:
    gates = evaluate_all_gates(cfg, metrics(), metrics(), baseline_report())
    table = format_table(gates)
    for gate in gates:
        assert gate.name in table
    assert "PASS" in table and "FAIL" not in table


def test_promote_flips_production_only_on_a_clean_sweep(cfg: Config) -> None:
    patch_eval(metrics(f1_macro=0.6450))
    registry = FakeRegistry(production="4")
    decision = promote(cfg, "7", CAND_ONNX, PROD_ONNX, client=registry)
    assert decision.promoted
    assert decision.payload["decision"] == "promoted"
    assert decision.payload["previous_production_version"] == "4"
    assert registry.sets("Candidate") == ["7"]
    assert registry.sets("Production") == ["7"]


def test_candidate_alias_is_set_before_evaluation(cfg: Config) -> None:
    """The candidate is inspectable under its own alias before it can serve."""
    patch_eval(metrics(f1_macro=0.6450))
    registry = FakeRegistry()
    promote(cfg, "7", CAND_ONNX, PROD_ONNX, client=registry)
    aliases = [call[2] for call in registry.calls if call[0] == "set"]
    assert aliases == ["Candidate", "Production"]


def test_trigger_version_tags_read_the_latest_report(cfg: Config) -> None:
    _write_trigger(cfg, "2026-09-30", reason="scheduled")
    _write_trigger(cfg, "2026-10-01", reason="psi_breach")
    tags = trigger_version_tags(cfg)
    assert tags["trigger_reason"] == "psi_breach"
    assert tags["trigger_date"] == "2026-10-01"
    assert tags["trigger_psi"] == "0.44"
    assert tags["trigger_psi_column"] == "positive"


def test_trigger_version_tags_absent_is_empty(cfg: Config) -> None:
    assert trigger_report_path(cfg) is None
    assert trigger_version_tags(cfg) == {}


def test_a_clean_night_report_does_not_tag_the_version(cfg: Config) -> None:
    # `reason: none` is a non-trigger. Tagging it would assert a drift-motivated
    # promotion that never happened, so it must yield no tags -- the same as an
    # absent report.
    _write_trigger(cfg, reason="none", triggered=False, psi={"fired": False})
    assert trigger_version_tags(cfg) == {}


def test_unreadable_trigger_report_yields_no_tags(cfg: Config) -> None:
    _write_trigger(cfg)
    path = trigger_report_path(cfg)
    assert path is not None
    path.write_text("{ not json")
    assert trigger_version_tags(cfg) == {}


def test_promotion_stamps_the_trigger_reason_on_the_version(cfg: Config) -> None:
    _write_trigger(cfg)
    patch_eval(metrics(f1_macro=0.6450))
    registry = FakeRegistry(production="4")
    decision = promote(cfg, "7", CAND_ONNX, PROD_ONNX, client=registry)
    assert decision.promoted
    tags = registry.tags_for("7")
    assert tags["trigger_reason"] == "psi_breach"
    assert decision.payload["trigger_tags_applied"]
    assert "trigger_reason" in decision.payload["trigger_tags_applied"]


def test_a_rejected_candidate_is_never_tagged(cfg: Config) -> None:
    # Tags describe served history; a version that never moved must not be
    # stamped as if it had.
    _write_trigger(cfg)
    patch_eval(metrics(f1_macro=0.30, accuracy=0.60))
    registry = FakeRegistry(production="4")
    decision = promote(cfg, "7", CAND_ONNX, PROD_ONNX, client=registry)
    assert not decision.promoted
    assert registry.tags_for("7") == {}


def test_dry_run_does_not_stamp_tags(cfg: Config) -> None:
    _write_trigger(cfg)
    patch_eval(metrics(f1_macro=0.6450))
    registry = FakeRegistry(production="4")
    promote(cfg, "7", CAND_ONNX, PROD_ONNX, client=registry, dry_run=True)
    assert registry.tags_for("7") == {}


def test_promotion_without_a_trigger_report_is_untagged(cfg: Config) -> None:
    patch_eval(metrics(f1_macro=0.6450))
    registry = FakeRegistry(production="4")
    decision = promote(cfg, "7", CAND_ONNX, PROD_ONNX, client=registry)
    assert decision.promoted
    assert registry.tags_for("7") == {}
    assert decision.payload["trigger_tags_applied"] == []


def test_failed_candidate_never_reaches_production(cfg: Config) -> None:
    patch_eval(metrics(f1_macro=0.30, accuracy=0.60))
    registry = FakeRegistry(production="4")
    decision = promote(cfg, "7", CAND_ONNX, PROD_ONNX, client=registry)
    assert not decision.promoted
    assert decision.payload["decision"] == "rejected"
    assert registry.sets("Production") == []
    assert registry.aliases["Production"] == "4"  # untouched
    assert "f1_macro_not_regressed" in decision.payload["failed_gates"]


def test_neutral_collapse_blocks_promotion(cfg: Config) -> None:
    """Same macro F1, same accuracy, Neutral gone -- and the gate still says no."""
    cand = metrics(
        per_class={
            **metrics()["per_class"],
            "neutral": {"recall": 0.0, "precision": 0.0, "f1": 0.0, "support": 409},
        }
    )
    patch_eval(cand)
    registry = FakeRegistry()
    decision = promote(cfg, "7", CAND_ONNX, PROD_ONNX, client=registry)
    assert not decision.promoted
    assert "recall_neutral_absolute_floor" in decision.payload["failed_gates"]


def test_a_dry_run_with_a_live_registry_still_does_not_flip_production(
    cfg: Config,
) -> None:
    """The measuring job has a real client and must still be powerless.

    This is the workflow's ``gate`` job. It needs a registry connection to
    resolve and stage the candidate graph, so it cannot be kept away from
    Production the way ``--skip-registry`` does by withholding a client -- the
    guard has to be the flag itself. While the flag was ignored, that job
    promoted before anyone approved anything, and the whole suite still passed.
    """
    patch_eval(metrics(f1_macro=0.6450))
    registry = FakeRegistry(production="4")
    decision = promote(cfg, "7", CAND_ONNX, PROD_ONNX, client=registry, dry_run=True)
    assert registry.sets("Candidate") == ["7"]
    assert registry.sets("Production") == []
    assert registry.aliases["Production"] == "4"
    assert not decision.promoted
    assert decision.payload["decision"] == "passed_not_promoted"
    assert decision.payload["promotion_blocked_reason"] == "dry_run"
    assert decision.payload["dry_run"] is True


def test_a_report_says_whether_the_promotion_was_withheld(cfg: Config) -> None:
    """A passed rehearsal and a real promotion must not read the same.

    A human approving a deployment reads this report, and the second job
    re-measures before it flips. If both a deliberately withheld promotion and
    a real one said only "passed", the report could not say whether the flip
    happened -- the one fact the approver needs.
    """
    patch_eval(metrics(f1_macro=0.6450))
    withheld = promote(cfg, "7", CAND_ONNX, PROD_ONNX, client=None)
    assert withheld.payload["promotion_blocked_reason"] == "no_registry_client"
    assert withheld.payload["dry_run"] is False

    real = promote(cfg, "7", CAND_ONNX, PROD_ONNX, client=FakeRegistry(production="4"))
    assert real.promoted
    assert real.payload["promotion_blocked_reason"] is None

    patch_eval(metrics(f1_macro=0.30, accuracy=0.60))
    rejected = promote(cfg, "7", CAND_ONNX, PROD_ONNX, client=None, dry_run=True)
    assert rejected.payload["decision"] == "rejected"
    assert rejected.payload["promotion_blocked_reason"] is None


def test_skip_registry_reports_passed_not_rejected(cfg: Config) -> None:
    patch_eval(metrics(f1_macro=0.6450))
    decision = promote(cfg, "7", CAND_ONNX, PROD_ONNX, client=None)
    assert not decision.promoted
    assert decision.payload["decision"] == "passed_not_promoted"
    assert decision.payload["failed_gates"] == []


def test_a_drifted_split_is_caught_before_anything_is_scored(
    cfg: Config, repo: Path, monkeypatch
) -> None:
    """The hash check must come first, not last.

    Scoring two graphs over 7,209 rows and then discovering the split was never
    the frozen one wastes the whole run and burns enough CPU to make the
    latency verdict meaningless. A drifted split is a broken pipeline, so it
    is exit 2 with no report -- not a rejection of the model.
    """
    called: list[str] = []

    def spy(cfg, onnx_path, frame):
        called.append(onnx_path)
        return pm.Graph(metrics=metrics())

    monkeypatch.setattr(promotion_graph, "load_graph", spy)
    monkeypatch.setattr(
        promotion_split, "load_split", lambda cfg: pd.DataFrame({"text": ["x"]})
    )
    repo.joinpath("test.csv").write_text("text\nمراجعة\n")

    with pytest.raises(pm.PromotionError) as excinfo:
        promote(cfg, "7", CAND_ONNX, PROD_ONNX, client=None)

    message = str(excinfo.value)
    assert "does not match" in message
    # Both hashes, or the operator debugging a drifted split has to go compute
    # one of them by hand.
    assert "observed md5" in message and "locked" in message
    assert called == [], "the graphs were scored against an unverified split"


def test_partial_evaluation_is_not_promotable_even_if_it_passes(
    cfg: Path, repo: Path
) -> None:
    patch_eval(metrics(f1_macro=0.6450))
    limited = Config(
        test_split=repo / "test.csv",
        floor_report=repo / "floor.json",
        parity_report=repo / "parity.json",
        dvc_lock=repo / "dvc.lock",
        report_dir=repo / "reports",
        eval_limit=800,
    )
    registry = FakeRegistry()
    decision = promote(limited, "7", CAND_ONNX, PROD_ONNX, client=registry)
    assert registry.sets("Production") == []
    assert "full_split_evaluated" in decision.payload["failed_gates"]


def test_first_promotion_survives_a_missing_production_alias(cfg: Config) -> None:
    """No Production alias yet is a first release, not a reason to crash."""
    patch_eval(metrics(f1_macro=0.6450))
    registry = FakeRegistry(production=None)
    decision = promote(cfg, "1", CAND_ONNX, PROD_ONNX, client=registry)
    assert decision.promoted
    assert decision.payload["previous_production_version"] is None
