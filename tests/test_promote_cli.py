"""Promotion report and command-line behavior tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from promote_model import (
    Config,
    build_payload,
    evaluate_all_gates,
    file_md5,
    main,
    promote,
    report_path,
    write_report,
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


@pytest.fixture
def live_registry(monkeypatch):
    """A registry the CLI path can actually reach.

    Every other CLI test passes ``--skip-registry``, which is precisely why a
    broken ``--dry-run`` went unnoticed: the flag is only meaningful with a
    client attached, and no test attached one.
    """
    import mlflow

    from raay.config import env as raay_env

    registry = FakeRegistry(production="4")
    monkeypatch.setattr(raay_env, "load_environment", lambda: None)
    monkeypatch.setattr(
        raay_env, "mlflow_tracking_uri", lambda: "http://unused.invalid"
    )
    monkeypatch.setattr(mlflow.tracking, "MlflowClient", lambda: registry)
    return registry


def test_payload_records_provenance_and_thresholds(cfg: Config) -> None:
    gates = evaluate_all_gates(cfg, metrics(), metrics(), baseline_report())
    payload = build_payload(
        cfg, "7", metrics(), metrics(), baseline_report(), gates, False, "4"
    )
    assert payload["registered_model"] == "ArabicSentiment"
    assert payload["candidate_version"] == "7"
    assert payload["test_split"]["md5"] == file_md5(cfg.test_split)
    assert payload["test_split"]["used_for_training"] is False
    assert payload["thresholds"]["floor_tolerance"] == 0.01
    assert payload["latency_noise"]["allowed_regression_pct"] == 10.0
    assert "resolvable_on_this_host" in payload["latency_noise"]
    # The reported figure has to be the one the measurement produced, not a
    # placeholder: nulling it in the payload would otherwise go unnoticed.
    prod = metrics(
        latency={
            "p50_ms": 10.0,
            "p95_ms": 40.0,
            "runs": 50,
            "noise_pct": 25.0,
            "control_gap_pct": 3.0,
            "control_p95_ms": 41.2,
        }
    )
    payload = build_payload(
        cfg, "7", metrics(), prod, baseline_report(), gates, False, "4"
    )
    assert payload["latency_noise"]["same_graph_noise_pct"] == 25.0
    assert payload["latency_noise"]["control_gap_pct"] == 3.0
    assert payload["latency_noise"]["resolvable_on_this_host"] is False
    assert payload["latency_noise"]["series_p95_ms"] == {
        "candidate": 27.5,
        "production": 40.0,
        "control": 41.2,
    }
    assert payload["latency_noise"]["control_gap_pct"] == 3.0
    assert payload["latency_noise"]["resolvable_on_this_host"] is False
    assert payload["latency_noise"]["series_p95_ms"]["production"] == 40.0
    assert "same machine" in payload["caveat"] or "this gate" in payload["caveat"]


def test_report_is_written_and_named_per_version(cfg: Config) -> None:
    gates = evaluate_all_gates(cfg, metrics(), metrics(), baseline_report())
    path = report_path(cfg, "7")
    write_report(
        path,
        build_payload(
            cfg, "7", metrics(), metrics(), baseline_report(), gates, True, "4"
        ),
    )
    assert path.name == "promotion_7.json"
    assert json.loads(path.read_text())["decision"] == "promoted"


def test_cli_exits_zero_when_gates_pass(repo: Path, capsys) -> None:
    patch_eval(metrics(f1_macro=0.6450), cand_paths=(str(repo / "cand.onnx"),))
    assert main(_cli_args(repo)) == 0
    assert "PASSED (not promoted)" in capsys.readouterr().out
    assert (repo / "reports" / "promotion_7.json").exists()


def test_cli_exits_nonzero_on_a_failed_gate(repo: Path, capsys) -> None:
    patch_eval(metrics(f1_macro=0.30), cand_paths=(str(repo / "cand.onnx"),))
    assert main(_cli_args(repo)) == 1
    out = capsys.readouterr().out
    assert "REJECTED" in out
    assert "the Production alias was not touched" in out
    report = json.loads((repo / "reports" / "promotion_7.json").read_text())
    assert report["decision"] == "rejected"


def test_cli_dry_run_does_not_promote(repo: Path, live_registry) -> None:
    """End to end through main(): the flag the workflow passes must be honoured."""
    patch_eval(metrics(f1_macro=0.6450), cand_paths=(str(repo / "cand.onnx"),))
    args = [a for a in _cli_args(repo) if a != "--skip-registry"] + ["--dry-run"]
    assert main(args) == 0
    assert live_registry.sets("Candidate") == ["7"]
    assert live_registry.sets("Production") == []
    report = json.loads((repo / "reports" / "promotion_7.json").read_text())
    assert report["promoted"] is False
    assert report["dry_run"] is True
    assert report["promotion_blocked_reason"] == "dry_run"


def test_cli_without_dry_run_does_promote(repo: Path, live_registry) -> None:
    """The control for the test above, so the guard cannot be a no-op.

    If this failed, the previous test would pass for the wrong reason -- a
    registry that never promotes at all rather than one that honours the flag.
    """
    patch_eval(metrics(f1_macro=0.6450), cand_paths=(str(repo / "cand.onnx"),))
    args = [a for a in _cli_args(repo) if a != "--skip-registry"]
    assert main(args) == 0
    assert live_registry.sets("Production") == ["7"]


def test_cli_reports_a_broken_environment_clearly(repo: Path, capsys) -> None:
    """A missing artifact is exit 2 with a message, not a traceback.

    Exit 1 means the gates ran and rejected the model, which is a decision.
    Exit 2 means the gate could not produce a decision at all, and CI needs to
    tell those apart: one is a model problem, the other is a broken pipeline.
    """
    patch_eval(metrics(f1_macro=0.6450))
    (repo / "parity.json").unlink()
    assert main(_cli_args(repo)) == 2
    assert "could not run" in capsys.readouterr().out
    assert not (repo / "reports" / "promotion_7.json").exists()


def test_unpinned_split_is_tolerated_but_noticed(repo: Path, monkeypatch) -> None:
    """A repo with no dvc.lock entry gets a *failing* pin gate, not a crash."""
    (repo / "dvc.lock").write_text("stages: {}\n")
    patch_eval(metrics(f1_macro=0.6450))
    registry = FakeRegistry()
    decision = promote(
        Config(
            test_split=repo / "test.csv",
            floor_report=repo / "floor.json",
            parity_report=repo / "parity.json",
            dvc_lock=repo / "dvc.lock",
            report_dir=repo / "reports",
        ),
        "7",
        CAND_ONNX,
        PROD_ONNX,
        client=registry,
    )
    assert "frozen_test_split" in decision.payload["failed_gates"]
    assert registry.sets("Production") == []


def test_a_drifted_split_is_exit_two_with_no_report(repo: Path, capsys) -> None:
    patch_eval(metrics(f1_macro=0.6450), cand_paths=(str(repo / "cand.onnx"),))
    repo.joinpath("test.csv").write_text("text\nمراجعة\n")
    assert main(_cli_args(repo)) == 2
    out = capsys.readouterr().out
    assert "does not match" in out
    assert "locked" in out
    assert not (repo / "reports" / "promotion_7.json").exists(), (
        "a report here would look like a verdict on the model"
    )
