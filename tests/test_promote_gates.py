"""Each promotion gate in isolation: the frozen split, macro-F1, accuracy,
per-class recall, latency, size, ONNX parity, label order and full-split
coverage.

Split out of the original 1387-line ``test_promote_model.py``. Each gate gets a
pass and a fail here; the *combination* (does the sweep hold, does a rejected
candidate stay out of Production) is in ``test_promote_flow.py``.

The recall and F1-floor numbers are the real recorded ones -- Production scores
0.6369 macro-F1 against an ``eval_baseline`` floor of 0.6407 -- so a tolerance
change that would reject the model already serving fails here.
"""

import json
from pathlib import Path

import pytest
from promote_model import (
    Config,
    PromotionError,
    check_frozen_split,
    file_md5,
    gate_accuracy,
    gate_f1_macro,
    gate_f1_macro_floor,
    gate_full_split,
    gate_label_order,
    gate_latency,
    gate_parity,
    gate_recall,
    gate_size,
    locked_test_split_md5,
    read_parity,
)
from promotion_helpers import (
    baseline_report,
    make_cfg,
    make_repo,
    metrics,
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


# --- the frozen split --------------------------------------------------------


def test_locked_test_split_md5_reads_dvc_lock(repo: Path) -> None:
    assert locked_test_split_md5(repo / "dvc.lock") == file_md5(repo / "test.csv")


def test_locked_test_split_md5_none_when_unpinned(tmp_path: Path) -> None:
    (tmp_path / "dvc.lock").write_text("stages:\n  split:\n    outs: []\n")
    assert locked_test_split_md5(tmp_path / "dvc.lock") is None


def test_frozen_split_passes_when_md5_matches(cfg: Config) -> None:
    result = check_frozen_split(cfg)
    assert result.passed
    assert result.observed == result.threshold


def test_frozen_split_fails_on_drift(cfg: Config) -> None:
    cfg.test_split.write_text("text,label\nمراجعة,0\n")
    result = check_frozen_split(cfg)
    assert not result.passed
    assert "drifted" in result.detail


def test_frozen_split_errors_when_split_missing(cfg: Config) -> None:
    cfg.test_split.unlink()
    with pytest.raises(PromotionError, match="dvc repro"):
        check_frozen_split(cfg)


def test_frozen_split_tolerable_when_lock_absent(tmp_path: Path) -> None:
    cfg = Config(
        test_split=tmp_path / "test.csv",
        dvc_lock=tmp_path / "absent.lock",
        require_frozen_split=False,
    )
    assert check_frozen_split(cfg).passed


# --- f1 / accuracy -----------------------------------------------------------


def test_f1_gate_allows_a_small_regression() -> None:
    cand, prod = metrics(f1_macro=0.6340), metrics(f1_macro=0.6369)
    assert gate_f1_macro(cand, prod, Config()).passed


def test_f1_gate_blocks_a_real_regression() -> None:
    cand, prod = metrics(f1_macro=0.6000), metrics(f1_macro=0.6369)
    result = gate_f1_macro(cand, prod, Config())
    assert not result.passed
    assert result.threshold == pytest.approx(0.6319)


def test_f1_gate_rejects_macro_f1_collapse_behind_good_accuracy() -> None:
    """The failure mode the gate exists for.

    A model that predicts ``positive`` for everything scores ~0.85 accuracy on
    this split -- identical to the real model -- while macro F1 collapses.
    """
    cand = metrics(accuracy=0.8520, f1_macro=0.42)
    result = gate_f1_macro(cand, metrics(), Config())
    assert cand["accuracy"] > metrics()["accuracy"]
    assert not result.passed


def test_f1_floor_accepts_the_current_production_graph() -> None:
    """Calibration: INT8 (0.6369) is 0.0038 under the baseline (0.6407).

    With ``floor_tolerance=0.0`` the model currently serving would be rejected
    by its own gate, which is why the default is 0.01. If this test ever fails,
    the floor moved and needs a human decision, not a new number.
    """
    baseline = baseline_report()
    cand = metrics(f1_macro=0.6369)
    assert gate_f1_macro_floor(cand, baseline, Config()).passed
    strict = Config(floor_tolerance=0.0)
    assert not gate_f1_macro_floor(cand, baseline, strict).passed


def test_f1_floor_blocks_the_distilled_graph() -> None:
    distilled = metrics(f1_macro=0.5975)
    result = gate_f1_macro_floor(distilled, baseline_report(), Config())
    assert not result.passed


def test_accuracy_gate_blocks_regression() -> None:
    assert not gate_accuracy(metrics(accuracy=0.70), metrics(), Config()).passed
    assert gate_accuracy(metrics(accuracy=0.8450), metrics(), Config()).passed


# --- per-class recall --------------------------------------------------------


def test_recall_gate_passes_an_improvement() -> None:
    gates = gate_recall(metrics(), metrics(), Config())
    assert gates
    assert all(gate.passed for gate in gates)


def test_recall_gate_blocks_loss_of_negative() -> None:
    cand = metrics(
        per_class={
            **metrics()["per_class"],
            "negative": {"precision": 0.4, "recall": 0.50, "f1": 0.44, "support": 1300},
        }
    )
    names = {gate.name: gate for gate in gate_recall(cand, metrics(), Config())}
    assert not names["recall_negative"].passed


def test_recall_absolute_floor_catches_neutral_collapse() -> None:
    """Relative-only comparison lets Neutral rot forever, one 2% step at a time."""
    prod = metrics(
        per_class={
            **metrics()["per_class"],
            "neutral": {"recall": 0.05, "precision": 0.1, "f1": 0.1, "support": 409},
        }
    )
    cand = metrics(
        per_class={
            **metrics()["per_class"],
            "neutral": {"recall": 0.04, "precision": 0.1, "f1": 0.1, "support": 409},
        }
    )
    names = {gate.name: gate for gate in gate_recall(cand, prod, Config())}
    assert names["recall_neutral"].passed  # only 1% worse
    assert not names["recall_neutral_absolute_floor"].passed


def test_recall_gate_flags_an_unknown_class() -> None:
    prod = metrics()
    prod["per_class"].pop("neutral")
    names = {gate.name: gate for gate in gate_recall(metrics(), prod, Config())}
    assert not names["recall_neutral"].passed
    assert "no 'neutral' class" in names["recall_neutral"].detail


def test_recall_gate_absent_floor_is_not_watched() -> None:
    gates = gate_recall(metrics(), metrics(), Config())
    assert not any(
        gate.name.endswith("_absolute_floor")
        for gate in gates
        if "positive" in gate.name
    )


# --- latency / size / parity -------------------------------------------------


def test_latency_gate_allows_ten_percent() -> None:
    cand = metrics(latency={"p50_ms": 13.0, "p95_ms": 30.0, "runs": 30})
    assert gate_latency(cand, metrics(), Config()).passed


def test_latency_gate_blocks_a_slowdown() -> None:
    cand = metrics(latency={"p50_ms": 40.0, "p95_ms": 60.0, "runs": 30})
    result = gate_latency(cand, metrics(), Config())
    assert not result.passed
    assert result.threshold == pytest.approx(30.25)


def test_size_gate_defaults_to_production_size() -> None:
    prod = metrics(size_mb=100.0)
    assert gate_size(metrics(size_mb=100.0), prod, Config()).passed
    assert not gate_size(metrics(size_mb=180.0), prod, Config()).passed


def test_size_gate_respects_an_explicit_limit() -> None:
    cfg = Config(max_size_mb=250.0)
    assert gate_size(metrics(size_mb=180.0), metrics(size_mb=100.0), cfg).passed


def test_parity_gate_reads_a_flat_report(cfg: Config) -> None:
    assert gate_parity(cfg).passed


def test_parity_gate_reads_a_nested_report(tmp_path: Path) -> None:
    (tmp_path / "p.json").write_text(json.dumps({"int8_parity": {"max_abs_diff": 0.2}}))
    assert gate_parity(Config(parity_report=tmp_path / "p.json")).passed


def test_parity_gate_blocks_label_disagreement(cfg: Config) -> None:
    cfg.parity_report.write_text(
        json.dumps({"max_abs_diff": 0.1, "label_agreement": 0.94})
    )
    assert not gate_parity(cfg).passed


def test_parity_gate_blocks_an_unquantized_graph(cfg: Config) -> None:
    cfg.parity_report.write_text(
        json.dumps({"max_abs_diff": 0.91, "label_agreement": 1.0})
    )
    assert not gate_parity(cfg).passed


def test_parity_gate_requires_the_report(cfg: Config) -> None:
    cfg.parity_report.unlink()
    with pytest.raises(PromotionError, match="quantize_onnx"):
        read_parity(cfg)


# --- label order / partial evaluations --------------------------------------


def test_label_order_gate_blocks_a_mismatch() -> None:
    swapped = metrics(label_names=["negative", "positive", "neutral"])
    assert not gate_label_order(swapped, metrics()).passed
    assert gate_label_order(metrics(), metrics()).passed


def test_partial_evaluation_can_never_promote(cfg: Config) -> None:
    limited = Config(eval_limit=800)
    assert not gate_full_split(limited, metrics(sample_size=800)).passed
    assert gate_full_split(cfg, metrics()).passed
