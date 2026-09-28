"""Unit tests for the promotion gate (``scripts/promote_model.py``).

The gate decides whether a model is allowed to start serving traffic, so the
tests care about the failure paths rather than the happy one: a macro-F1
regression hidden behind a good accuracy, a Neutral class quietly collapsing,
a test split that has drifted from ``dvc.lock``, and -- the one that would be
most damaging -- a failed candidate reaching the ``Production`` alias anyway.

The graph evaluation is faked, so the suite is hermetic and instant. The real
thresholds are exercised against the real recorded numbers (``eval_baseline``
0.6407 vs the INT8 graph's 0.6369) because that pair is what the tolerances
were calibrated against, and a threshold change that quietly rejects the model
already in Production is exactly the regression these tests exist to catch.
"""

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import promote_model as pm
from promote_model import (
    Config,
    PromotionError,
    build_payload,
    check_frozen_split,
    evaluate_all_gates,
    file_md5,
    format_table,
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
    main,
    promote,
    read_parity,
    report_path,
    write_report,
)

LABELS = ["positive", "negative", "neutral"]
PROD_ONNX = "models/onnx/model_int8.onnx"
CAND_ONNX = "models/onnx/candidate.onnx"


def metrics(**over: object) -> dict:
    """A metrics block shaped exactly like ``evaluate_graph``'s output."""
    base: dict = {
        "onnx_path": PROD_ONNX,
        "size_mb": 129.81,
        "label_names": list(LABELS),
        "id2label": {"0": "positive", "1": "negative", "2": "neutral"},
        "accuracy": 0.8503,
        "f1_macro": 0.6369,
        "f1_weighted": 0.8612,
        "per_class": {
            "positive": {
                "precision": 0.93,
                "recall": 0.9331,
                "f1": 0.9315,
                "support": 5500,
            },
            "negative": {
                "precision": 0.852,
                "recall": 0.8599,
                "f1": 0.8559,
                "support": 1300,
            },
            "neutral": {
                "precision": 0.198,
                "recall": 0.123,
                "f1": 0.1553,
                "support": 409,
            },
        },
        "confusion_matrix": [[5000, 200, 300], [100, 1100, 100], [40, 40, 329]],
        "sample_size": 7209,
        "latency": {"p50_ms": 12.0, "p95_ms": 27.5, "runs": 30},
        "metadata": {"model_name": "aubmindlab/bert-base-arabertv02"},
        "dialect_breakdown": None,
    }
    base.update(over)
    return base


def baseline_report() -> dict:
    """``reports/eval_baseline.json`` as recorded, at full precision."""
    return {
        "model_dir": "models/baseline/final",
        "split": "test",
        "sample_size": 7209,
        "accuracy": 0.84917,
        "f1_macro": 0.6406636813692393,
        "f1_weighted": 0.86197,
        "label_names": list(LABELS),
        "per_class": {
            "positive": {
                "precision": 0.928,
                "recall": 0.9361,
                "f1": 0.932,
                "support": 5500,
            },
            "negative": {
                "precision": 0.855,
                "recall": 0.8577,
                "f1": 0.8564,
                "support": 1300,
            },
            "neutral": {
                "precision": 0.232,
                "recall": 0.1393,
                "f1": 0.1746,
                "support": 409,
            },
        },
        "dialect_breakdown": None,
    }


class FakeRegistry:
    """Stands in for ``MlflowClient`` and records the alias mutations."""

    def __init__(self, production: str | None = "4") -> None:
        self.production = production
        self.calls: list[tuple] = []
        self.aliases: dict[str, str] = {"Production": production} if production else {}

    def get_model_version_by_alias(self, name: str, alias: str) -> str | None:
        self.calls.append(("get", name, alias))
        if alias == "Production" and self.production is None:
            raise RuntimeError(f"Registered Model alias {alias} not found")
        return self.aliases.get(alias)

    def set_registered_model_alias(self, name: str, alias: str, version: str) -> None:
        self.calls.append(("set", name, alias, version))
        self.aliases[alias] = version

    def sets(self, alias: str) -> list[str]:
        return [call[3] for call in self.calls if call[0] == "set" and call[2] == alias]


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A miniature repo: split, dvc.lock, floor report, parity report, graphs."""
    split = tmp_path / "test.csv"
    split.write_text(
        "text,label\n" + "\n".join(f"مراجعة {i},{i % 3}" for i in range(30))
    )
    digest = file_md5(split)

    lock = tmp_path / "dvc.lock"
    lock.write_text(
        "stages:\n"
        "  split:\n"
        "    outs:\n"
        "    - path: data/processed/test.csv\n"
        f"      md5: {digest}\n"
    )
    (tmp_path / "floor.json").write_text(json.dumps(baseline_report()))
    (tmp_path / "parity.json").write_text(
        json.dumps({"n_samples": 6, "max_abs_diff": 0.4646, "label_agreement": 1.0})
    )
    (tmp_path / "prod.onnx").write_bytes(b"x" * 2048)
    (tmp_path / "cand.onnx").write_bytes(b"x" * 1024)
    return tmp_path


@pytest.fixture
def cfg(repo: Path) -> Config:
    return Config(
        test_split=repo / "test.csv",
        floor_report=repo / "floor.json",
        parity_report=repo / "parity.json",
        dvc_lock=repo / "dvc.lock",
        report_dir=repo / "reports",
    )


def patch_eval(
    candidate: dict | None = None,
    production: dict | None = None,
    cand_paths: tuple[str, ...] = (CAND_ONNX,),
) -> None:
    """Replace graph loading with fixture metrics, keyed by path.

    ``cand_paths`` exists because the CLI tests point ``--candidate-onnx`` at a
    tmp file rather than the module constant, and a fake that silently answered
    "production" for the candidate would have made a rejection test pass for the
    wrong reason.
    """
    cand, prod = candidate or metrics(), production or metrics()

    def fake_load(cfg: Config, onnx_path: str, frame) -> "pm.Graph":
        return pm.Graph(metrics=dict(cand if onnx_path in cand_paths else prod))

    # A Graph with no session is what the fake returns, and
    # measure_latency_pair skips graphs without one, so the fixture latency
    # values survive untouched -- which is what lets these tests drive the
    # latency and size gates at all.
    pm.load_graph = fake_load  # type: ignore[assignment]


@pytest.fixture(autouse=True)
def restore_eval():
    original = pm.load_graph
    yield
    pm.load_graph = original


class FakeSession:
    """Stands in for an ORT session, which starts with no ``.config``."""

    def __init__(self) -> None:
        self.config = None
        self.runs = 0

    def run(self, names, feeds):
        self.runs += 1
        return [np.zeros((1, 3), dtype=np.float32)]


class FakeConfig:
    def __init__(self) -> None:
        self.id2label = {0: "positive", 1: "negative", 2: "neutral"}


class FakeTokenizer:
    def __call__(self, texts, **kwargs):
        return {
            "input_ids": np.zeros((len(texts), 8), dtype=np.int64),
            "attention_mask": np.ones((len(texts), 8), dtype=np.int64),
        }


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


class TagTokenizer(FakeTokenizer):
    """Tags each batch so a session can record *which* graph just ran."""

    def __init__(self, tag: int) -> None:
        self.tag = tag

    def __call__(self, texts, **kwargs):
        batch = super().__call__(texts, **kwargs)
        batch["input_ids"][:, 0] = self.tag
        return batch


class TagSession(FakeSession):
    """Records every call into one shared log, so global order is observable.

    Per-session logs cannot show interleaving -- they would look alternating even
    if the code measured one graph entirely before the other.
    """

    def __init__(self, log: list[int]) -> None:
        super().__init__()
        self.log = log

    def run(self, names, feeds):
        self.log.append(int(feeds["input_ids"][0][0]))
        return super().run(names, feeds)


def _graph_pair(runs: int, warmup: int):
    log: list[int] = []
    cand = pm.Graph(metrics={}, session=TagSession(log), tokenizer=TagTokenizer(1))
    prod = pm.Graph(metrics={}, session=TagSession(log), tokenizer=TagTokenizer(2))
    cfg = dataclasses.replace(Config(), latency_runs=runs, latency_warmup=warmup)
    return cand, prod, cfg


def test_latency_is_measured_interleaved_with_the_order_swapped() -> None:
    """Sequential timing could not resolve a 10% band on a noisy 2-core box.

    Two runs of the same graph came out 21% apart, so the two graphs have to
    share the loop, and the order has to flip every round so neither one gets a
    systematically warmer slot.
    """
    cand, prod, cfg = _graph_pair(runs=4, warmup=0)
    pm.measure_latency_pair(cand, prod, ["نص", "آخر"], cfg)
    # Three slots (candidate, production, control) rotating, so each takes each
    # position equally often: 1,2,2 then 2,2,1 then 2,1,2 then 1,2,2.
    assert cand.session.log == [1, 2, 2, 2, 2, 1, 2, 1, 2, 1, 2, 2]
    assert cand.session.log.count(1) == 4, (
        "the candidate is not measured once per round"
    )


def _merged_order(cand, prod) -> list[int]:
    merged: list[int] = []
    for index in range(len(cand.session.seen)):
        merged.append(cand.session.seen[index])
        merged.append(prod.session.seen[index])
    return merged


def test_latency_discards_the_warmup_rounds() -> None:
    cand, prod, cfg = _graph_pair(runs=3, warmup=2)
    pm.measure_latency_pair(cand, prod, ["نص"], cfg)
    assert cand.metrics["latency"]["runs"] == 3
    assert prod.metrics["latency"]["runs"] == 3
    # 5 rounds x 3 series: 2 discarded, 3 kept.
    assert len(cand.session.log) == 15


def _script_the_clock(monkeypatch) -> None:
    """A monotonic clock whose per-call cost is a fixed, repeating pattern.

    The measurement itself is what is under test, not the scheduler: three
    series of the same graph must produce the same 10 ms every time, so the
    spread is exactly 0 and the assertions are arithmetic rather than luck.
    """
    ticks = iter(range(1_000_000))

    def fake_clock() -> int:
        return next(ticks)

    monkeypatch.setattr(pm.time, "perf_counter", fake_clock)


def test_latency_records_a_null_control_for_the_same_graph(monkeypatch) -> None:
    """A second series of the production graph measures the host, not the model.

    Without it, two runs of the identical graph differing by more than the
    allowed band are indistinguishable from a slow candidate -- which is exactly
    what the first full-split run did.
    """
    cand, prod, cfg = _graph_pair(runs=4, warmup=0)
    # FakeSession is a no-op, so without a scripted clock these "timings" are
    # wall-clock measurements of an empty function and the series differ by
    # whatever the scheduler did. That is not what this test is about, and it
    # made the assertion below depend on machine load.
    _script_the_clock(monkeypatch)
    pm.measure_latency_pair(cand, prod, ["نص"], cfg)
    assert "control_p95_ms" in prod.metrics["latency"]
    assert "control_p95_ms" not in cand.metrics["latency"]
    assert prod.metrics["latency"]["noise_pct"] is not None
    assert prod.metrics["latency"]["control_gap_pct"] is not None
    # Every call now costs exactly 1 ms, so all three series agree and the noise
    # figure is exactly zero -- asserted rather than merely bounded, because a
    # scripted clock makes it arithmetic.
    assert (
        prod.metrics["latency"]["p95_ms"] == prod.metrics["latency"]["control_p95_ms"]
    )
    assert prod.metrics["latency"]["noise_pct"] == 0.0
    assert prod.metrics["latency"]["control_gap_pct"] == 0.0
    assert (
        prod.metrics["latency"]["noise_pct"]
        >= prod.metrics["latency"]["control_gap_pct"] - 1e-9
    ), "the spread across all three series bounds the control gap"


def test_the_control_series_is_not_the_production_series() -> None:
    """A structural pin, because no behavioural test can exist for this.

    The control and the production series are the same graph in a symmetric
    loop, so they are statistically identical by construction -- swapping which
    list the control is summarised from would leave every assertion in this
    suite passing while making the noise figure permanently zero and the
    inconclusive diagnosis unreachable. The only way to see it is the code.
    """
    source = (
        Path(__file__).resolve().parents[1] / "scripts" / "promote_model.py"
    ).read_text()
    assert "np.percentile(control, 95)" in source
    assert (
        'control_p95_ms"] = _round(\n        np.percentile(right, 95), 3\n    )'
        not in source
    )
    # The spread is relative to the *fastest* of the three series, so one slow
    # outlier reads as the full spread instead of being divided away by itself.
    # Also unobservable through the fake loop, where the series are equal.
    assert "/ min(left_p95, right_p95, control_p95)" in source
    assert "/ max(left_p95, right_p95, control_p95)" not in source


def test_the_noise_figure_is_computed_not_assumed() -> None:
    """Pins the arithmetic behind the inconclusive diagnosis.

    Tested directly rather than through the timing loop because the two control
    series are the same fake session, so an end-to-end assertion would only ever
    see noise of ~0 and would pass just as happily against a hard-coded zero --
    which is exactly what mutation M20 does.
    """
    assert pm._noise_pct({"p95_ms": 40.0}, 50.0) == 25.0
    assert pm._noise_pct({"p95_ms": 40.0}, 30.0) == 25.0, "signed, not directional"
    assert pm._noise_pct({"p95_ms": 40.0}, 40.0) == 0.0
    assert pm._noise_pct({"p95_ms": 0}, 50.0) is None, "no baseline, no claim"


def test_a_slow_candidate_on_a_noisy_host_is_inconclusive_not_slow() -> None:
    """Both block the promotion, but only one of them is about the model."""
    production = metrics(
        latency={
            "p50_ms": 10.0,
            "p95_ms": 40.0,
            "runs": 50,
            "noise_pct": 25.0,
            "control_gap_pct": 4.0,
        }
    )
    candidate = metrics(latency={"p50_ms": 12.0, "p95_ms": 46.0, "runs": 50})
    result = gate_latency(candidate, production, Config())
    assert not result.passed
    assert "INCONCLUSIVE" in result.detail
    assert "same-graph noise=25.0%" in result.detail
    assert "control gap 4.0%" in result.detail
    assert "is NOT being cleared" in result.detail


def test_a_wide_spread_outranks_a_tight_control_gap() -> None:
    """The measured case that motivated the spread: a 4% control gap, a 14% spread.

    Comparing the int8 graph against itself, the two production series landed
    within 4% while a third series of the same graph landed 14% higher. Judging
    the noise by the control gap alone would have reported a confident "14%
    regression" about a graph that did not change at all.
    """
    production = metrics(
        latency={
            "p50_ms": 10.0,
            "p95_ms": 56.025,
            "runs": 50,
            "noise_pct": 14.36,
            "control_gap_pct": 4.08,
        }
    )
    candidate = metrics(latency={"p50_ms": 12.0, "p95_ms": 64.07, "runs": 50})
    result = gate_latency(candidate, production, Config())
    assert not result.passed
    assert "INCONCLUSIVE" in result.detail
    # 64.07 against a budget of 56.025 * 1.1 = 61.63 is 4% over budget, while
    # the spread that makes it unresolvable is 14.4%.
    assert "4.0% over budget" in result.detail
    assert "same-graph noise=14.36%" in result.detail


def test_a_slow_candidate_on_a_quiet_host_is_plainly_slow() -> None:
    production = metrics(
        latency={
            "p50_ms": 10.0,
            "p95_ms": 40.0,
            "runs": 50,
            "noise_pct": 2.0,
            "control_gap_pct": 1.0,
        }
    )
    candidate = metrics(latency={"p50_ms": 12.0, "p95_ms": 60.0, "runs": 50})
    result = gate_latency(candidate, production, Config())
    assert not result.passed
    assert "INCONCLUSIVE" not in result.detail
    assert "same-graph noise=2.0%" in result.detail


def test_a_fast_candidate_passes_even_on_a_noisy_host() -> None:
    """Noise must not turn a genuinely faster candidate into a failure."""
    production = metrics(
        latency={
            "p50_ms": 10.0,
            "p95_ms": 40.0,
            "runs": 50,
            "noise_pct": 30.0,
            "control_gap_pct": 12.0,
        }
    )
    candidate = metrics(latency={"p50_ms": 9.0, "p95_ms": 38.0, "runs": 50})
    assert gate_latency(candidate, production, Config()).passed


def test_latency_records_that_it_was_paired() -> None:
    cand, prod, cfg = _graph_pair(runs=2, warmup=0)
    pm.measure_latency_pair(cand, prod, ["نص"], cfg)
    assert "interleaved" in cand.metrics["latency"]["method"]
    assert "control" in cand.metrics["latency"]["method"]


def test_latency_skips_both_graphs_when_either_is_unavailable() -> None:
    """All-or-nothing on purpose.

    Timing production alone would produce a number that is not comparable with
    the report's candidate figure, which is worse than having no figure at all:
    the gate should say it could not measure rather than half-measure.
    """
    empty = pm.Graph(metrics={})
    _cand, prod, cfg = _graph_pair(runs=2, warmup=0)
    pm.measure_latency_pair(empty, prod, ["نص"], cfg)
    assert "latency" not in empty.metrics
    assert "latency" not in prod.metrics
    assert prod.session.log == []


def test_single_graph_latency_works() -> None:
    graph = pm.Graph(metrics={}, session=TagSession([]), tokenizer=TagTokenizer(7))
    cfg = dataclasses.replace(Config(), latency_runs=2, latency_warmup=1)
    pm.measure_latency(graph, ["نص"], cfg)
    assert graph.metrics["latency"]["runs"] == 2
    assert graph.metrics["latency"]["p95_ms"] > 0


# --- the label-order pin inside the evaluator --------------------------------


def test_evaluate_graph_attaches_the_config_that_keeps_label_order(
    repo: Path, monkeypatch
) -> None:
    """The gate must read id2label, not infer it.

    An ORT session has no ``.config``, so without this line ``evaluate_on_split``
    falls back to alphabetical label order -- negative, neutral, positive --
    against a model whose ids are positive=0, negative=1, neutral=2. Every
    metric would still be computed and would still look plausible, with Neutral
    and Negative silently swapped: the worst possible failure for a gate, since
    it produces a confident wrong answer rather than an error.
    """
    import transformers

    from raay.training import evaluate as ev

    session = FakeSession()
    seen: dict = {}

    def fake_eval(frame, sess, tokenizer, model_name, max_length):
        seen["config"] = sess.config
        return metrics()

    seen_options: dict = {}

    def fake_session_load(path, *, session_options=None):
        seen_options["options"] = session_options
        return session

    monkeypatch.setattr(ev, "load_onnx_session", fake_session_load)
    monkeypatch.setattr(ev, "evaluate_on_split", fake_eval)
    monkeypatch.setattr(ev, "dialect_breakdown", lambda *a, **k: {})
    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained", lambda *a, **k: FakeConfig()
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: FakeTokenizer()
    )

    cfg = dataclasses.replace(
        Config(
            test_split=repo / "test.csv",
            tokenizer_dir=repo / "tok",
            dvc_lock=repo / "dvc.lock",
        ),
        latency_runs=2,
        latency_warmup=1,
    )
    frame = pd.DataFrame({"text": ["مراجعة جيدة", "خدمة سيئة", "عادي"]})

    out = pm.evaluate_graph(cfg, str(repo / "prod.onnx"), frame)

    assert isinstance(seen["config"], FakeConfig), "id2label was never attached"
    # The session being timed is also the session that was scored, so this is
    # the only place the gate's session configuration can be pinned. Spinning
    # workers are what made two sessions in one process look 25% apart.
    options = seen_options["options"]
    assert options is not None, "the gate scored the graph on default options"
    assert options.get_session_config_entry("session.intra_op.allow_spinning") == "0"
    # Spinning off, pool intact. Pinning intra_op_num_threads=1 was the obvious
    # "reduce the noise" move and it makes this graph ~3x slower (measured here:
    # 242 ms against 86 ms per call), which would change what the gate is
    # measuring rather than stabilise it.
    assert options.intra_op_num_threads == 0, "the thread pool must stay at its default"
    assert [seen["config"].id2label[i] for i in range(3)] == list(LABELS)
    assert out["label_names"] == list(LABELS)
    assert out["id2label"] == {"0": "positive", "1": "negative", "2": "neutral"}
    # 1 warmup call is made and then discarded, so 3 calls yield 2 samples --
    # keeping the warmup in the percentile would understate p95.
    assert session.runs == 3, "the latency measurement did not actually run"
    assert out["latency"]["runs"] == 2, "the warmup call leaked into the samples"


def test_the_warmup_is_long_enough_to_reach_steady_state() -> None:
    """Pinned because the evidence for it is a one-off measurement, not a law.

    Three warm-up calls were measured leaving the first of the three series
    still paying for the other two graphs' pools coming up, which is a
    systematic bias rather than noise -- and systematic bias is exactly what a
    rotating order does not cancel. Ten costs about a second.
    """
    assert Config().latency_warmup == 10
    assert Config().latency_runs == 50


def test_the_shared_session_loader_keeps_its_default_behaviour(monkeypatch) -> None:
    """Passing options must be opt-in, or this changes the parity checker too.

    ``load_onnx_session`` is also used by ``quantize_onnx`` and ``evaluate``.
    The gate needs tuned options; those callers do not, and silently giving them
    no-spinning sessions would change what every other latency number in the
    repo means.
    """
    import onnxruntime as ort

    from raay.training import evaluate as ev

    calls: list[dict] = []

    def fake_session(path, **kwargs):
        calls.append(kwargs)
        return "session"

    monkeypatch.setattr(ort, "InferenceSession", fake_session)

    ev.load_onnx_session("model.onnx")
    assert "sess_options" not in calls[0], "the default path started passing options"

    options = ort.SessionOptions()
    ev.load_onnx_session("model.onnx", session_options=options)
    assert calls[1]["sess_options"] is options
    assert calls[1]["providers"] == ["CPUExecutionProvider"]


def test_evaluate_graph_refuses_a_missing_graph(repo: Path) -> None:
    cfg = Config(test_split=repo / "test.csv")
    with pytest.raises(PromotionError, match="graph not found"):
        pm.evaluate_graph(cfg, str(repo / "absent.onnx"), pd.DataFrame({"text": ["x"]}))


# --- the gate as a whole -----------------------------------------------------


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


# --- the registry flow -------------------------------------------------------


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

    monkeypatch.setattr(pm, "load_graph", spy)
    monkeypatch.setattr(pm, "load_split", lambda cfg: pd.DataFrame({"text": ["x"]}))
    repo.joinpath("test.csv").write_text("text\nمراجعة\n")

    with pytest.raises(pm.PromotionError) as excinfo:
        promote(cfg, "7", CAND_ONNX, PROD_ONNX, client=None)

    message = str(excinfo.value)
    assert "does not match" in message
    # Both hashes, or the operator debugging a drifted split has to go compute
    # one of them by hand.
    assert "observed md5" in message and "locked" in message
    assert called == [], "the graphs were scored against an unverified split"


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


# --- the report --------------------------------------------------------------


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


# --- the CLI -----------------------------------------------------------------


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
