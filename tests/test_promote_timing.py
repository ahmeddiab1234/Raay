"""Interleaved promotion latency gate tests."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import promote_model as pm
import pytest
from promote_model import Config, gate_latency

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

from promotion_helpers import (
    FakeSession,
    FakeTokenizer,
    metrics,
    promotion_sources,
    restore_eval,
)


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


@pytest.fixture(autouse=True)
def _restore_eval():
    with restore_eval():
        yield


def _graph_pair(runs: int, warmup: int):
    log: list[int] = []
    cand = pm.Graph(metrics={}, session=TagSession(log), tokenizer=TagTokenizer(1))
    prod = pm.Graph(metrics={}, session=TagSession(log), tokenizer=TagTokenizer(2))
    cfg = dataclasses.replace(Config(), latency_runs=runs, latency_warmup=warmup)
    return cand, prod, cfg


def _merged_order(cand, prod) -> list[int]:
    merged: list[int] = []
    for index in range(len(cand.session.seen)):
        merged.append(cand.session.seen[index])
        merged.append(prod.session.seen[index])
    return merged


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


def test_latency_discards_the_warmup_rounds() -> None:
    cand, prod, cfg = _graph_pair(runs=3, warmup=2)
    pm.measure_latency_pair(cand, prod, ["نص"], cfg)
    assert cand.metrics["latency"]["runs"] == 3
    assert prod.metrics["latency"]["runs"] == 3
    # 5 rounds x 3 series: 2 discarded, 3 kept.
    assert len(cand.session.log) == 15


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


def test_the_flow_calls_through_module_attributes_not_re_exports() -> None:
    """Structural, because the failure it prevents is silent.

    ``promote_model.py`` re-exports the siblings' names. ``promote()`` must
    therefore reach them as ``promotion_graph.load_graph(...)`` rather than
    binding them with ``from promotion_graph import load_graph``: a
    from-import copies the reference at import time, so every fixture that
    replaces ``promotion_graph.load_graph`` -- which is how *all* nineteen of the
    graph-loading tests inject their metrics -- would patch a name the flow never
    reads. Nothing fails at import and nothing fails loudly; the flow just keeps
    loading the real 136 MB graph. This exact regression happened once already.

    ``pm.time`` is the counter-example that shows why the rule is not "never
    touch the facade": patching that *does* work, because ``time`` is a shared
    module object rather than a rebound function name.
    """
    source = (SCRIPTS / "promotion_flow.py").read_text()
    assert "import promotion_graph" in source
    assert "promotion_graph.load_graph(" in source
    assert "promotion_split.load_split(" in source
    for name in ("load_graph", "load_split", "measure_latency_pair"):
        assert f"from promotion_graph import {name}" not in source
        assert f"from promotion_split import {name}" not in source
        assert f"from promotion_timing import {name}" not in source
        # A bare call means the name resolved from a from-import or the module
        # globals, not through the defining module.
        assert f"\n    {name}(" not in source, f"{name}() is called unqualified"


def test_the_control_series_is_not_the_production_series() -> None:
    """A structural pin, because no behavioural test can exist for this.

    The control and the production series are the same graph in a symmetric
    loop, so they are statistically identical by construction -- swapping which
    list the control is summarised from would leave every assertion in this
    suite passing while making the noise figure permanently zero and the
    inconclusive diagnosis unreachable. The only way to see it is the code.
    """
    source = promotion_sources()
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
