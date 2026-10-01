"""Latency measurement for the promotion gate.

The measurement is deliberately awkward: interleaved, rotating, and carrying a
null control. Two runs of the *same* int8 graph on an idle 2-core box came out
21% apart, which is larger than the 10% band the gate allows, so a sequential
A-then-B measurement would be deciding on CPU contention rather than on model
quality.

Nothing here is compared against ``reports/benchmark_table.csv``: a GitHub runner
is several times slower than the box that produced that table, and an absolute
budget copied from it would fail for reasons that have nothing to do with the
candidate.
"""

from __future__ import annotations

import time

import numpy as np
from promotion_types import Config, Graph, _round


def _one_call(graph: Graph, texts: list[str], index: int, cfg: Config) -> float:
    from raay.serving.serve import _preprocess

    text = _preprocess(texts[index % len(texts)], cfg.model_name)
    enc = graph.tokenizer(
        [text],
        truncation=True,
        padding=True,
        max_length=cfg.max_length,
        return_tensors="np",
    )
    start = time.perf_counter()
    graph.session.run(
        ["logits"], {key: enc[key] for key in ("input_ids", "attention_mask")}
    )
    return (time.perf_counter() - start) * 1000.0


def _summarize(samples: list[float], method: str) -> dict[str, object]:
    return {
        "p50_ms": _round(np.percentile(samples, 50), 3),
        "p95_ms": _round(np.percentile(samples, 95), 3),
        "runs": len(samples),
        "method": method,
    }


def measure_latency(graph: Graph, texts: list[str], cfg: Config) -> None:
    """Time one graph in place, mirroring the batch-1 method in ``benchmark.py``."""
    if graph.session is None:
        return
    samples: list[float] = []
    for run in range(cfg.latency_warmup + cfg.latency_runs):
        if run == cfg.latency_warmup:
            samples.clear()
        samples.append(_one_call(graph, texts, run, cfg))
    graph.metrics["latency"] = _summarize(samples, "single graph, same process")


def _noise_pct(reference: dict[str, object], control_p95: float) -> float | None:
    """How far apart two series of the *same* graph landed, as a percentage."""
    base = reference.get("p95_ms")
    if not base:
        return None
    return _round(abs(control_p95 - base) / base * 100.0, 2)


def measure_latency_pair(
    candidate: Graph, production: Graph, texts: list[str], cfg: Config
) -> None:
    """Time both graphs in one interleaved loop, in place.

    Interleaving puts both graphs through the same load conditions, and swapping
    the order every round cancels any first-slot advantage. What is left is the
    difference, which is the only part that is about the model.

    The third series is a **control**: the production graph, timed again through
    the same loop. Two series of the *same* graph differing by N% is a direct
    measurement of how much of any observed gap is the host rather than the
    model. Without the control the gate would report "your model is slower"
    about a machine that cannot tell the difference.
    """
    if candidate.session is None or production.session is None:
        return
    left: list[float] = []
    right: list[float] = []
    control: list[float] = []
    for run in range(cfg.latency_warmup + cfg.latency_runs):
        if run == cfg.latency_warmup:
            left.clear()
            right.clear()
            control.clear()
        series = [(candidate, left), (production, right), (production, control)]
        # Rotating rather than reversing: three slots, so every graph takes every
        # position the same number of times.
        offset = run % 3
        series = series[offset:] + series[:offset]
        for graph, sink in series:
            sink.append(_one_call(graph, texts, run, cfg))
    candidate.metrics["latency"] = _summarize(left, "interleaved A/B + control")
    production.metrics["latency"] = _summarize(right, "interleaved A/B + control")
    left_p95 = float(np.percentile(left, 95))
    right_p95 = float(np.percentile(right, 95))
    control_p95 = float(np.percentile(control, 95))
    production.metrics["latency"]["control_p95_ms"] = _round(control_p95, 3)
    production.metrics["latency"]["control_gap_pct"] = _noise_pct(
        production.metrics["latency"], production.metrics["latency"]["control_p95_ms"]
    )
    # The control gap is one sample of the host's noise; the spread across all
    # three series is the honest bound. Two identical graphs measured at 56.0
    # and 64.1 ms can produce a control gap of only 4% while the true spread is
    # 14% -- and a 4% "noise" would then read as a real 14% regression.
    production.metrics["latency"]["noise_pct"] = _round(
        (max(left_p95, right_p95, control_p95) - min(left_p95, right_p95, control_p95))
        / min(left_p95, right_p95, control_p95)
        * 100.0,
        2,
    )


def latency_session_options():
    """Session options for the graph being timed, not for the graph being scored.

    The default ORT configuration sizes a thread pool to the machine and leaves
    its workers spinning between runs. One such pool is fine; the gate needs
    *two* in one process, and they fight: measured here, the spread across the
    three series of one identical graph was 1.2% with a single session and 6.3%
    with two, purely from that interference. Turning spinning off leaves the
    pool in place -- the graph is ~3x slower on one thread, so the pool is not
    optional -- and brings the two-session figure back to 2.8%.

    Only the gate does this. ``serve.py`` builds its own session and is
    deliberately left alone: this is a measurement fix, not a serving change.
    """
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return options
