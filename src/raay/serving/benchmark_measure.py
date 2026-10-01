"""Measure the serving path: one tokenize+ORT call per chunk, timed.

Split out of ``benchmark`` so the timing loop is not tangled with the MLflow
logging. The loop is deliberately the *real* online shape -- each timed chunk is
exactly one ``predict_probs`` call, i.e. one request's worth of work -- so
``chunk_latency_ms`` is directly comparable to a serving p50/p95.

Accuracy is computed from the **first pass only** (``pass_idx == 1``): later
passes re-score the same text, so including them would only add samples that
cannot change the answer.

``latency_ms_per_req_equivalent`` divides chunk latency by the batch size. On a
latency-bound backend that correctly says a batch-32 call is *worse* per request
than batch 1 -- which is what this repo measured on its 2-core box (batch-32
per-item ≈57-80 ms vs ~12 ms single), and why ``pure_cpu`` speedup <1 is an
expected result there rather than a bug.
"""

from __future__ import annotations

import statistics
import time
from typing import Any

import numpy as np

from raay.serving.serve import predict_probs


def benchmark_batch_size(
    session: Any,
    tokenizer: Any,
    texts: list[str],
    y_true: np.ndarray,
    label_to_id: dict[str, int],
    model_name: str,
    max_length: int,
    batch_size: int,
    runs: int,
) -> dict[str, Any]:
    """Time full passes over ``texts`` at one batch size.

    Each chunk is one tokenize+ORT call (one online call at that size). Warm
    up one chunk first, then record every chunk across ``runs`` passes.
    """
    n = len(texts)
    first = texts[:batch_size]
    predict_probs(session, tokenizer, first, model_name, max_length)

    chunk_ms: list[float] = []
    argmax_seen: list[int] = []
    for pass_idx in range(1, runs + 1):
        for i in range(0, n, batch_size):
            chunk = texts[i : i + batch_size]
            t0 = time.perf_counter()
            probs = predict_probs(session, tokenizer, chunk, model_name, max_length)
            chunk_ms.append((time.perf_counter() - t0) * 1000.0)
            if pass_idx == 1:
                argmax_seen.extend(np.argmax(probs, axis=-1).tolist())

    chunk_ms = sorted(chunk_ms)
    total_sec = sum(chunk_ms) / 1000.0
    total_items = n * runs
    per_item = [ms / batch_size for ms in chunk_ms]

    y_pred = np.array(argmax_seen[:n])
    accuracy = float(np.mean(y_pred == y_true))

    def pct(samples: list[float], p: float) -> float:
        return float(np.percentile(samples, p))

    return {
        "batch_size": batch_size,
        "n_samples": n,
        "n_calls": len(chunk_ms),
        "n_runs": runs,
        "chunk_latency_ms": {
            "mean": float(statistics.mean(chunk_ms)),
            "p50": pct(chunk_ms, 50),
            "p95": pct(chunk_ms, 95),
            "p99": pct(chunk_ms, 99),
        },
        "latency_ms_per_req_equivalent": {
            "mean": float(statistics.mean(per_item)),
            "p50": pct(per_item, 50),
            "p95": pct(per_item, 95),
            "p99": pct(per_item, 99),
        },
        "throughput_req_per_s": float(total_items / total_sec),
        "accuracy": accuracy,
    }
