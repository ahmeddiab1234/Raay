"""Benchmark helpers: the serial reference, the report shape, and logging."""

from __future__ import annotations

import argparse
import time
from typing import Any

from loguru import logger

from raay.enums.constants import DefaultPaths
from raay.inference.consumer_loop import BatchConsumer
from raay.inference.consumer_messages import ReviewMessage
from raay.inference.consumer_queues import MemoryQueue, QueueBackend, RedisQueue
from raay.inference.consumer_scorer import InferenceScorer


def _serial_reference(
    scorer: InferenceScorer, texts: list[str], overhead_ms: float
) -> dict[str, float]:
    """Time scoring every review one-at-a-time: the batch-win baseline.

    ``overhead_ms`` models a fixed per-call cost (e.g. an HTTP /predict or JSON
    round-trip); each review pays it once in the serial path.
    """
    if not texts:
        return {
            "pull_n_calls": 0,
            "elapsed_sec": 0.0,
            "overhead_sec": 0.0,
            "total_sec": 0.0,
            "reviews_per_sec": 0.0,
            "avg_ms_per_review": 0.0,
        }
    scorer.score(texts[:1])  # warm the ORT session
    t0 = time.perf_counter()
    for text in texts:
        scorer.score([text])
    elapsed = time.perf_counter() - t0
    n_calls = len(texts)
    overhead = n_calls * overhead_ms / 1000.0
    total = elapsed + overhead
    return {
        "n_calls": n_calls,
        "elapsed_sec": elapsed,
        "overhead_sec": overhead,
        "total_sec": total,
        "reviews_per_sec": len(texts) / total if total else 0.0,
        "avg_ms_per_review": total * 1000.0 / len(texts) if texts else 0.0,
    }


def _build_queue_from_args(args: argparse.Namespace) -> QueueBackend:
    if args.mode == "benchmark" and not args.redis:
        return MemoryQueue()
    return RedisQueue(
        url=args.redis_url,
        input_key=args.input_queue,
        result_key=args.result_queue,
        poll_ms=args.poll_ms,
    )


def _load_texts(csv_path: str, samples: int) -> list[str]:
    import pandas as pd

    df = pd.read_csv(csv_path).head(samples)
    texts = df["text"].astype(str).tolist()
    logger.info(f"Loaded {len(texts)} reviews from {csv_path}")
    return texts


def _produce(
    queue_backend: QueueBackend, texts: list[str], rate: float
) -> tuple[int, float]:
    t0 = time.perf_counter()
    total = 0
    interval = 1.0 / rate if rate > 0 else 0.0
    for i, text in enumerate(texts):
        queue_backend.push([ReviewMessage(text=text, id=str(i))])
        total += 1
        if i < len(texts) - 1 and interval > 0:
            time.sleep(interval)
    elapsed = time.perf_counter() - t0
    logger.info(
        f"Produced {total} reviews ({elapsed:.1f}s)"
        + (f", {rate:.0f}/s rate" if rate > 0 else ", burst")
    )
    return total, elapsed


def _run_report(
    consumer: BatchConsumer,
    texts_done: list[str],
    serial: dict[str, float],
    args: argparse.Namespace,
    queue_factory: str,
    leftover: int = 0,
) -> dict[str, Any]:
    stats = consumer.stats
    call_amortization = stats.items_processed / stats.batches if stats.batches else 0.0
    batch_overhead_sec = stats.batches * args.overhead_ms / 1000.0
    speedup_cpu = (
        serial["elapsed_sec"] / stats.inference_work_sec
        if stats.inference_work_sec > 0
        else 0.0
    )
    speedup_total = (
        serial["total_sec"] / (stats.inference_work_sec + batch_overhead_sec)
        if stats.inference_work_sec > 0
        else 0.0
    )
    return {
        "metadata": {
            "backend": queue_factory,
            "mode": args.mode,
            "max_batch_size": args.max_batch,
            "max_wait_ms": args.max_wait_ms,
            "overhead_ms_per_call": args.overhead_ms,
            "onnx_path": args.onnx_path or DefaultPaths.ONNX_INT8_MODEL.value,
            "tokenizer_dir": args.tokenizer_dir or DefaultPaths.BASELINE_MODEL.value,
            "n_requested": len(texts_done),
            "n_processed": stats.items_processed,
            "n_leftover": leftover,
        },
        "micro_batch": {
            "batches": stats.batches,
            "avg_batch_size": stats.avg_batch_size,
            "fill_ratio_avg": stats.fill_ratio_avg,
            "fill_p50": stats.fill_p50,
            "drain_time_sec": stats.drain_time_sec,
            "score_time_sec": stats.score_time_sec,
            "inference_work_sec": stats.inference_work_sec,
            "elapsed_sec": stats.total_time_sec,
            "reviews_per_sec": stats.reviews_per_sec,
            "work_reviews_per_sec": stats.work_reviews_per_sec,
            "call_amortization": call_amortization,
        },
        "serial_reference": serial,
        "speedup_vs_serial": {
            "pure_cpu": speedup_cpu,
            "with_overhead": speedup_total,
        },
    }


def _log_summary(consumer: BatchConsumer) -> None:
    s = consumer.stats
    logger.info(
        f"consumer summary: {s.batches} batches, {s.items_processed} reviews, "
        f"avg batch={s.avg_batch_size:.1f}, fill p50={s.fill_p50 * 100:.0f}%, "
        f"{s.reviews_per_sec:.0f} reviews/sec over {s.total_time_sec:.1f}s"
    )
