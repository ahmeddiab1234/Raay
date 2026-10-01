"""CLI for the micro-batch consumer: consumer / producer / benchmark modes."""

from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path

from loguru import logger

from raay.config.env import load_environment
from raay.enums.constants import DefaultPaths
from raay.inference.consumer_benchmark import (
    _build_queue_from_args,
    _load_texts,
    _log_summary,
    _produce,
    _run_report,
    _serial_reference,
)
from raay.inference.consumer_loop import BatchConsumer
from raay.inference.consumer_queues import QueueBackend
from raay.inference.consumer_scorer import InferenceScorer


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["consumer", "producer", "benchmark"], default="consumer"
    )
    parser.add_argument(
        "--redis-url", default="redis://localhost:6379", help="Redis transport URL."
    )
    parser.add_argument(
        "--redis", action="store_true", help="Force Redis for benchmark."
    )
    parser.add_argument("--input-queue", default="reviews")
    parser.add_argument("--result-queue", default="reviews-results")
    parser.add_argument("--poll-ms", type=int, default=100)
    parser.add_argument("--max-batch", type=int, default=32)
    parser.add_argument("--max-wait-ms", type=int, default=100)
    parser.add_argument("--onnx-path", default=None)
    parser.add_argument("--tokenizer-dir", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--csv", default=DefaultPaths.TEST_SPLIT.value)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument(
        "--rate", type=float, default=0.0, help="Producer arrivals/sec."
    )
    parser.add_argument(
        "--duration", type=float, default=15.0, help="Consumer wall clock."
    )
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument(
        "--max-idle", type=float, default=3.0, help="Stop after idle secs."
    )
    parser.add_argument(
        "--overhead-ms",
        type=float,
        default=0.0,
        help="Simulated fixed cost per 'call' (e.g. HTTP /predict round-trip). "
        "Serial pays it once per review, the consumer once per batch.",
    )
    parser.add_argument("--output", default=DefaultPaths.QUEUE_BENCHMARK.value)
    return parser


def _run_benchmark(
    args: argparse.Namespace, queue_backend: QueueBackend, scorer: InferenceScorer
) -> None:
    """Feed a queue live, then compare the consumer against one-at-a-time scoring."""
    consumer = BatchConsumer(
        queue_backend,
        scorer,
        max_batch_size=args.max_batch,
        max_wait_ms=args.max_wait_ms,
    )
    texts = _load_texts(args.csv, args.samples)
    scorer.score(texts[:1])  # warm the ORT session before timing

    def _feed() -> None:
        _produce(queue_backend, texts, args.rate)

    thread = threading.Thread(target=_feed, daemon=True)
    thread.start()
    consumer.run(
        duration_sec=args.duration,
        max_batches=args.max_batches,
        stop_on_idle_sec=args.max_idle,
    )
    thread.join(timeout=max(args.duration, 30))
    leftover = len(texts) - consumer.stats.items_processed
    serial = _serial_reference(
        scorer, texts[: consumer.stats.items_processed], args.overhead_ms
    )

    report = _run_report(
        consumer,
        texts,
        serial,
        args,
        queue_factory=type(queue_backend).__name__,
        leftover=max(0, leftover),
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Wrote micro-batch benchmark: {args.output}")
    _log_summary(consumer)
    mb = report["micro_batch"]
    spd = report["speedup_vs_serial"]
    logger.info(
        f"serial one-at-a-time: {serial['reviews_per_sec']:.0f} rev/s "
        f"({serial['avg_ms_per_review'] * 1000:.0f}us/rev incl {args.overhead_ms}ms/call); "
        f"micro-batch: {mb['work_reviews_per_sec']:.0f} rev/s (work-time); "
        f"amortization={mb['call_amortization']:.1f}x "
        f"({mb['batches']} calls vs {serial['n_calls']}); "
        f"speedup pure-CPU={spd['pure_cpu']:.2f}x, "
        f"with-overhead={spd['with_overhead']:.2f}x"
    )


def main() -> None:
    args = _build_parser().parse_args()

    load_environment()
    queue_backend = _build_queue_from_args(args)
    scorer = InferenceScorer(
        onnx_path=args.onnx_path,
        tokenizer_dir=args.tokenizer_dir,
        model_name=args.model_name,
        max_length=args.max_length,
    )

    if args.mode == "producer":
        texts = _load_texts(args.csv, args.samples)
        _produce(queue_backend, texts, args.rate)
        return

    if args.mode == "consumer":
        consumer = BatchConsumer(
            queue_backend,
            scorer,
            max_batch_size=args.max_batch,
            max_wait_ms=args.max_wait_ms,
        )
        try:
            consumer.run(duration_sec=args.duration, max_batches=args.max_batches)
        except KeyboardInterrupt:
            consumer.stop()
        _log_summary(consumer)
        return

    _run_benchmark(args, queue_backend, scorer)
