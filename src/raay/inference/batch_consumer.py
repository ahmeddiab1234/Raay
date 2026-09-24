"""Micro-batch queue consumer for the INT8 ONNX sentiment scorer.

Phase 5 step 2 (near-real-time pattern): reviews arrive continuously on a
queue and are scored in small batches instead of one at a time so the shared
``serve.predict_probs`` path does one tokenize + ONNX ``session.run`` per
micro-batch (this is the client-side equivalent of BentoML's ``batchable=True``
Runner aggregation: N items in, one ORT call out).

Pipeline: input queue --becomes--> [micro-batch] -> INT8 ONNX predictor -->
Redis result queue, with throughput (reviews/sec) and batch-fill efficiency
logged per batch.

Transports
----------
- :class:`RedisQueue` (default): a Redis **list**; ``BLPOP``'s block timeout
  gives the "wait up to T ms" side of the micro-batch trigger natively. Run
  with ``docker compose up -d queue-redis``.
- :class:`MemoryQueue`: pure-Python fallback used for unit tests and the
  ``benchmark`` self-contained mode (no Redis server required).

Payloads
--------
Queue elements are either a plain review string or a JSON object
``{"id": ..., "text": ...}``. Results are always JSON
``{"id"?, "text", "label", "score"}``.

Run
---
    # consumer (default): drains reviews -> scores in micro-batches -> publishes
    uv run python -m raay.inference.batch_consumer --duration 30

    # producer: push reviews from a CSV (optionally at a fixed arrival rate)
    uv run python -m raay.inference.batch_consumer --mode producer \
        --csv data/processed/test.csv --samples 200 --rate 100

    # benchmark: feed a queue live, measure fill efficiency + reviews/sec and
    # compare against one-at-a-time scoring; writes reports/queue_benchmark.json
    uv run python -m raay.inference.batch_consumer --mode benchmark \
        --csv data/processed/test.csv --samples 200 --duration 15

Env-tunable like the serving endpoint: ``RAAY_ONNX_PATH`` (default
``models/onnx/model_int8.onnx``), ``RAAY_TOKENIZER_DIR``, ``RAAY_MODEL_NAME``,
``RAAY_MAX_LENGTH``.
"""

from __future__ import annotations

import argparse
import json
import queue
import statistics
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import onnxruntime as ort
import redis
from loguru import logger
from redis import exceptions as redis_exceptions
from transformers import AutoConfig, AutoTokenizer

from raay.config.env import load_environment
from raay.enums.constants import DefaultPaths, Models
from raay.serving.serve import predict_probs, to_predictions


@dataclass(frozen=True)
class ReviewMessage:
    """One queue element: the review text plus an optional correlation id."""

    text: str
    id: str | None = None


def encode_message(message: ReviewMessage) -> str:
    """Encode a message for the wire: JSON when it carries an id, else raw text."""
    if message.id is None:
        return message.text
    return json.dumps({"id": message.id, "text": message.text}, ensure_ascii=False)


def decode_message(payload: str) -> ReviewMessage | None:
    """Decode a queue payload; ``None`` for elements we must skip (bad JSON)."""
    payload = payload if isinstance(payload, str) else str(payload)
    try:
        data = json.loads(payload)
    except ValueError:
        return ReviewMessage(text=payload)
    if isinstance(data, dict) and isinstance(data.get("text"), str):
        msg_id = data.get("id")
        return ReviewMessage(text=data["text"], id=str(msg_id) if msg_id else None)
    logger.warning(f"Dropping undecodable queue element: {payload[:120]!r}")
    return None


def encode_result(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False)


class QueueBackend(Protocol):
    """A pull queue for inbound reviews plus a result sink."""

    def drain(self, max_items: int, max_wait_ms: int) -> list[ReviewMessage]: ...
    def push(self, messages: list[ReviewMessage]) -> int: ...
    def push_results(self, results: list[dict[str, Any]]) -> int: ...


class MemoryQueue:
    """Thread-safe queue over a Python ``queue.Queue`` (tests + benchmark mode).

    ``drain`` honours the micro-batch trigger exactly: return up to
    ``max_items``, or block for at most ``max_wait_ms`` total, whichever comes
    first.
    """

    def __init__(self) -> None:
        self._input: queue.Queue[str] = queue.Queue()
        self._results: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def drain(self, max_items: int, max_wait_ms: int) -> list[ReviewMessage]:
        deadline = time.monotonic() + max_wait_ms / 1000.0
        out: list[ReviewMessage] = []
        while len(out) < max_items:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                payload = self._input.get(timeout=min(remaining, 0.05))
            except queue.Empty:
                break
            message = decode_message(payload)
            if message is not None:
                out.append(message)
        return out

    def push(self, messages: list[ReviewMessage]) -> int:
        for message in messages:
            self._input.put(encode_message(message))
        return len(messages)

    def push_results(self, results: list[dict[str, Any]]) -> int:
        with self._lock:
            self._results.extend(results)
        return len(results)

    def results(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._results)


class RedisQueue:
    """Redis list transport: ``BLPOP`` input key, ``RPUSH`` output key.

    Elements are strings (see :func:`encode_message`). ``drain`` slices the
    block timeout into ``poll_ms`` windows so a slow trickle of items still
    yields early once ``max_items`` fill the batch without waiting the whole
    window after the deadline.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379",
        input_key: str = "reviews",
        result_key: str = "reviews-results",
        poll_ms: int = 250,
        client: Any = None,
    ) -> None:
        self._client = (
            client
            if client is not None
            else redis.from_url(url, decode_responses=True, socket_timeout=3.0)
        )
        self._input_key = input_key
        self._result_key = result_key
        self._poll_ms = poll_ms
        self._transient_logged = False

    def drain(self, max_items: int, max_wait_ms: int) -> list[ReviewMessage]:
        deadline = time.monotonic() + max_wait_ms / 1000.0
        out: list[ReviewMessage] = []
        while len(out) < max_items:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            block_ms = max(1, int(min(remaining, self._poll_ms / 1000.0) * 1000))
            try:
                pair = self._client.blpop(self._input_key, timeout=block_ms)
            except (redis_exceptions.TimeoutError, redis_exceptions.ConnectionError):
                # Transient client-side socket timeout over NAT/port-forwarding;
                # retry the same poll window instead of failing the whole drain.
                if not self._transient_logged:
                    logger.warning(
                        "Redis blpop timed out; treating as an empty poll window."
                    )
                    self._transient_logged = True
                time.sleep(0.01)
                continue
            if pair is None:
                continue
            message = decode_message(str(pair[1]))
            if message is not None:
                out.append(message)
        return out

    def push(self, messages: list[ReviewMessage]) -> int:
        if not messages:
            return 0
        return int(
            self._client.rpush(self._input_key, *[encode_message(m) for m in messages])
        )

    def push_results(self, results: list[dict[str, Any]]) -> int:
        if not results:
            return 0
        return int(
            self._client.rpush(self._result_key, *[encode_result(r) for r in results])
        )

    def count_results(self) -> int:
        return int(self._client.llen(self._result_key))


class InferenceScorer:
    """Lazy INT8 ONNX scorer sharing ``serve.predict_probs``.

    ``score`` returns ``{"label", "score"}`` per input in order, exactly like
    a single ``/predict`` call would — but a micro-batch of ≤32 texts is one
    tokenize + one ``session.run`` instead of 32 of each.
    """

    def __init__(
        self,
        onnx_path: str | None = None,
        tokenizer_dir: str | None = None,
        model_name: str | None = None,
        max_length: int = 128,
    ) -> None:
        self._onnx_path = str(onnx_path or DefaultPaths.ONNX_INT8_MODEL.value)
        self._tokenizer_dir = str(tokenizer_dir or DefaultPaths.BASELINE_MODEL.value)
        self._model_name = model_name or Models.TEACHER.value
        self._max_length = max_length
        self._lock = threading.Lock()
        self._session: Any = None
        self._tokenizer: Any = None
        self._id2label: dict[int, str] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            self._session = ort.InferenceSession(
                self._onnx_path, providers=["CPUExecutionProvider"]
            )
            self._tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_dir)
            config = AutoConfig.from_pretrained(self._tokenizer_dir)
            raw = getattr(config, "id2label", None) or {}
            self._id2label = {int(k): v for k, v in raw.items()}
            self._loaded = True
            logger.info(
                f"Batch scorer loaded {self._onnx_path} with labels "
                f"{self._id2label} [{self._session.get_providers()}]"
            )

    def score(self, texts: list[str]) -> list[dict[str, Any]]:
        if not texts:
            return []
        self._ensure_loaded()
        probs = predict_probs(
            self._session,
            self._tokenizer,
            texts,
            self._model_name,
            self._max_length,
        )
        return to_predictions(probs, self._id2label)


@dataclass
class BatchStats:
    """Accumulated consumer statistics across one ``run``."""

    batches: int = 0
    items_processed: int = 0
    drain_time_sec: float = 0.0
    score_time_sec: float = 0.0
    total_time_sec: float = 0.0
    fill_ratios: list[float] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)

    @property
    def avg_batch_size(self) -> float:
        return self.items_processed / self.batches if self.batches else 0.0

    @property
    def fill_p50(self) -> float:
        return float(np.percentile(self.fill_ratios, 50)) if self.fill_ratios else 0.0

    @property
    def fill_ratio_avg(self) -> float:
        return statistics.mean(self.fill_ratios) if self.fill_ratios else 0.0

    @property
    def inference_work_sec(self) -> float:
        return self.drain_time_sec + self.score_time_sec

    @property
    def work_reviews_per_sec(self) -> float:
        return (
            self.items_processed / self.inference_work_sec
            if self.inference_work_sec
            else 0.0
        )

    @property
    def reviews_per_sec(self) -> float:
        return (
            self.items_processed / self.total_time_sec if self.total_time_sec else 0.0
        )


class BatchConsumer:
    """Drain -> batch-score -> publish loop with micro-batch triggers.

    One cycle = ``drain(max_batch_size, max_wait_ms)`` then a single
    batch ``score`` then a publish to the result queue. ``run_once`` returns
    ``True`` when a (possibly partial) batch was processed.
    """

    def __init__(
        self,
        queue_backend: QueueBackend,
        scorer: InferenceScorer,
        max_batch_size: int = 32,
        max_wait_ms: int = 100,
    ) -> None:
        self._queue = queue_backend
        self._scorer = scorer
        self._max_batch_size = max_batch_size
        self._max_wait_ms = max_wait_ms
        self._stop = threading.Event()
        self._stats = BatchStats()

    def stop(self) -> None:
        self._stop.set()

    @property
    def stats(self) -> BatchStats:
        return self._stats

    def run_once(self) -> bool:
        if self._stop.is_set():
            return False
        t0 = time.perf_counter()
        messages = self._queue.drain(self._max_batch_size, self._max_wait_ms)
        drain_sec = time.perf_counter() - t0
        self._stats.drain_time_sec += drain_sec
        if not messages:
            return False

        texts = [m.text for m in messages]
        t1 = time.perf_counter()
        predictions = self._scorer.score(texts)
        self._stats.score_time_sec += time.perf_counter() - t1
        if len(predictions) != len(texts):
            raise RuntimeError(
                f"Scorer returned {len(predictions)} predictions for "
                f"{len(texts)} texts; mismatched batch."
            )

        results = [
            {
                **({"id": m.id} if m.id is not None else {}),
                "text": m.text,
                **predictions[i],
            }
            for i, m in enumerate(messages)
        ]
        self._queue.push_results(results)

        self._stats.batches += 1
        self._stats.items_processed += len(messages)
        self._stats.fill_ratios.append(len(messages) / self._max_batch_size)
        self._stats.texts.extend(texts)
        logger.info(
            f"batch {self._stats.batches}: size={len(messages)} "
            f"fill={len(messages) / self._max_batch_size * 100:.0f}% "
            f"score={self._stats.score_time_sec * 1000 / self._stats.batches:.1f}ms "
            f"(cumulative {self._stats.items_processed} reviews)"
        )
        return True

    def run(
        self,
        duration_sec: float | None = None,
        max_batches: int | None = None,
        stop_on_idle_sec: float | None = None,
    ) -> BatchStats:
        deadline = time.monotonic() + duration_sec if duration_sec else None
        last_batch_at = time.monotonic()
        t_start = time.monotonic()
        while not self._stop.is_set():
            if deadline is not None and time.monotonic() >= deadline:
                break
            if max_batches is not None and self._stats.batches >= max_batches:
                break
            processed = self.run_once()
            if processed:
                last_batch_at = time.monotonic()
            elif (
                stop_on_idle_sec is not None
                and time.monotonic() - last_batch_at >= stop_on_idle_sec
            ):
                logger.info(f"Idle for {stop_on_idle_sec}s; stopping.")
                break
        self._stats.total_time_sec = time.monotonic() - t_start
        return self._stats


def _serial_reference(
    scorer: InferenceScorer, texts: list[str], overhead_ms: float
) -> dict[str, float]:
    """Time scoring every review one-at-a-time: the batch-win baseline.

    `overhead_ms` models a fixed per-call cost (e.g. an HTTP /predict or JSON
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


def main() -> None:
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
    args = parser.parse_args()

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

    consumer = BatchConsumer(
        queue_backend,
        scorer,
        max_batch_size=args.max_batch,
        max_wait_ms=args.max_wait_ms,
    )

    if args.mode == "consumer":
        try:
            consumer.run(duration_sec=args.duration, max_batches=args.max_batches)
        except KeyboardInterrupt:
            consumer.stop()
        _log_summary(consumer)
        return

    texts = _load_texts(args.csv, args.samples)
    scorer.score(texts[:1])  # warm the ORT session before timing
    producer_done: list[bool] = [False]

    def _feed() -> None:
        _produce(queue_backend, texts, args.rate)
        producer_done[0] = True

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


def _log_summary(consumer: BatchConsumer) -> None:
    s = consumer.stats
    logger.info(
        f"consumer summary: {s.batches} batches, {s.items_processed} reviews, "
        f"avg batch={s.avg_batch_size:.1f}, fill p50={s.fill_p50 * 100:.0f}%, "
        f"{s.reviews_per_sec:.0f} reviews/sec over {s.total_time_sec:.1f}s"
    )


if __name__ == "__main__":
    main()
