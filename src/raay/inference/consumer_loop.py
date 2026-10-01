"""The drain -> batch-score -> publish loop."""

from __future__ import annotations

import threading
import time

from loguru import logger

from raay.inference.consumer_queues import QueueBackend
from raay.inference.consumer_scorer import InferenceScorer
from raay.inference.consumer_stats import BatchStats


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
