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
    uv run python -m raay.inference.batch_consumer --mode producer \\
        --csv data/processed/test.csv --samples 200 --rate 100

    # benchmark: feed a queue live, measure fill efficiency + reviews/sec and
    # compare against one-at-a-time scoring; writes reports/queue_benchmark.json
    uv run python -m raay.inference.batch_consumer --mode benchmark \\
        --csv data/processed/test.csv --samples 200 --duration 15

Env-tunable like the serving endpoint: ``RAAY_ONNX_PATH`` (default
``models/onnx/model_int8.onnx``), ``RAAY_TOKENIZER_DIR``, ``RAAY_MODEL_NAME``,
``RAAY_MAX_LENGTH``.

Implementation lives in siblings -- ``consumer_messages`` (payloads),
``consumer_queues`` (memory + Redis transports), ``consumer_scorer`` (the lazy
INT8 scorer), ``consumer_stats`` (run statistics), ``consumer_loop`` (the
drain/score/publish cycle), ``consumer_benchmark`` (serial reference + report),
``consumer_cli`` (this CLI) -- re-exported here so callers keep one import path.
"""

from __future__ import annotations

from raay.inference.consumer_benchmark import (
    _build_queue_from_args,
    _load_texts,
    _log_summary,
    _produce,
    _run_report,
    _serial_reference,
)
from raay.inference.consumer_cli import main
from raay.inference.consumer_loop import BatchConsumer
from raay.inference.consumer_messages import (
    ReviewMessage,
    decode_message,
    encode_message,
    encode_result,
)
from raay.inference.consumer_queues import MemoryQueue, QueueBackend, RedisQueue
from raay.inference.consumer_scorer import InferenceScorer
from raay.inference.consumer_stats import BatchStats

__all__ = [
    "BatchConsumer",
    "BatchStats",
    "InferenceScorer",
    "MemoryQueue",
    "QueueBackend",
    "RedisQueue",
    "ReviewMessage",
    "decode_message",
    "encode_message",
    "encode_result",
    "main",
]
# The benchmark helpers are private, but they are the documented seam the unit
# tests drive (e.g. ``_serial_reference`` against a fake scorer), so they stay
# importable from the façade as well.
_PRIVATE_REEXPORTS = (
    _build_queue_from_args,
    _load_texts,
    _log_summary,
    _produce,
    _run_report,
    _serial_reference,
)


if __name__ == "__main__":
    main()
