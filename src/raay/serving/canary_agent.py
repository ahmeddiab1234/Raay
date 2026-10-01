"""Canary / shadow comparison agent for the Raay rollout (Phase 5 step 5).

Collects one prediction event per request from each worker, pairs the stable
and candidate sides, and exports **agreement / latency / error** metrics in
Prometheus text format. ``prom/prometheus`` scrapes this agent; the stage
gates in ``scripts/canary_promote.py`` then query Prometheus.

Model-free by design: the agent never touches the graph, tokenizer or MLflow.
It only sums up events it is handed, so it stays small enough to be fully
covered by hermetic unit tests. The event schema and pairing key live in
``canary_events``; this module is the agent itself.

Endpoints (run from the bento image as
``python -m uvicorn raay.serving.canary_agent:app --host 0.0.0.0 --port 9100``):

* ``POST /ingest`` -- one telemetry event from a worker (see
  ``PredictionEvent``). Workers only POST when their deployment enables
  telemetry (``RAAY_TELEMETRY_URL`` + ``RAAY_WORKER``).
* ``GET /metrics`` -- Prometheus text exposition.
* ``GET /health`` -- liveness.

Pairing semantics
-----------------

A request is *shadowed* when nginx set ``X-Raay-Shadow: 1`` on it -- which the
shadow conf does on **both** the main ``/predict`` hop and the ``/shadow``
mirror hop, so the stable and candidate events for the same request both carry
``shadow=true`` and the agent knows a pair is expected. During shadow, *every*
live request therefore produces exactly two events; the sides agree when their
predicted label sequences match. If a shadowed pair expires with only one side
seen, the missing side is counted as a ``pair_gap`` -- the strongest signal
that a worker stopped receiving traffic (candidate down during shadow). This
works even when the candidate is dead: the stable side's event alone marks the
request as shadowed, so the candidate gap is still counted.

During the weighted canary phases the conf sends no shadow header, so no
request is shadowed and nothing is counted as a gap; the agent simply
aggregates per-worker latency/error rates, which is exactly what those stage
gates need. Unshadowed events are never paired -- pairing would only burn the
pair buffer on requests that have no counterpart.

The agent is deliberately stateless across restarts (counters live in memory
and reset on restart). Stage gates must therefore evaluate over a rate window
and treat an empty series as INCONCLUSIVE, never as a pass.
"""

from __future__ import annotations

import os
import time
from collections import deque
from typing import Any

from prometheus_client import generate_latest
from starlette.applications import Starlette

from raay.serving.canary_app import build_app
from raay.serving.canary_events import PredictionEvent, _Pair
from raay.serving.canary_metrics import build_metrics

_DEFAULT_PAIR_TTL_S = 30.0
_DEFAULT_MAX_PAIRS = 4096


class CanaryAgent:
    """Pairs worker telemetry events and exports Prometheus metrics."""

    def __init__(
        self,
        *,
        pair_ttl_s: float | None = None,
        max_pairs: int | None = None,
        clock: Any = None,
    ) -> None:
        self._pair_ttl_s = pair_ttl_s or float(
            os.environ.get("RAAY_AGENT_PAIR_TTL", _DEFAULT_PAIR_TTL_S)
        )
        self._max_pairs = max_pairs or int(
            os.environ.get("RAAY_AGENT_MAX_PAIRS", _DEFAULT_MAX_PAIRS)
        )
        self._clock = clock or time.monotonic
        self._pairs: dict[str, _Pair] = {}
        self._order: deque[str] = deque()
        metrics = build_metrics()
        self._registry = metrics.registry
        self._ingest = metrics.ingest
        self._errors = metrics.errors
        self._latency = metrics.latency
        self._agreement = metrics.agreement
        self._pair_gap = metrics.pair_gap
        self.asgi = self._build_app()

    # ------------------------------------------------------------- ingestion

    def ingest(self, event: PredictionEvent) -> None:
        """Record one telemetry event and pair it where possible."""
        self._expire()

        self._ingest.labels(worker=event.worker).inc()
        if event.is_error:
            self._errors.labels(worker=event.worker).inc()
        if event.latency_ms is not None and event.latency_ms >= 0:
            self._latency.labels(worker=event.worker).observe(event.latency_ms / 1000.0)

        key = event.pair_key
        if not key or not (event.shadow or event.request_id):
            return

        pair = self._pairs.get(key)
        if pair is None:
            if len(self._pairs) >= self._max_pairs:
                self._evict_oldest()
            pair = _Pair(key, self._clock())
            self._pairs[key] = pair
            self._order.append(key)
        pair.add(event)

        if "stable" in pair.events and "candidate" in pair.events:
            self._resolve(pair)

    def _expire(self) -> None:
        now = self._clock()
        while self._order:
            oldest_id = self._order[0]
            pair = self._pairs.get(oldest_id)
            if pair is None:
                self._order.popleft()
                continue
            if now - pair.created < self._pair_ttl_s:
                break
            self._order.popleft()
            self._close(pair, count_gap=True)

    def _evict_oldest(self) -> None:
        if not self._order:
            return
        oldest_id = self._order.popleft()
        pair = self._pairs.pop(oldest_id, None)
        if pair is not None:
            self._close(pair)

    # ---------------------------------------------------------------- pairing

    def _resolve(self, pair: _Pair) -> None:
        stable = pair.events.get("stable")
        candidate = pair.events.get("candidate")
        if not (pair.shadowed and stable is not None and candidate is not None):
            self._close(pair)
            return
        stable_labels = stable.labels
        candidate_labels = candidate.labels
        if stable_labels and candidate_labels:
            agree = stable_labels == candidate_labels
            self._agreement.labels(status="agree" if agree else "disagree").inc()
        self._close(pair)

    def _close(self, pair: _Pair, *, count_gap: bool = False) -> None:
        self._pairs.pop(pair.key, None)
        if not count_gap or not pair.shadowed:
            return
        if "stable" in pair.events and "candidate" not in pair.events:
            self._pair_gap.labels(worker="candidate").inc()
        elif "candidate" in pair.events and "stable" not in pair.events:
            self._pair_gap.labels(worker="stable").inc()

    # ---------------------------------------------------------------- metrics

    def render_metrics(self) -> bytes:
        """Prometheus text exposition for this agent's own registry."""

        return generate_latest(self._registry)

    # ------------------------------------------------------------------ app

    def _build_app(self) -> Starlette:
        return build_app(self)


# Module-level pair used when the agent runs as its own process:
#   python -m uvicorn raay.serving.canary_agent:app --host 0.0.0.0 --port 9100
app = CanaryAgent().asgi
