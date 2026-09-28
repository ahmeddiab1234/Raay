"""Canary / shadow comparison agent for the Raay rollout (Phase 5 step 5).

Collects one prediction event per request from each worker, pairs the stable
and candidate sides, and exports **agreement / latency / error** metrics in
Prometheus text format. ``prom/prometheus`` scrapes this agent; the stage
gates in ``scripts/canary_promote.py`` then query Prometheus.

Model-free by design: the agent never touches the graph, tokenizer or MLflow.
It only sums up events it is handed, so it stays small enough to be fully
covered by hermetic unit tests.

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

**Pairing key**: the two events are paired by ``request_id`` when the request
carried one (a client or an upstream gateway set ``X-Request-ID``, which nginx
mirror legitimately clones), otherwise by the SHA-256 of the request ``texts``
-- the only identifier stable and candidate provably share. ``$request_id`` is
useless here by design: a mirror subrequest is a separate request object, so
its ``$request_id`` differs from the main request's, and ``proxy_set_header``
on the main location is *not* seen by the mirror (it clones only client-sent
headers). Both workers score the identical payload (mirror clones the body),
so content is the honest correlation key. Two identical review batches inside
the 30 s TTL hash to the same key; the agent pairs them FIFO, and the pair is
still a true same-input comparison, so this can undercount pairs but never
fabricate a disagreement.

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

import hashlib
import json
import os
import time
from collections import deque
from typing import Any

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field, ValidationError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route

_DEFAULT_PAIR_TTL_S = 30.0
_DEFAULT_MAX_PAIRS = 4096


class PredictionEvent(BaseModel):
    """One telemetry event emitted by a worker per ``/predict`` call.

    ``worker`` plus the pairing key (``request_id`` or the request ``texts``)
    pair the two sides. ``shadow`` is true only while the shadow conf is live
    (nginx set ``X-Raay-Shadow: 1``), which is when the agent expects a
    stable+candidate pair for the key.
    """

    request_id: str = ""
    worker: str = "unknown"
    model_version: str = ""
    status: int = 200
    error: str | None = None
    latency_ms: float | None = None
    shadow: bool = False
    predictions: list[dict[str, Any]] = Field(default_factory=list)
    texts: list[str] = Field(default_factory=list)

    @property
    def is_error(self) -> bool:
        return self.status >= 400 or bool(self.error)

    @property
    def labels(self) -> list[str]:
        return [str(p.get("label", "")) for p in self.predictions]

    @property
    def pair_key(self) -> str:
        """Correlate the stable and candidate sides of one incoming request.

        ``request_id`` wins when present (a client or upstream gateway supplied
        it, so both workers saw the identical value). Otherwise the SHA-256 of
        the request ``texts`` -- the body is cloned by the nginx mirror, so
        this is the value stable and candidate provably share.
        """
        if self.request_id:
            return self.request_id
        if self.texts:
            digest = hashlib.sha256(
                json.dumps(self.texts, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest()
            return digest
        return ""


class _Pair:
    """One pairing key awaiting stable + candidate events."""

    __slots__ = ("created", "events", "key", "shadowed")

    def __init__(self, key: str, now: float) -> None:
        self.key = key
        self.events: dict[str, PredictionEvent] = {}
        self.shadowed = False
        self.created = now

    def add(self, event: PredictionEvent) -> None:
        self.events[event.worker] = event
        if event.shadow:
            self.shadowed = True


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
        registry = CollectorRegistry()
        self._registry = registry
        self._ingest = Counter(
            "raay_ingest_total",
            "Prediction events received, per worker.",
            ("worker",),
            registry=registry,
        )
        self._errors = Counter(
            "raay_errors_total",
            "Prediction events that errored, per worker.",
            ("worker",),
            registry=registry,
        )
        self._latency = Histogram(
            "raay_latency_seconds",
            "Worker-side /predict latency.",
            ("worker",),
            registry=registry,
        )
        self._agreement = Counter(
            "raay_agreement_total",
            "Paired shadow requests by agreement status.",
            ("status",),
            registry=registry,
        )
        self._pair_gap = Counter(
            "raay_pair_gap_total",
            "Shadowed request ids that expired with one side missing, by the "
            "worker that *did not* arrive.",
            ("worker",),
            registry=registry,
        )
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
        async def _ingest_route(request: Request) -> JSONResponse:
            try:
                raw = await request.json()
            except json.JSONDecodeError:
                return JSONResponse({"error": "invalid json body"}, status_code=400)
            except Exception:  # noqa: BLE001 # pragma: no cover - defensive
                return JSONResponse({"error": "unreadable body"}, status_code=400)
            try:
                event = PredictionEvent.model_validate(raw)
            except ValidationError:
                return JSONResponse({"error": "invalid event"}, status_code=422)
            self.ingest(event)
            return JSONResponse({"ok": True})

        async def _metrics_route(_request: Request) -> PlainTextResponse:
            return PlainTextResponse(self.render_metrics())

        async def _health_route(_request: Request) -> JSONResponse:
            return JSONResponse({"status": "healthy"})

        return Starlette(
            routes=[
                Route("/ingest", _ingest_route, methods=["POST"]),
                Route("/metrics", _metrics_route, methods=["GET"]),
                Route("/health", _health_route, methods=["GET"]),
            ]
        )


# Module-level pair used when the agent runs as its own process:
#   python -m uvicorn raay.serving.canary_agent:app --host 0.0.0.0 --port 9100
app = CanaryAgent().asgi
