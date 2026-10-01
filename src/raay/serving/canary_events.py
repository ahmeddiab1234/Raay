"""Telemetry event schema + the pairing key that correlates the two rollout sides.

**Pairing key**: the stable and candidate events for one request are paired by
``request_id`` when the request carried one (a client or an upstream gateway set
``X-Request-ID``, which nginx mirror legitimately clones), otherwise by the
SHA-256 of the request ``texts`` -- the only identifier stable and candidate
provably share. ``$request_id`` is useless here by design: a mirror subrequest is
a separate request object, so its ``$request_id`` differs from the main request's,
and ``proxy_set_header`` on the main location is *not* seen by the mirror (it
clones only client-sent headers). Both workers score the identical payload (mirror
clones the body), so content is the honest correlation key.

Two identical review batches inside the TTL hash to the same key; the agent pairs
them FIFO, and the pair is still a true same-input comparison, so this can
undercount pairs but never fabricate a disagreement.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, Field


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
        """Correlate the stable and candidate sides of one incoming request."""
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
