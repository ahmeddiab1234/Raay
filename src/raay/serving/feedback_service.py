"""Customer-service override capture (Phase 6 step 4).

A small internal endpoint where a support agent records that the model got a
review's sentiment wrong during a dispute. It is the closest thing this project
has to production traffic, and therefore the only source of signal that is not a
seeded draw from ``data/processed/test.csv`` -- see the ``caveat`` field on
every drift report, which says exactly that.

Runs as its own process beside the model, the same shape as
``raay.serving.canary_agent`` and for the same reason: a feedback endpoint
inside the BentoML service would drag an auth story, a durable write path and a
second failure mode into the request path that serves ``/predict``.

    python -m uvicorn raay.serving.feedback_service:app --host 127.0.0.1 --port 9101

Endpoints:

* ``POST /feedback`` -- one agent override (see :class:`FeedbackOverride`).
* ``GET /health``   -- liveness.
* ``GET /stats``    -- capture counters.

Two decisions that shape everything downstream
----------------------------------------------

**A confirmation is stored, not rejected.** ``model_label == corrected_label``
returns 200 and writes a row with ``is_correction=False``. It is never training
data -- those rows would only re-teach what the model already does -- but they
are the *denominator* of ``production_error_rate`` in
``reports/feedback_metrics.json``. A tool that only posts disputes produces an
uninterpretable error rate, because the number of times the model was right is
never observed. Making a confirmation as cheap to post as a dispute is what
makes that measurement exist at all.

**Auth is on by default and the endpoint refuses to serve without it.** This is
an unauthenticated write path into a file that ends up in a training set, so it
is a label-poisoning vector, not an internal convenience. The token travels
through a path or the environment, never argv, and is compared with
``hmac.compare_digest``. ``RAAY_FEEDBACK_ALLOW_ANON=1`` is the single escape
hatch and exists for tests.

Durability and concurrency
--------------------------

The sink is an append-only CSV per UTC day under ``data/feedback/raw/``. It is
durable -- a restart loses nothing -- and it is **single-writer**: one uvicorn
worker, guarded by a lock. That is the honest limit of a CSV. Several workers
means a real database; pretending a lock makes a CSV concurrent is how rows get
silently interleaved into unparseable lines.
"""

from __future__ import annotations

import csv
import hashlib
import hmac
import json
import os
import re
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError, field_validator
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from raay.enums.constants import LABELS, DefaultPaths

_TOKEN_ENV = "RAAY_FEEDBACK_TOKEN"
_TOKEN_FILE_ENV = "RAAY_FEEDBACK_TOKEN_FILE"
_ALLOW_ANON_ENV = "RAAY_FEEDBACK_ALLOW_ANON"
_DIR_ENV = "RAAY_FEEDBACK_DIR"

#: Restricted to characters that cannot break the append-only CSV: an
#: ``agent_id`` is written into a row, and a comma or newline in it would forge a
#: second record. It also keys the corroboration groups in
#: ``raay.data.feedback``, so it doubles as the roster identity.
AGENT_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

_BEARER_PREFIX = "bearer "

#: The on-disk schema of ``data/feedback/raw/{date}.csv``. A module constant
#: rather than something derived from the pydantic model, because the sink and
#: the review stage must agree on it, and a field-ordering change derived from
#: the model would silently rewrite historical files instead of appending to
#: them.
RAW_COLUMNS: tuple[str, ...] = (
    "override_id",
    "captured_at",
    "text",
    "model_label",
    "corrected_label",
    "agent_id",
    "model_score",
    "model_version",
    "company",
    "note",
    "guideline_version",
)


class FeedbackOverride(BaseModel):
    """One agent's assertion about one review.

    ``model_label`` is what the model said when the agent looked at it. It is
    recorded rather than recomputed, because the graph that produced it need not
    be the one running when the row is reviewed months later.
    """

    text: str
    model_label: str
    corrected_label: str
    agent_id: str
    captured_at: datetime | None = None
    model_score: float | None = None
    model_version: str = ""
    company: str = ""
    note: str = ""
    guideline_version: str = "v1.0"

    @field_validator("model_label", "corrected_label")
    @classmethod
    def _known_label(cls, value: str) -> str:
        # Deliberately `LABELS` -- the id2label order every ONNX graph carries
        # (configs/train.yaml: labels [positive, negative, neutral]) -- and not
        # the encoding in docs/labeling_guidelines.md section 1, which states
        # positive=2/negative=0/neutral=1 and is inverted relative to every
        # graph in the repo. Following the doc here flips every label silently.
        if value not in LABELS:
            raise ValueError(
                f"label must be one of {list(LABELS)} (the id2label order the graphs "
                f"use), got {value!r}"
            )
        return value

    @field_validator("agent_id")
    @classmethod
    def _safe_agent_id(cls, value: str) -> str:
        if not AGENT_ID_RE.match(value):
            raise ValueError(
                "agent_id must match [A-Za-z0-9._-]{1,64}: it is written into an "
                "append-only CSV and keys the corroboration groups"
            )
        return value

    @field_validator("text")
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be empty or whitespace")
        return value

    @property
    def is_correction(self) -> bool:
        """False when the agent confirmed the model instead of disputing it."""
        return self.model_label != self.corrected_label

    def override_id(self) -> str:
        """Content hash identifying this assertion.

        ``captured_at`` is excluded so a retried POST is idempotent: a CS tool
        that times out and retries must not produce two rows for one assertion.
        The tradeoff is deliberate and real -- a genuine second correction by the
        *same* agent on the same text with the same labels collapses into one
        row. Distinct agents still get distinct ids, which is what the
        two-annotator corroboration rule depends on.
        """
        parts = (
            self.text.strip(),
            self.model_label,
            self.corrected_label,
            self.agent_id,
        )
        return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]


class FeedbackSink:
    """Append-only CSV per UTC day, single writer.

    ``base_dir`` and ``clock`` are injected so tests never touch ``data/``.
    """

    def __init__(
        self,
        base_dir: str = DefaultPaths.FEEDBACK_RAW.value,
        clock: Any = None,
    ) -> None:
        self._base_dir = Path(base_dir)
        self._clock = clock or (lambda: datetime.now(UTC))
        self._lock = threading.Lock()
        self.rows_written = 0

    def now(self) -> datetime:
        """The sink's clock, always timezone-aware UTC."""
        stamp = self._clock()
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=UTC)
        return stamp.astimezone(UTC)

    @staticmethod
    def day_of(stamp: datetime) -> str:
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=UTC)
        return stamp.astimezone(UTC).date().isoformat()

    def path_for(self, day: str) -> Path:
        return self._base_dir / f"{day}.csv"

    def ids_for(self, day: str) -> set[str]:
        """``override_id``s already persisted for ``day`` (empty when absent).

        Read from disk rather than kept in memory precisely because the process
        restarts: an in-memory set would let a replayed POST through as a new
        row after every deploy.
        """
        path = self.path_for(day)
        if not path.exists():
            return set()
        with open(path, newline="", encoding="utf-8") as handle:
            return {row.get("override_id", "") for row in csv.DictReader(handle)}

    def append(self, override: FeedbackOverride, override_id: str) -> None:
        captured_at = override.captured_at or self.now()
        row = {
            "override_id": override_id,
            "captured_at": captured_at.astimezone(UTC).isoformat(),
            "text": override.text,
            "model_label": override.model_label,
            "corrected_label": override.corrected_label,
            "agent_id": override.agent_id,
            "model_score": "" if override.model_score is None else override.model_score,
            "model_version": override.model_version,
            "company": override.company,
            "note": override.note,
            "guideline_version": override.guideline_version,
        }
        with self._lock:
            path = self.path_for(self.day_of(captured_at))
            path.parent.mkdir(parents=True, exist_ok=True)
            # Existence is tested inside the lock: two concurrent first-writes
            # would otherwise both emit a header.
            is_new_file = not path.exists()
            with open(path, "a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(RAW_COLUMNS))
                if is_new_file:
                    writer.writeheader()
                writer.writerow(row)
            self.rows_written += 1


def resolve_token(explicit: str | None = None, env: Any = None) -> str | None:
    """Token from an explicit value, a secret file, or the environment.

    Same shape as ``raay.inference.retrain_trigger.resolve_token``: the secret
    travels through a path or the environment, never argv, so it cannot land in
    ``ps`` output on a shared host. The env var wins over the default file path
    so a container can point anywhere; the default file
    (``airflow_runtime/secrets/feedback_token``) matches the convention
    ``github_dispatch_token`` already set.
    """
    env = os.environ if env is None else env
    if explicit and explicit.strip():
        return explicit.strip()
    path = env.get(_TOKEN_FILE_ENV, "").strip() or DefaultPaths.FEEDBACK_SECRETS.value
    if Path(path).exists():
        return Path(path).read_text().strip() or None
    return env.get(_TOKEN_ENV, "").strip() or None


class FeedbackService:
    """Validates, authenticates and persists agent overrides."""

    def __init__(
        self,
        *,
        token: str | None = None,
        sink: FeedbackSink | None = None,
        allow_anon: bool = False,
    ) -> None:
        if not token and not allow_anon:
            raise RuntimeError(
                "the feedback endpoint is an unauthenticated write path into a "
                f"training set, so it will not serve without a token. Set "
                f"{_TOKEN_ENV}, point {_TOKEN_FILE_ENV} at a secret file, or set "
                f"{_ALLOW_ANON_ENV}=1 for a local/test run only."
            )
        self._token = token or ""
        self._allow_anon = allow_anon
        self._sink = sink or FeedbackSink()
        self.rejected_unauthorized = 0
        self.duplicate_hits = 0
        self.asgi = self._build_app()

    def authorize(self, header: str | None) -> bool:
        """Constant-time bearer check. An anonymous service accepts anything."""
        if self._allow_anon:
            return True
        if not header or not header.lower().startswith(_BEARER_PREFIX):
            return False
        presented = header[len(_BEARER_PREFIX) :].strip()
        if not presented or not self._token:
            return False
        return hmac.compare_digest(presented, self._token)

    def record(self, override: FeedbackOverride) -> tuple[bool, str]:
        """Persist one override. Returns ``(is_new, override_id)``."""
        override_id = override.override_id()
        day = self._sink.day_of(override.captured_at or self._sink.now())
        if override_id in self._sink.ids_for(day):
            self.duplicate_hits += 1
            return False, override_id
        self._sink.append(override, override_id)
        return True, override_id

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "rows_written": self._sink.rows_written,
            "duplicate_hits": self.duplicate_hits,
            "rejected_unauthorized": self.rejected_unauthorized,
            "auth_required": not self._allow_anon,
        }

    def _build_app(self) -> Starlette:
        async def _feedback_route(request: Request) -> JSONResponse:
            if not self.authorize(request.headers.get("authorization")):
                self.rejected_unauthorized += 1
                return JSONResponse({"error": "unauthorized"}, status_code=401)
            try:
                raw = await request.json()
            except json.JSONDecodeError:
                return JSONResponse({"error": "invalid json body"}, status_code=400)
            except Exception:  # noqa: BLE001 # pragma: no cover - defensive
                return JSONResponse({"error": "unreadable body"}, status_code=400)
            try:
                override = FeedbackOverride.model_validate(raw)
            except ValidationError as exc:
                return JSONResponse(
                    {
                        "error": "invalid override",
                        # Only loc/msg/type. `ValidationError.errors()` also
                        # carries a `ctx` holding the original ValueError object
                        # -- not JSON serializable, so returning it verbatim
                        # 500s this handler instead of answering 422 -- and an
                        # `input` field that would echo the submitted review
                        # text back to the caller. Neither belongs in an error
                        # body.
                        "detail": [
                            {"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]}
                            for e in exc.errors()
                        ],
                    },
                    status_code=422,
                )
            is_new, override_id = self.record(override)
            return JSONResponse(
                {
                    "ok": True,
                    "override_id": override_id,
                    # A confirmation is a successful capture, not an error: it is
                    # the denominator of the production error rate.
                    "is_correction": override.is_correction,
                    "duplicate": not is_new,
                }
            )

        async def _health_route(_request: Request) -> JSONResponse:
            return JSONResponse({"status": "healthy"})

        async def _stats_route(_request: Request) -> JSONResponse:
            return JSONResponse(self.stats)

        return Starlette(
            routes=[
                Route("/feedback", _feedback_route, methods=["POST"]),
                Route("/stats", _stats_route, methods=["GET"]),
                Route("/health", _health_route, methods=["GET"]),
            ]
        )


class _LazyFeedbackApp:
    """Defers building the real service until the first request.

    ``canary_agent`` can bind ``app`` at import time because it needs nothing.
    This service refuses to exist without a token, and raising that at *import*
    time would break ``import raay.serving.feedback_service`` for any tooling
    that only wants the classes -- so the failure moves to first use, where it
    is still a refusal to serve rather than a refusal to import.
    """

    def __init__(self) -> None:
        self._service: FeedbackService | None = None
        self._error: str | None = None

    def _resolve(self) -> FeedbackService | None:
        if self._service is not None or self._error is not None:
            return self._service
        from raay.config.env import load_environment

        load_environment()
        try:
            self._service = FeedbackService(
                token=resolve_token(),
                sink=FeedbackSink(
                    base_dir=os.environ.get(_DIR_ENV, "").strip()
                    or DefaultPaths.FEEDBACK_RAW.value
                ),
                allow_anon=os.environ.get(_ALLOW_ANON_ENV, "").strip() == "1",
            )
        except RuntimeError as exc:
            self._error = str(exc)
        return self._service

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        service = self._resolve()
        if service is None:
            response = JSONResponse(
                {"error": self._error or "unavailable"}, status_code=503
            )
            await response(scope, receive, send)
            return
        await service.asgi(scope, receive, send)


#: Process-level app: ``python -m uvicorn raay.serving.feedback_service:app``.
app = _LazyFeedbackApp()
