"""Starlette app for the feedback capture endpoint.

``FeedbackService`` holds the auth decision and the counters; ``_LazyFeedbackApp``
is the process-level ASGI object ``uvicorn`` binds. The laziness is load-bearing:
the service refuses to exist without a token, and raising that at *import* time
would break ``import raay.serving.feedback_service`` for any tooling that only
wants the classes -- so the failure moves to first use, where it is still a
refusal to serve rather than a refusal to import.
"""

from __future__ import annotations

import hmac
import json
import os
from typing import Any

from pydantic import ValidationError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from raay.enums.constants import DefaultPaths
from raay.serving.feedback_auth import (
    _ALLOW_ANON_ENV,
    _DIR_ENV,
    _TOKEN_ENV,
    _TOKEN_FILE_ENV,
    resolve_token,
)
from raay.serving.feedback_schema import FeedbackOverride
from raay.serving.feedback_sink import FeedbackSink

_BEARER_PREFIX = "bearer "


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
