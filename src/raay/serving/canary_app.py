"""The agent's three routes.

Small on its own; kept separate so ``canary_agent`` is about pairing and counting
and nothing else.
"""

from __future__ import annotations

import json
from typing import Protocol

from pydantic import ValidationError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route

from raay.serving.canary_events import PredictionEvent


class _Agent(Protocol):
    """The slice of the agent these routes touch."""

    def ingest(self, event: PredictionEvent) -> None: ...

    def render_metrics(self) -> bytes: ...


def build_app(agent: _Agent) -> Starlette:
    """Wire ``/ingest``, ``/metrics`` and ``/health`` onto ``agent``."""

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
        agent.ingest(event)
        return JSONResponse({"ok": True})

    async def _metrics_route(_request: Request) -> PlainTextResponse:
        return PlainTextResponse(agent.render_metrics())

    async def _health_route(_request: Request) -> JSONResponse:
        return JSONResponse({"status": "healthy"})

    return Starlette(
        routes=[
            Route("/ingest", _ingest_route, methods=["POST"]),
            Route("/metrics", _metrics_route, methods=["GET"]),
            Route("/health", _health_route, methods=["GET"]),
        ]
    )
