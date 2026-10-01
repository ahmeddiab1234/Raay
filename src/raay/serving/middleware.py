"""Route-shape ASGI middleware for the serving app.

Two behaviours BentoML does not give us for free and the deploy/verification
chain depends on:

* ``GET /health`` must answer 200 on the exact bare path (a Starlette ``Mount``
  307-redirects to a trailing slash, which a Docker ``HEALTHCHECK`` following
  redirects still survives but ``curl`` assertions do not).
* A pydantic validation failure must answer 422, not BentoML 1.4's 400, because
  the phase checklist and the staging deploy both assert 422.

Both are mounted by :mod:`raay.serving.app`, and both report ``model_version``
so an operator can answer "which graph is this container serving?" from the
first request they make.
"""

from __future__ import annotations

import json
from typing import Any

from starlette.responses import JSONResponse

from raay.serving.runtime import model_version

_VALIDATION_ERROR_MARKER = "validation error for"


class HealthRouteMiddleware:
    """Serve ``GET /health`` as a top-level route returning the liveness body.

    BentoML only lets us mount ASGI apps at a prefix, and a Starlette ``Mount``
    307-redirects the bare prefix to a trailing slash (``/health`` ->
    ``/health/``); a middleware short-circuits the exact path instead, so the
    Docker ``HEALTHCHECK`` and ``curl`` both see a plain 200 on ``/health``.

    The body also carries ``model_version``: this is the route an orchestrator
    and an operator hit first, so it is the cheapest place to answer "which
    graph is this container actually serving?".
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") == "http" and scope.get("path") == "/health":
            response = JSONResponse(
                {"status": "healthy", "model_version": model_version()}
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


class Validation422Middleware:
    """Rewrite BentoML's pydantic-validation ``400`` responses to ``422``.

    BentoML 1.4 maps a ``pydantic.ValidationError`` to ``400``; the phase
    checklist (and AGENTS.md) promise ``422`` for malformed/missing payloads.
    This buffers a ``400`` response body and re-emits status ``422`` when the
    payload matches the framework's validation-error shape (``error``
    containing "validation error for" plus a ``detail`` list).
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        buffer: dict[str, Any] = {}

        async def send_wrapper(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                if message["status"] != 400:
                    await send(message)
                    return
                buffer["status"] = message["status"]
                buffer["headers"] = message["headers"]
                buffer["chunks"] = []
                return
            if "chunks" not in buffer:
                await send(message)
                return
            if message["type"] == "http.response.body":
                buffer["chunks"].append(message.get("body", b""))
                if not message.get("more_body"):
                    await self._emit(
                        buffer["headers"], b"".join(buffer["chunks"]), send
                    )
            return

        await self.app(scope, receive, send_wrapper)

    async def _emit(self, headers: Any, body: bytes, send: Any) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": 422 if _is_validation_error(body) else 400,
                "headers": headers,
            }
        )
        await send({"type": "http.response.body", "body": body, "more_body": False})


def _is_validation_error(body: bytes) -> bool:
    """Does this buffered 400 body carry a pydantic validation failure?

    Shape-matched rather than status-matched on purpose: a 400 that is *not* a
    validation error (a real application error) must keep its 400, or a genuine
    bad request would be reported as a client schema problem.
    """
    try:
        payload = json.loads(body)
        error: Any = payload.get("error")
        return (
            isinstance(error, str) and _VALIDATION_ERROR_MARKER in error
        ) and isinstance(payload.get("detail"), list)
    except (ValueError, TypeError, AttributeError):
        return False
