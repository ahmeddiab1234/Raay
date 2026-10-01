"""Shared ASGI harness for the serving middleware unit tests.

The middlewares are plain ASGI callables, so they are driven directly with a
three-message scope instead of through ``TestClient``: that keeps each test
about the middleware's own behaviour (status rewriting, event payload) rather
than about BentoML's routing, and it needs no model graph on disk.
"""

from __future__ import annotations

import asyncio
from typing import Any


def scope(path: str = "/predict", headers: list[tuple[bytes, bytes]] | None = None):
    """A minimal but complete HTTP request scope for a middleware under test."""
    return {
        "type": "http",
        "method": "POST",
        "path": path,
        "headers": headers or [],
        "query_string": b"",
        "scheme": "http",
        "server": ("127.0.0.1", 3000),
        "client": ("127.0.0.1", 50000),
    }


async def _empty_receive() -> dict[str, Any]:
    return {"type": "http.request", "body": b"", "more_body": False}


def run_health_middleware(middleware) -> list[dict[str, Any]]:
    """Drive a middleware for a bodyless ``GET /health`` and capture messages."""

    messages: list[dict[str, Any]] = []

    async def send(message: dict[str, Any]) -> None:
        messages.append(message)

    async def downstream(_scope, _receive, _send) -> None:
        raise AssertionError("downstream app must not be reached for /health")

    instance = middleware(downstream)
    asyncio.run(instance(scope(path="/health"), _empty_receive, send))
    return messages


def run_middleware(
    middleware,
    *,
    scp=None,
    body: bytes = b"",
    status: int = 400,
    request_body: bytes = b"",
    app: Any = None,
) -> list[dict[str, Any]]:
    """Drive ``middleware`` once and return the ASGI messages it emitted."""
    messages: list[dict[str, Any]] = []

    async def send(message: dict[str, Any]) -> None:
        messages.append(message)

    async def receive() -> dict[str, Any]:
        return {
            "type": "http.request",
            "body": request_body,
            "more_body": False,
        }

    async def default_app(_scope, _receive, send) -> None:
        await send({"type": "http.response.start", "status": status, "headers": []})
        await send({"type": "http.response.body", "body": body, "more_body": False})

    instance = middleware(app or default_app)
    asyncio.run(instance(scp or scope(), receive, send))
    return messages


def response_body(messages: list[dict[str, Any]]) -> bytes:
    """Concatenate the body chunks of a captured message list."""
    return b"".join(m.get("body", b"") for m in messages[1:])
