"""Body buffering + event shape for the telemetry middleware.

Split out of ``telemetry`` so the ASGI plumbing (``TelemetryMiddleware``) is not
also responsible for parsing request and response bytes.

The buffering contract is the delicate part: ``drain_request`` consumes the
incoming body and ``replay_request`` hands the *same* messages back downstream,
so instrumenting a route cannot change what the app sees.
"""

from __future__ import annotations

import json
from typing import Any


async def drain_request(receive: Any) -> list[dict[str, Any]]:
    """Consume the ASGI request body so its ``texts`` can be reported.

    The downstream app receives these same messages again via ``replay_request``,
    so buffering does not change what the app sees.
    """
    chunks: list[dict[str, Any]] = []
    while True:
        message = await receive()
        chunks.append(message)
        if message.get("type") == "http.disconnect" or not message.get("more_body"):
            break
    return chunks


def replay_request(chunks: list[dict[str, Any]]) -> Any:
    """Return a ``receive`` callable that replays the drained request body."""
    index = 0

    async def replay() -> dict[str, Any]:
        nonlocal index
        if index < len(chunks):
            chunk = chunks[index]
            index += 1
            return chunk
        return {"type": "http.disconnect", "body": b"", "more_body": False}

    return replay


def request_texts(chunks: list[dict[str, Any]]) -> list[str]:
    """Extract the ``texts`` the client asked to score (the pairing key input)."""
    body = b"".join(
        c.get("body", b"") for c in chunks if c.get("type") == "http.request"
    )
    try:
        payload = json.loads(body.decode("utf-8", "replace") or "{}")
    except (ValueError, AttributeError, TypeError):
        return []
    texts = payload.get("texts") or []
    return [t for t in texts if isinstance(t, str)]


def predictions_from(body: bytes) -> list[dict[str, Any]]:
    """Pull the ``predictions`` array out of a ``/predict`` response body."""
    try:
        payload = json.loads(body.decode("utf-8", "replace"))
        return payload.get("predictions") or []
    except (ValueError, AttributeError, TypeError):
        return []
