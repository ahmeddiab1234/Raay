"""Per-request prediction events for the canary/shadow agent.

Active only when BOTH ``RAAY_WORKER`` and ``RAAY_TELEMETRY_URL`` are set (the
shadow/canary compose sets them; a normal deployment is untouched). For every
``/predict`` call it records the request id nginx injected (``X-Request-ID``),
whether the conf shadowed the request (``X-Raay-Shadow: 1``), the worker's own
latency, the response status and the prediction payload, then hands the event
to the agent off the serving path.

Two rules make this safe to leave in the production image:

* the POST is fire-and-forget on a daemon thread, so a slow or dead agent
  cannot add user-visible latency and cannot raise into the request;
* the request body is drained and replayed byte-identically, so instrumenting
  a route cannot change what the app sees.

``RAAY_TELEMETRY_URL`` is therefore optional plumbing, not a dependency: with it
unset the middleware is a pass-through with no measurable cost.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
from typing import Any

from loguru import logger

from raay.serving.runtime import model_version
from raay.serving.telemetry_body import (
    drain_request,
    predictions_from,
    replay_request,
    request_texts,
)

TELEMETRY_TIMEOUT_S = 0.5


def _telemetry_target() -> str:
    return os.environ.get("RAAY_TELEMETRY_URL", "").strip()


def _worker_name() -> str:
    return os.environ.get("RAAY_WORKER", "").strip()


def _post_event(event: dict[str, Any]) -> None:
    """POST one telemetry event to the canary-agent; never raises."""
    target = _telemetry_target()
    if not target:
        return
    url = f"{target.rstrip('/')}/ingest"
    try:
        data = json.dumps(event).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=TELEMETRY_TIMEOUT_S) as resp:
            resp.read()
    except Exception:  # noqa: BLE001 - telemetry must never break the main path
        logger.debug("Telemetry POST to {} failed; swallowed", url, exc_info=True)


def _dispatch_event(event: dict[str, Any]) -> None:
    """Hand the event to the agent off the serving path.

    A daemon thread per event keeps the (sync) urllib POST off the event loop
    so a slow or dead agent cannot add user-visible latency. Telemetry is only
    enabled during a shadow/canary rollout, so the thread churn is bounded to
    that window.

    This is the seam the telemetry tests patch: rebinding it on the *façade*
    would not be seen by ``TelemetryMiddleware``, which resolves it in this
    module.
    """
    threading.Thread(target=_post_event, args=(event,), daemon=True).start()


class TelemetryMiddleware:
    """Report per-request prediction events to the canary-agent sidecar.

    A broken or unreachable agent can never change a client-facing response:
    the POST is fire-and-forget and swallowed, and a downstream exception is
    still re-raised after the failure has been reported.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http" or scope.get("path") != "/predict":
            await self.app(scope, receive, send)
            return
        if not (_telemetry_target() and _worker_name()):
            await self.app(scope, receive, send)
            return

        headers = {k.lower(): v for k, v in (scope.get("headers") or [])}
        request_id = headers.get(b"x-request-id", b"").decode("latin1")
        shadow = headers.get(b"x-raay-shadow", b"") == b"1"
        worker = _worker_name()
        started = time.perf_counter()

        # Buffer the request body so its ``texts`` reach the agent (the shadow
        # agent pairs stable/candidate events on content) and replay it to the
        # app untouched.
        request_chunks = await drain_request(receive)
        texts = request_texts(request_chunks)
        replay_receive = replay_request(request_chunks)

        messages: dict[str, Any] = {"chunks": []}

        async def send_wrapper(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                messages["status"] = message["status"]
            await send(message)
            if message["type"] == "http.response.body":
                messages["chunks"].append(message.get("body", b""))

        try:
            await self.app(scope, replay_receive, send_wrapper)
        except Exception as exc:
            self._report(
                request_id=request_id,
                worker=worker,
                shadow=shadow,
                texts=texts,
                status=500,
                body=b"",
                latency_ms=(time.perf_counter() - started) * 1000.0,
                error=str(exc) or "exception",
            )
            raise

        body = b"".join(messages["chunks"])
        self._report(
            request_id=request_id,
            worker=worker,
            shadow=shadow,
            texts=texts,
            status=int(messages.get("status", 500)),
            body=body,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            error=None,
        )

    def _report(
        self,
        *,
        request_id: str,
        worker: str,
        shadow: bool,
        texts: list[str],
        status: int,
        body: bytes,
        latency_ms: float,
        error: str | None,
    ) -> None:
        predictions: list[dict[str, Any]] = []
        if status < 400 and not error:
            predictions = predictions_from(body)
        if status < 400 and not predictions:
            # An empty `texts: []` /predict has nothing worth comparing.
            return
        error = error or (f"http_{status}" if status >= 400 else None)
        _dispatch_event(
            {
                "request_id": request_id,
                "worker": worker,
                "model_version": model_version(),
                "status": status,
                "error": error,
                "latency_ms": round(latency_ms, 3),
                "shadow": shadow,
                "predictions": predictions,
                "texts": texts,
            }
        )
