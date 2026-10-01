"""The canary/shadow telemetry middleware and its fire-and-forget POST.

The dispatch seam is ``raay.serving.telemetry._dispatch_event`` rather than a
name on the ``raay.serving.serve`` facade: the middleware calls its own module
global, so patching the facade would leave a real (and unreachable) POST in the
test.
"""

import pytest
from serving_harness import run_middleware, scope

import raay.serving.telemetry as telemetry_mod
from raay.serving.telemetry import TelemetryMiddleware

TELEMETRY_ENV = {
    "RAAY_TELEMETRY_URL": "http://raay-canary-agent:9100",
    "RAAY_WORKER": "stable",
}


def _capture_events(monkeypatch):
    events = []

    def capture(event):
        events.append(event)

    monkeypatch.setattr(telemetry_mod, "_dispatch_event", capture)
    return events


def _emit(
    monkeypatch,
    *,
    body=b'{"predictions": [{"label": "positive", "score": 0.9}]}',
    status=200,
    request_id=b"r-1",
    shadow=False,
    path="/predict",
    request_body=b"",
    worker="stable",
    app=None,
):
    for key, value in {**TELEMETRY_ENV, "RAAY_WORKER": worker}.items():
        monkeypatch.setenv(key, value)
    headers = [(b"x-request-id", request_id)]
    if shadow:
        headers.append((b"x-raay-shadow", b"1"))
    return run_middleware(
        TelemetryMiddleware,
        scp=scope(path=path, headers=headers),
        body=body,
        status=status,
        request_body=request_body,
        app=app,
    )


def test_telemetry_off_by_default_passes_through(monkeypatch):
    monkeypatch.delenv("RAAY_TELEMETRY_URL", raising=False)
    monkeypatch.delenv("RAAY_WORKER", raising=False)
    events = _capture_events(monkeypatch)
    messages = _emit(monkeypatch, worker="")
    monkeypatch.delenv("RAAY_WORKER", raising=False)
    assert messages[0]["status"] == 200  # downstream ran, response passed through
    assert events == []


def test_telemetry_emits_prediction_event(monkeypatch):
    monkeypatch.setenv("RAAY_MODEL_VERSION", "int8-687d587004c6")
    events = _capture_events(monkeypatch)
    _emit(monkeypatch, request_id=b"req-42")
    assert len(events) == 1
    event = events[0]
    assert event["request_id"] == "req-42"
    assert event["worker"] == "stable"
    assert event["model_version"] == "int8-687d587004c6"
    assert event["status"] == 200
    assert event["shadow"] is False
    assert event["error"] is None
    assert event["latency_ms"] >= 0.0
    assert event["predictions"] == [{"label": "positive", "score": 0.9}]


def test_telemetry_marks_shadowed_requests(monkeypatch):
    events = _capture_events(monkeypatch)
    _emit(monkeypatch, shadow=True, worker="candidate")
    assert len(events) == 1
    assert events[0]["shadow"] is True
    assert events[0]["worker"] == "candidate"


def test_telemetry_carries_request_texts_for_pairing(monkeypatch):
    events = _capture_events(monkeypatch)
    _emit(
        monkeypatch,
        request_body='{"texts": ["منتج رائع", "وصلت متأخرة"]}'.encode(),
        shadow=True,
        worker="candidate",
    )
    assert len(events) == 1
    assert events[0]["texts"] == ["منتج رائع", "وصلت متأخرة"]
    assert events[0]["predictions"] == [{"label": "positive", "score": 0.9}]


def test_telemetry_skips_empty_predictions(monkeypatch):
    events = _capture_events(monkeypatch)
    _emit(monkeypatch, body=b'{"predictions": []}')
    assert events == []


def test_telemetry_still_reports_errors_without_predictions(monkeypatch):
    events = _capture_events(monkeypatch)
    _emit(monkeypatch, body=b'{"error": "boom"}', status=500)
    assert len(events) == 1
    assert events[0]["status"] == 500
    assert events[0]["error"] == "http_500"
    assert events[0]["predictions"] == []


def test_telemetry_ignores_non_predict_paths(monkeypatch):
    events = _capture_events(monkeypatch)
    _emit(monkeypatch, path="/health")
    assert events == []


def test_telemetry_rethrows_downstream_exceptions_after_reporting(monkeypatch):
    events = _capture_events(monkeypatch)

    async def boom(_scope, _receive, _send):
        raise RuntimeError("graph exploded")

    with pytest.raises(RuntimeError, match="graph exploded"):
        _emit(monkeypatch, app=boom)
    assert len(events) == 1
    assert events[0]["status"] == 500
    assert events[0]["error"] == "graph exploded"


def test_telemetry_post_failures_are_swallowed(monkeypatch):
    monkeypatch.setenv("RAAY_TELEMETRY_URL", "http://127.0.0.1:9")

    def boom(*_args, **_kwargs):
        raise OSError("agent unreachable")

    monkeypatch.setattr(telemetry_mod.urllib.request, "urlopen", boom)
    # Must not raise: telemetry can never break user traffic.
    telemetry_mod._post_event({"request_id": "r1"})


def test_telemetry_post_is_a_noop_without_a_target(monkeypatch):
    monkeypatch.delenv("RAAY_TELEMETRY_URL", raising=False)

    def boom(*_args, **_kwargs):  # pragma: no cover - must never run
        raise AssertionError("no target means no request")

    monkeypatch.setattr(telemetry_mod.urllib.request, "urlopen", boom)
    telemetry_mod._post_event({"request_id": "r1"})
