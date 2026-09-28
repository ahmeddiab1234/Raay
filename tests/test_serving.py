import asyncio
import json

import numpy as np
import pytest
import torch
from starlette.testclient import TestClient
from transformers import BertConfig, BertForSequenceClassification

import raay.serving.serve as serve_mod
from raay.inference.export_onnx import export_to_onnx
from raay.serving.serve import (
    HealthRouteMiddleware,
    PredictResponse,
    TelemetryMiddleware,
    Validation422Middleware,
    _resolve_onnx_path,
    model_version,
    predict_probs,
    softmax,
    svc,
    to_predictions,
)


@pytest.fixture(scope="module")
def client():
    """In-process ASGI client for the real BentoML app.

    ``TestClient`` must be used as a context manager: BentoML instantiates the
    inner service in the Starlette lifespan (``create_instance``), so without it
    every route resolves against a ``None`` instance and 500s. No model graph
    is needed here -- ``/health`` short-circuits in
    ``HealthRouteMiddleware`` and malformed payloads are rejected by the input
    model before ``RaaySV._ensure_loaded()`` runs.
    """
    with TestClient(svc.to_asgi()) as test_client:
        yield test_client


def test_asgi_health_returns_200_without_redirect(client, monkeypatch):
    monkeypatch.setenv("RAAY_MODEL_VERSION", "int8-687d587004c6")
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "model_version": "int8-687d587004c6",
    }


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({}, id="missing-field"),
        pytest.param({"texts": 5}, id="not-a-list"),
        pytest.param({"texts": None}, id="null"),
        pytest.param({"texts": [1, 2]}, id="non-string-items"),
    ],
)
def test_asgi_malformed_payloads_return_422(client, payload):
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    body = response.json()
    assert "validation error for" in body["error"]
    assert isinstance(body["detail"], list) and body["detail"]


def test_service_has_middlewares():
    middlewares = [cls for cls, _ in svc.middlewares]
    assert HealthRouteMiddleware in middlewares
    assert Validation422Middleware in middlewares


def test_health_middleware_returns_status_json(monkeypatch):
    monkeypatch.delenv("RAAY_MODEL_VERSION", raising=False)

    async def _passthrough(_scope, _receive, _send):
        raise AssertionError("downstream app must not be reached for /health")

    middleware = HealthRouteMiddleware(_passthrough)
    messages = []

    async def send(message):
        messages.append(message)

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/health",
        "headers": [],
        "query_string": b"",
        "scheme": "http",
        "server": ("127.0.0.1", 3000),
        "client": ("127.0.0.1", 50000),
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    asyncio.run(middleware(scope, receive, send))
    assert messages[0]["type"] == "http.response.start"
    assert messages[0]["status"] == 200
    body = json.loads(b"".join(m["body"] for m in messages[1:]))
    assert body == {"status": "healthy", "model_version": "unversioned"}


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [
        pytest.param("int8-687d587004c6", "int8-687d587004c6", id="int8"),
        pytest.param("int8-newvalue1234", "int8-newvalue1234", id="other"),
        pytest.param("  spaced  ", "spaced", id="surrounding-whitespace"),
        pytest.param("", "unversioned", id="empty"),
        pytest.param("   ", "unversioned", id="whitespace-only"),
    ],
)
def test_model_version_reads_env_at_call_time(monkeypatch, env_value, expected):
    """Read per call, not at import, so a worker restart picks up the value.

    A blank value falls back to the literal ``unversioned`` rather than
    reporting an empty string that would read as a real, empty version.
    """
    monkeypatch.setenv("RAAY_MODEL_VERSION", env_value)
    assert model_version() == expected


def test_model_version_defaults_when_unset(monkeypatch):
    monkeypatch.delenv("RAAY_MODEL_VERSION", raising=False)
    assert model_version() == "unversioned"


def test_predict_response_carries_model_version(monkeypatch):
    """The response echoes the stamp, so a client needs no second call."""
    monkeypatch.setenv("RAAY_MODEL_VERSION", "int8-687d587004c6")
    response = PredictResponse(predictions=[])
    assert response.model_version == "int8-687d587004c6"


def test_predict_response_model_version_is_optional_field():
    """Additive field: old clients ignoring it keep working, and an explicit
    ``unversioned`` default keeps the schema stable if it is ever deserialized
    from an older payload."""
    dumped = PredictResponse(predictions=[]).model_dump()
    assert "model_version" in dumped
    assert PredictResponse.model_fields["model_version"].is_required() is False


def _run_middleware(middleware_cls, body, status=400):
    messages = []

    async def send(message):
        messages.append(message)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def downstream(_scope, _receive, send):
        await send({"type": "http.response.start", "status": status, "headers": []})
        await send({"type": "http.response.body", "body": body, "more_body": False})

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/predict",
        "headers": [],
        "query_string": b"",
        "scheme": "http",
        "server": ("127.0.0.1", 3000),
        "client": ("127.0.0.1", 50000),
    }
    middleware = middleware_cls(downstream)
    asyncio.run(middleware(scope, receive, send))
    return messages


def test_validation_422_middleware_rewrites_pydantic_400():
    body = (
        b'{"error": "1 validation error for PredictRequest", '
        b'"detail": [{"loc": ["texts"], "msg": "Field required", "type": "missing"}]}'
    )
    messages = _run_middleware(Validation422Middleware, body)
    assert messages[0]["status"] == 422
    assert b"".join(m["body"] for m in messages[1:]) == body


def test_validation_422_middleware_leaves_non_validation_400():
    body = b'{"error": "task_id is required"}'
    messages = _run_middleware(Validation422Middleware, body)
    assert messages[0]["status"] == 400


# ------------------------------------------------------- telemetry middleware


def test_service_has_telemetry_middleware():
    middlewares = [cls for cls, _ in svc.middlewares]
    assert TelemetryMiddleware in middlewares


def _run_telemetry(
    body=b'{"predictions": [{"label": "positive", "score": 0.9}]}',
    status=200,
    request_id=b"r-1",
    shadow=False,
    path="/predict",
    app=None,
    request_body=b"",
):
    messages = []

    async def send(message):
        messages.append(message)

    async def receive():
        return {"type": "http.request", "body": request_body, "more_body": False}

    async def default_app(_scope, _receive, send):
        await send({"type": "http.response.start", "status": status, "headers": []})
        await send({"type": "http.response.body", "body": body, "more_body": False})

    headers = [(b"x-request-id", request_id)]
    if shadow:
        headers.append((b"x-raay-shadow", b"1"))
    scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "headers": headers,
        "query_string": b"",
        "scheme": "http",
        "server": ("127.0.0.1", 3000),
        "client": ("127.0.0.1", 50000),
    }
    middleware = TelemetryMiddleware(app or default_app)
    asyncio.run(middleware(scope, receive, send))
    return messages


def _capture_events(monkeypatch):
    events = []

    def capture(event):
        events.append(event)

    monkeypatch.setattr(serve_mod, "_dispatch_event", capture)
    return events


def test_telemetry_off_by_default_passes_through(monkeypatch):
    monkeypatch.delenv("RAAY_TELEMETRY_URL", raising=False)
    monkeypatch.delenv("RAAY_WORKER", raising=False)
    events = _capture_events(monkeypatch)
    messages = _run_telemetry()
    assert messages[0]["status"] == 200  # downstream ran, response passed through
    assert events == []


def test_telemetry_emits_prediction_event(monkeypatch):
    monkeypatch.setenv("RAAY_TELEMETRY_URL", "http://raay-canary-agent:9100")
    monkeypatch.setenv("RAAY_WORKER", "stable")
    monkeypatch.setenv("RAAY_MODEL_VERSION", "int8-687d587004c6")
    events = _capture_events(monkeypatch)
    _run_telemetry(request_id=b"req-42")
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
    monkeypatch.setenv("RAAY_TELEMETRY_URL", "http://raay-canary-agent:9100")
    monkeypatch.setenv("RAAY_WORKER", "candidate")
    events = _capture_events(monkeypatch)
    _run_telemetry(shadow=True)
    assert len(events) == 1
    assert events[0]["shadow"] is True
    assert events[0]["worker"] == "candidate"


def test_telemetry_carries_request_texts_for_pairing(monkeypatch):
    monkeypatch.setenv("RAAY_TELEMETRY_URL", "http://raay-canary-agent:9100")
    monkeypatch.setenv("RAAY_WORKER", "candidate")
    events = _capture_events(monkeypatch)
    _run_telemetry(
        request_body='{"texts": ["منتج رائع", "وصلت متأخرة"]}'.encode(),
        shadow=True,
    )
    assert len(events) == 1
    assert events[0]["texts"] == ["منتج رائع", "وصلت متأخرة"]
    assert events[0]["predictions"] == [{"label": "positive", "score": 0.9}]


def test_telemetry_skips_empty_predictions(monkeypatch):
    monkeypatch.setenv("RAAY_TELEMETRY_URL", "http://raay-canary-agent:9100")
    monkeypatch.setenv("RAAY_WORKER", "stable")
    events = _capture_events(monkeypatch)
    _run_telemetry(body=b'{"predictions": []}')
    assert events == []


def test_telemetry_still_reports_errors_without_predictions(monkeypatch):
    monkeypatch.setenv("RAAY_TELEMETRY_URL", "http://raay-canary-agent:9100")
    monkeypatch.setenv("RAAY_WORKER", "stable")
    events = _capture_events(monkeypatch)
    _run_telemetry(body=b'{"error": "boom"}', status=500)
    assert len(events) == 1
    assert events[0]["status"] == 500
    assert events[0]["error"] == "http_500"
    assert events[0]["predictions"] == []


def test_telemetry_ignores_non_predict_paths(monkeypatch):
    monkeypatch.setenv("RAAY_TELEMETRY_URL", "http://raay-canary-agent:9100")
    monkeypatch.setenv("RAAY_WORKER", "stable")
    events = _capture_events(monkeypatch)
    _run_telemetry(path="/health")
    assert events == []


def test_telemetry_rethrows_downstream_exceptions_after_reporting(monkeypatch):
    monkeypatch.setenv("RAAY_TELEMETRY_URL", "http://raay-canary-agent:9100")
    monkeypatch.setenv("RAAY_WORKER", "stable")
    events = _capture_events(monkeypatch)

    async def boom(_scope, _receive, _send):
        raise RuntimeError("graph exploded")

    with pytest.raises(RuntimeError, match="graph exploded"):
        _run_telemetry(app=boom)
    assert len(events) == 1
    assert events[0]["status"] == 500
    assert events[0]["error"] == "graph exploded"


def test_telemetry_post_failures_are_swallowed(monkeypatch):
    monkeypatch.setenv("RAAY_TELEMETRY_URL", "http://127.0.0.1:9")

    def boom(*_args, **_kwargs):
        raise OSError("agent unreachable")

    monkeypatch.setattr(serve_mod.urllib.request, "urlopen", boom)
    # Must not raise: telemetry can never break user traffic.
    serve_mod._post_event({"request_id": "r1"})


def test_resolve_uses_raay_onnx_path_override():
    env = {"RAAY_ONNX_PATH": "/tmp/override.onnx"}
    source, path, detail = _resolve_onnx_path(env.get, download=lambda uri: uri)
    assert source == "RAAY_ONNX_PATH"
    assert path == "/tmp/override.onnx"
    assert detail == "/tmp/override.onnx"


def test_resolve_from_registry_alias(tmp_path):
    model_dir = tmp_path / "registered"
    model_dir.mkdir()
    (model_dir / "model.onnx").write_bytes(b"fake")
    env = {}
    source, path, _ = _resolve_onnx_path(env.get, download=lambda uri: str(model_dir))
    assert source == "models:/ArabicSentiment/Production"
    assert path == str(model_dir / "model.onnx")


def test_resolve_registry_alias_missing_onnx_raises(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    try:
        _resolve_onnx_path({}.get, download=lambda uri: str(empty))
    except RuntimeError as exc:
        assert "model.onnx" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for missing model.onnx")


def test_resolve_raises_when_download_fails():
    def boom(_uri):
        raise RuntimeError("no such model")

    try:
        _resolve_onnx_path({}.get, download=boom)
    except RuntimeError as exc:
        assert "RAAY_ONNX_PATH" in str(exc)
    else:
        raise AssertionError("expected RuntimeError when the alias is unresolvable")


def test_softmax_rows_sum_to_one_and_argmax_preserved():
    logits = np.array([[1.0, 2.0, 0.5], [0.1, 0.2, 3.0], [-1.0, 4.0, 5.0]])
    probs = softmax(logits)
    np.testing.assert_allclose(probs.sum(axis=-1), np.ones(3), atol=1e-6)
    np.testing.assert_array_equal(np.argmax(probs, axis=-1), np.argmax(logits, axis=-1))


def test_to_predictions_maps_labels_in_id_order():
    probs = np.array([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]])
    id2label = {0: "positive", 1: "negative", 2: "neutral"}
    predicted = to_predictions(probs, id2label)
    assert [p["label"] for p in predicted] == ["negative", "positive"]
    assert abs(predicted[0]["score"] - 0.7) < 1e-6
    assert abs(predicted[1]["score"] - 0.8) < 1e-6


class _FakeTokenizer:
    """Minimal callable tokenizer standing in for the HF fast tokenizer."""

    vocab_size = 512
    max_len = 32

    def __call__(
        self, texts, truncation=True, padding=True, max_length=128, return_tensors="pt"
    ):
        n = len(texts)
        length = min(max_length, self.max_len)
        return {
            "input_ids": torch.randint(
                0, self.vocab_size, (n, length), dtype=torch.long
            ),
            "attention_mask": torch.ones((n, length), dtype=torch.long),
        }


def _tiny_session(tmp_path, with_labels=False):
    torch.manual_seed(0)
    config = BertConfig(
        vocab_size=512,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
        num_labels=3,
        id2label={str(i): s for i, s in enumerate(["positive", "negative", "neutral"])},
    )
    model = BertForSequenceClassification(config)
    model.eval()
    input_ids = torch.randint(0, 512, (2, 16), dtype=torch.long)
    enc = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
    onnx_path = str(tmp_path / "tiny.onnx")
    export_to_onnx(model, enc, onnx_path, opset=17)
    import onnxruntime as ort

    return ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])


def test_predict_probs_runs_tiny_onnx(tmp_path):
    session = _tiny_session(tmp_path)
    texts = ["هذا ممتاز", "الجودة رديئة"]
    probs = predict_probs(
        session, _FakeTokenizer(), texts, model_name="tiny", max_length=32
    )
    assert probs.shape == (2, 3)
    np.testing.assert_allclose(probs.sum(axis=-1), np.ones(2), atol=1e-6)


def test_predict_probs_batches_cleanly(tmp_path):
    session = _tiny_session(tmp_path)
    texts = [f"مراجعة رقم {i}" for i in range(7)]
    probs = predict_probs(
        session,
        _FakeTokenizer(),
        texts,
        model_name="tiny",
        max_length=32,
        batch_size=4,
    )
    assert probs.shape == (7, 3)


def test_to_predictions_roundtrip_via_probs(tmp_path):
    session = _tiny_session(tmp_path)
    id2label = {0: "positive", 1: "negative", 2: "neutral"}
    probs = predict_probs(
        session, _FakeTokenizer(), ["شيء ممتاز"], model_name="tiny", max_length=32
    )
    predictions = to_predictions(probs, id2label)
    assert len(predictions) == 1
    assert predictions[0]["label"] in id2label.values()
    assert 0.0 <= predictions[0]["score"] <= 1.0
