"""The BentoML app's own routes: /health, /predict validation, model_version.

These run against the real ``svc`` in-process via ``TestClient(svc.to_asgi())``.
No model graph is needed: ``/health`` short-circuits in
``HealthRouteMiddleware`` and malformed payloads are rejected by the input model
before ``RaaySV._ensure_loaded()`` runs.
"""

import pytest
from starlette.testclient import TestClient

from raay.serving.serve import (
    HealthRouteMiddleware,
    PredictResponse,
    TelemetryMiddleware,
    Validation422Middleware,
    model_version,
    svc,
)


@pytest.fixture(scope="module")
def client():
    """In-process ASGI client for the real BentoML app.

    ``TestClient`` must be used as a context manager: BentoML instantiates the
    inner service in the Starlette lifespan (``create_instance``), so without it
    every route resolves against a ``None`` instance and 500s.
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


def test_telemetry_is_the_outermost_middleware():
    """It must see the final status/body, including Validation422's rewrite."""
    assert [cls for cls, _ in svc.middlewares][-1] is TelemetryMiddleware


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
