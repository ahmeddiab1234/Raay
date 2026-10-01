"""The two route-shape middlewares, driven directly as ASGI callables."""

import json

from serving_harness import response_body, run_health_middleware, run_middleware

from raay.serving.middleware import (
    HealthRouteMiddleware,
    Validation422Middleware,
)


def test_health_middleware_returns_status_json(monkeypatch):
    monkeypatch.delenv("RAAY_MODEL_VERSION", raising=False)
    messages = run_health_middleware(HealthRouteMiddleware)
    assert messages[0]["type"] == "http.response.start"
    assert messages[0]["status"] == 200
    assert json.loads(response_body(messages)) == {
        "status": "healthy",
        "model_version": "unversioned",
    }


def test_validation_422_middleware_rewrites_pydantic_400():
    body = (
        b'{"error": "1 validation error for PredictRequest", '
        b'"detail": [{"loc": ["texts"], "msg": "Field required", "type": "missing"}]}'
    )
    messages = run_middleware(Validation422Middleware, body=body)
    assert messages[0]["status"] == 422
    assert response_body(messages) == body


def test_validation_422_middleware_leaves_non_validation_400():
    """A 400 that is a real application error keeps its status: rewriting it
    would report a server-side failure as a client schema problem."""
    messages = run_middleware(
        Validation422Middleware, body=b'{"error": "task_id is required"}'
    )
    assert messages[0]["status"] == 400


def test_validation_422_middleware_leaves_non_http_scope_alone():
    seen = []

    async def app(scope, _receive, _send):
        seen.append(scope["type"])

    run_middleware(
        Validation422Middleware,
        scp={"type": "lifespan"},
        app=app,
    )
    assert seen == ["lifespan"]
