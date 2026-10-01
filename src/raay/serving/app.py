"""The BentoML service object: I/O models, the inner service, the app.

``svc`` is what ``bentoml serve src/raay/serving/serve.py:svc`` resolves to and
what ``bentofile.yaml`` ships as ``raay.serving.serve:svc``; the module is kept
as the stable entry point while the pieces it composes live in
:mod:`raay.serving.runtime`, :mod:`raay.serving.middleware` and
:mod:`raay.serving.telemetry`.
"""

from __future__ import annotations

import os
import threading
from typing import Any

import onnxruntime as ort
from bentoml import Service, api
from loguru import logger
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

from raay.enums.constants import Models
from raay.serving.middleware import (
    HealthRouteMiddleware,
    Validation422Middleware,
)
from raay.serving.runtime import (
    DEFAULT_MAX_LENGTH,
    DEFAULT_TOKENIZER_DIR,
    _resolve_onnx_path,
    load_id2label,
    model_version,
    predict_probs,
    to_predictions,
)
from raay.serving.telemetry import TelemetryMiddleware


class PredictRequest(BaseModel):
    """One ``/predict`` call: a list of raw Arabic review texts."""

    texts: list[str]


class Prediction(BaseModel):
    label: str
    score: float


class PredictResponse(BaseModel):
    predictions: list[Prediction]
    # Echoes RAAY_MODEL_VERSION so a client can tie a response to the exact
    # graph baked into the image, without a second call.
    model_version: str = Field(default_factory=model_version)


class RaaySV:
    """BentoML inner service wrapping the INT8 ONNX graph.

    Lazy-loads per worker under a double-checked lock: BentoML instantiates one
    inner service per worker and a burst of concurrent ``/predict`` calls
    arrives the moment the port opens, so a plain ``if self._loaded`` would let
    every one of them build a ~136 MB ORT session.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loaded = False
        self._session: Any = None
        self._tokenizer: Any = None
        self._id2label: dict[int, str] = {}

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            source_label, onnx_path, _detail = _resolve_onnx_path(os.environ.get)
            tokenizer_dir = os.environ.get("RAAY_TOKENIZER_DIR", DEFAULT_TOKENIZER_DIR)
            self._session = ort.InferenceSession(
                onnx_path, providers=["CPUExecutionProvider"]
            )
            self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
            self._id2label = load_id2label(tokenizer_dir)
            self._loaded = True
            logger.info(
                f"Serving {onnx_path} [{source_label}] via "
                f"{self._session.get_providers()} with labels {self._id2label}"
            )

    @api(input_spec=PredictRequest, output_spec=PredictResponse)
    def predict(self, texts: list[str]) -> PredictResponse:
        """Classify raw Arabic review texts; returns top label + softmax score.

        BentoML's new-style SDK binds each input model field as a keyword
        argument, so the signature mirrors ``PredictRequest.texts``.
        """
        self._ensure_loaded()
        if not texts:
            return PredictResponse(predictions=[])
        max_length = int(os.environ.get("RAAY_MAX_LENGTH", DEFAULT_MAX_LENGTH))
        model_name = os.environ.get("RAAY_MODEL_NAME", Models.TEACHER.value)
        probs = predict_probs(
            self._session, self._tokenizer, texts, model_name, max_length
        )
        predictions = to_predictions(probs, self._id2label)
        return PredictResponse(
            predictions=[Prediction(**prediction) for prediction in predictions]
        )


svc = Service(name="raay-sentiment", inner=RaaySV)
svc.add_asgi_middleware(HealthRouteMiddleware)
svc.add_asgi_middleware(Validation422Middleware)
# Telemetry is registered last so it is the OUTERMOST middleware: it must see
# the final response status/body (including the 422 that Validation422 re-emits).
svc.add_asgi_middleware(TelemetryMiddleware)
