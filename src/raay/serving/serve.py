"""BentoML serving entrypoint for the compressed AraBERT ONNX model.

Phase 3 (compression/optimization), step 5 + step 8: serve the dynamically
quantized ``models/onnx/model_int8.onnx`` graph through BentoML. This is the
decision artifact for the GPU/TensorRT question: run it, measure p50/p95/p99
against the product SLA, and only pursue a TRT engine if the numbers miss.

Run from the repo root:

    uv run bentoml serve src/raay/serving/serve.py:svc --reload

then POST to http://localhost:3000/predict:

    curl -s http://localhost:3000/predict \
      -H 'Content-Type: application/json' \
      -d '{"texts": ["المنتج ممتاز", "الطلبية وصلت متأخرة"]}'

This module is the stable public surface (``raay.serving.serve:svc`` is what
``bentofile.yaml`` and the compose/CI files reference) and it re-exports the
pieces the benchmark, the nightly scorer and the unit tests use, so those do
not have to know how the package is decomposed:

* :mod:`raay.serving.runtime` -- env-tunable graph resolution
  (``RAAY_ONNX_PATH`` / registry alias), ``predict_probs``, ``softmax``,
  ``to_predictions`` and the ``RAAY_MODEL_VERSION`` provenance stamp.
* :mod:`raay.serving.middleware` -- ``GET /health`` and the 400 -> 422 rewrite.
* :mod:`raay.serving.telemetry` -- the per-request canary/shadow event POST.
* :mod:`raay.serving.app` -- the pydantic I/O models and ``svc`` itself.

Request/response use BentoML's new-style pydantic I/O descriptors
(``bentoml.api``; the ``bentoml.io`` module is deprecated since v1.4). The
service lazy-loads per worker: an ORT CPU session plus the HF tokenizer and
label map from the fine-tuned tokenizer dir.
"""

from __future__ import annotations

from raay.serving.app import (
    Prediction,
    PredictRequest,
    PredictResponse,
    RaaySV,
    svc,
)
from raay.serving.middleware import (
    HealthRouteMiddleware,
    Validation422Middleware,
)
from raay.serving.runtime import (
    _download_artifacts,
    _preprocess,
    _resolve_onnx_path,
    load_id2label,
    model_version,
    predict_probs,
    softmax,
    to_predictions,
)
from raay.serving.telemetry import (
    TelemetryMiddleware,
    _dispatch_event,
    _post_event,
)

__all__ = [
    "HealthRouteMiddleware",
    "PredictRequest",
    "PredictResponse",
    "Prediction",
    "RaaySV",
    "TelemetryMiddleware",
    "Validation422Middleware",
    "_dispatch_event",
    "_download_artifacts",
    "_post_event",
    "_preprocess",
    "_resolve_onnx_path",
    "load_id2label",
    "model_version",
    "predict_probs",
    "softmax",
    "svc",
    "to_predictions",
]
