"""BentoML serving endpoint for the INT8-compressed AraBERT ONNX model.

Phase 3 (compression/optimization), step 5: serve the dynamically quantized
``models/onnx/model_int8.onnx`` graph through BentoML. This is the decision
artifact for the GPU/TensorRT question: run it, measure p50/p95/p99 against
the product SLA, and only pursue a TRT engine if the numbers miss.

Run from the repo root:

    uv run bentoml serve src/raay/serving/serve.py:svc --reload

then POST to http://localhost:3000/predict:

    curl -s http://localhost:3000/predict \
      -H 'Content-Type: application/json' \
      -d '{"texts": ["المنتج ممتاز", "الطلبية وصلت متأخرة"]}'

Request/response use BentoML's new-style pydantic I/O descriptors
(``bentoml.api``; the ``bentoml.io`` module is deprecated since v1.4). The
service lazy-loads per worker: an ORT CPU session plus the HF tokenizer and
label map from the fine-tuned tokenizer dir. Behaviour is env-tunable:

    RAAY_ONNX_PATH      ONNX graph to serve (default models/onnx/model_int8.onnx)
    RAAY_TOKENIZER_DIR  checkpoint dir for tokenizer + id2label (default models/baseline/final)
    RAAY_MAX_LENGTH     tokenizer truncation length (default 128)
    RAAY_MODEL_NAME     ArabertPreprocessor model name (default aubmindlab/bert-base-arabertv02)

The predict/softmax machinery is module-level so the benchmark
(``raay.serving.benchmark``) and unit tests reuse it without a BentoML
round-trip.
"""

from __future__ import annotations

import os
import threading
import warnings
from typing import Any

import numpy as np
import onnxruntime as ort
from bentoml import Service, api
from loguru import logger
from pydantic import BaseModel
from transformers import AutoConfig, AutoTokenizer

from raay.enums.constants import DefaultPaths, Models

warnings.filterwarnings("ignore", category=SyntaxWarning)

try:
    from arabert.preprocess import ArabertPreprocessor
except ImportError:  # pragma: no cover - import path guard
    ArabertPreprocessor = None

_DEFAULT_ONNX_PATH = DefaultPaths.ONNX_INT8_MODEL.value
_DEFAULT_TOKENIZER_DIR = DefaultPaths.BASELINE_MODEL.value
_DEFAULT_MAX_LENGTH = 128
_BATCH_SIZE = 32

# arabert instantiation is expensive (pulls in pyarabic); cache one processor
# per model name so every ``/predict`` call only pays for ``preprocess``.
_PREPROCESSORS: dict[str, Any] = {}


def _preprocess(text: str, model_name: str) -> str:
    if ArabertPreprocessor is None:
        return str(text)
    proc = _PREPROCESSORS.get(model_name)
    if proc is None:
        proc = _PREPROCESSORS.setdefault(
            model_name, ArabertPreprocessor(model_name=model_name)
        )
    return proc.preprocess(text)


def softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax over the last axis (numerically stable)."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def predict_probs(
    session: Any,
    tokenizer: Any,
    texts: list[str],
    model_name: str,
    max_length: int,
    batch_size: int = _BATCH_SIZE,
) -> np.ndarray:
    """Preprocess -> tokenize -> run the ONNX graph; return row-softmax probs.

    ``session`` is an ORT ``InferenceSession``; ``tokenizer`` is any callable
    returning ``{input_ids, attention_mask}`` tensors (in production the HF
    fast tokenizer). ``texts`` are raw Arabic strings, not yet preprocessed.
    """
    chunks: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = [_preprocess(t, model_name) for t in texts[i : i + batch_size]]
        enc = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        logits = session.run(
            ["logits"],
            {
                "input_ids": enc["input_ids"].numpy(),
                "attention_mask": enc["attention_mask"].numpy(),
            },
        )[0]
        chunks.append(logits)
    return softmax(np.concatenate(chunks, axis=0))


def to_predictions(probs: np.ndarray, id2label: dict[int, str]) -> list[dict[str, Any]]:
    """Map softmax rows to ``{label, score}`` dicts in label-id order."""
    labels = [id2label[i] for i in sorted(id2label)]
    predictions: list[dict[str, Any]] = []
    for row in probs:
        idx = int(np.argmax(row))
        predictions.append({"label": labels[idx], "score": float(row[idx])})
    return predictions


class PredictRequest(BaseModel):
    """One ``/predict`` call: a list of raw Arabic review texts."""

    texts: list[str]


class Prediction(BaseModel):
    label: str
    score: float


class PredictResponse(BaseModel):
    predictions: list[Prediction]


class RaaySV:
    """BentoML inner service wrapping the INT8 ONNX graph."""

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
            onnx_path = os.environ.get("RAAY_ONNX_PATH", _DEFAULT_ONNX_PATH)
            tokenizer_dir = os.environ.get("RAAY_TOKENIZER_DIR", _DEFAULT_TOKENIZER_DIR)
            self._session = ort.InferenceSession(
                onnx_path, providers=["CPUExecutionProvider"]
            )
            self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
            config = AutoConfig.from_pretrained(tokenizer_dir)
            raw = getattr(config, "id2label", None) or {}
            self._id2label = {int(k): v for k, v in raw.items()}
            self._loaded = True
            logger.info(
                f"Serving {onnx_path} via "
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
        max_length = int(os.environ.get("RAAY_MAX_LENGTH", _DEFAULT_MAX_LENGTH))
        model_name = os.environ.get("RAAY_MODEL_NAME", Models.TEACHER.value)
        probs = predict_probs(
            self._session, self._tokenizer, texts, model_name, max_length
        )
        predictions = to_predictions(probs, self._id2label)
        return PredictResponse(
            predictions=[Prediction(**prediction) for prediction in predictions]
        )


svc = Service(name="raay-sentiment", inner=RaaySV)
