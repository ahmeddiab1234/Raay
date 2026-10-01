"""Lazy INT8 ONNX scorer for the micro-batch consumer."""

from __future__ import annotations

import threading
from typing import Any

import onnxruntime as ort
from loguru import logger
from transformers import AutoConfig, AutoTokenizer

from raay.enums.constants import DefaultPaths, Models
from raay.serving.serve import predict_probs, to_predictions


class InferenceScorer:
    """Lazy INT8 ONNX scorer sharing ``serve.predict_probs``.

    ``score`` returns ``{"label", "score"}`` per input in order, exactly like
    a single ``/predict`` call would -- but a micro-batch of <=32 texts is one
    tokenize + one ``session.run`` instead of 32 of each.
    """

    def __init__(
        self,
        onnx_path: str | None = None,
        tokenizer_dir: str | None = None,
        model_name: str | None = None,
        max_length: int = 128,
    ) -> None:
        self._onnx_path = str(onnx_path or DefaultPaths.ONNX_INT8_MODEL.value)
        self._tokenizer_dir = str(tokenizer_dir or DefaultPaths.BASELINE_MODEL.value)
        self._model_name = model_name or Models.TEACHER.value
        self._max_length = max_length
        self._lock = threading.Lock()
        self._session: Any = None
        self._tokenizer: Any = None
        self._id2label: dict[int, str] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            self._session = ort.InferenceSession(
                self._onnx_path, providers=["CPUExecutionProvider"]
            )
            self._tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_dir)
            config = AutoConfig.from_pretrained(self._tokenizer_dir)
            raw = getattr(config, "id2label", None) or {}
            self._id2label = {int(k): v for k, v in raw.items()}
            self._loaded = True
            logger.info(
                f"Batch scorer loaded {self._onnx_path} with labels "
                f"{self._id2label} [{self._session.get_providers()}]"
            )

    def score(self, texts: list[str]) -> list[dict[str, Any]]:
        if not texts:
            return []
        self._ensure_loaded()
        probs = predict_probs(
            self._session,
            self._tokenizer,
            texts,
            self._model_name,
            self._max_length,
        )
        return to_predictions(probs, self._id2label)
