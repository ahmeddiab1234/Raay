"""Model-graph resolution and the shared predict/softmax pipeline.

Everything in here is deliberately BentoML-free: the middleware, the service
class, the latency benchmark and the nightly batch scorer all need the same
"turn raw Arabic text into probabilities" path, and none of them should have to
go through a REST round-trip to get it.

Env-tunable knobs, read per call rather than cached at import so a restarted
worker picks up a new value:

    RAAY_ONNX_PATH           explicit graph override (default: none). Kept for
                             the benchmark/Locust drivers and quick dev swaps.
    RAAY_REGISTERED_MODEL    registered model name (default ArabicSentiment)
    RAAY_ALIAS               registry alias to resolve (default Production)
                             -> loads models:/<name>/<alias> via mlflow, so a
                             newly promoted model is picked up on restart.
    RAAY_TOKENIZER_DIR       checkpoint dir for tokenizer + id2label (default models/baseline/final)
    RAAY_MAX_LENGTH          tokenizer truncation length (default 128)
    RAAY_MODEL_NAME          ArabertPreprocessor model name (default aubmindlab/bert-base-arabertv02)
    RAAY_MODEL_VERSION       provenance stamp reported as ``model_version`` on
                             /predict and /health (default "unversioned"). CI
                             bakes it in via
                             ``bentoml containerize --build-arg`` and repeats it
                             as an OCI label, so a response can be traced back
                             to the exact graph bytes in the image.
"""

from __future__ import annotations

import os
import warnings
from typing import Any

import numpy as np
from transformers import AutoConfig

from raay.enums.constants import DefaultPaths
from raay.serving.graph_source import (
    DEFAULT_ALIAS,
    DEFAULT_REGISTERED_MODEL,
    download_artifacts,
    resolve_onnx_path,
)

# Graph resolution moved to ``graph_source``; these private aliases keep the
# historical ``raay.serving.runtime._resolve_onnx_path`` import path working
# (``serve.py`` re-exports it, and tests import it from here). Patch
# ``graph_source`` -- not these -- to stub resolution: ``resolve_onnx_path``
# takes ``download=`` as a parameter precisely so callers inject a stub instead.
_resolve_onnx_path = resolve_onnx_path
_download_artifacts = download_artifacts

warnings.filterwarnings("ignore", category=SyntaxWarning)

try:
    from arabert.preprocess import ArabertPreprocessor
except ImportError:  # pragma: no cover - import path guard
    ArabertPreprocessor = None

DEFAULT_TOKENIZER_DIR = DefaultPaths.BASELINE_MODEL.value
DEFAULT_MAX_LENGTH = 128
# Reported when nothing stamped a provenance value. Never guess a version here:
# "unversioned" is a visible gap in traceability, a wrong value is a silent lie.
DEFAULT_MODEL_VERSION = "unversioned"
BATCH_SIZE = 32


def model_version() -> str:
    """The provenance stamp for the graph this worker is serving.

    Read per call rather than cached at import so a restarted worker, the
    in-process tests, and ``RAAY_MODEL_VERSION`` overridden at container start
    all report the same value.
    """
    value = os.environ.get("RAAY_MODEL_VERSION", "").strip()
    return value or DEFAULT_MODEL_VERSION


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
    batch_size: int = BATCH_SIZE,
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


def load_id2label(tokenizer_dir: str) -> dict[int, str]:
    """Read the fine-tuned checkpoint's ``id2label`` map as ``{int: str}``.

    The id order is load-bearing and is *not* alphabetical: the graphs use
    positive=0/negative=1/neutral=2. An ONNX session has no config, so anything
    reading labels has to go through a real checkpoint (or an ``AutoConfig``
    assigned onto the session) rather than guessing.
    """
    config = AutoConfig.from_pretrained(tokenizer_dir)
    raw = getattr(config, "id2label", None) or {}
    return {int(k): v for k, v in raw.items()}


def describe_graph(onnx_path: str) -> str:
    """One-line human description of a loaded ORT session, for startup logs."""
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    return f"{onnx_path} providers={session.get_providers()}"


__all__ = [
    "BATCH_SIZE",
    "DEFAULT_ALIAS",
    "DEFAULT_MAX_LENGTH",
    "DEFAULT_MODEL_VERSION",
    "DEFAULT_REGISTERED_MODEL",
    "DEFAULT_TOKENIZER_DIR",
    "load_id2label",
    "model_version",
    "predict_probs",
    "softmax",
    "to_predictions",
]
