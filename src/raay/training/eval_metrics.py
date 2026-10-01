"""Scoring backends and split metrics for the evaluation harness.

``predict`` is backend-agnostic on purpose: ``model`` is either a torch Module
or an ONNX Runtime session, and the promotion gate scores the same split through
both to compare them.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from raay.data.dialect import add_dialect_column

warnings.filterwarnings("ignore", category=SyntaxWarning)

try:
    from arabert.preprocess import ArabertPreprocessor
except ImportError:  # pragma: no cover - import path guard
    ArabertPreprocessor = None


def _preprocess(text: str, model_name: str) -> str:
    if ArabertPreprocessor is not None:
        return ArabertPreprocessor(model_name=model_name).preprocess(text)
    return str(text)


def load_model(model_dir: str):
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    id2label = getattr(getattr(model, "config", None), "id2label", None)
    return model, tokenizer, id2label


def load_onnx_session(
    onnx_path: str, *, session_options: ort.SessionOptions | None = None
) -> ort.InferenceSession:
    """Load an exported ONNX model for evaluation (frontends an ORT session).

    ``session_options`` is optional and defaults to None, i.e. exactly the
    previous behaviour for every existing caller. The promotion gate passes
    tuned options because it has to compare the latency of two sessions in one
    process, and two default thread pools with spin-waiting interfere with each
    other badly enough to invent a 25% difference between identical graphs.
    Callers that are not measuring latency should leave it alone.
    """
    if session_options is None:
        return ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    return ort.InferenceSession(
        onnx_path, sess_options=session_options, providers=["CPUExecutionProvider"]
    )


def predict(
    model: Any, tokenizer: Any, texts: list[str], model_name: str, max_length: int
) -> np.ndarray:
    """Predict labels; ``model`` is either a torch Module or an ORT session."""
    ort_session = isinstance(model, ort.InferenceSession)
    probs_list: list[np.ndarray] = []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = [_preprocess(t, model_name) for t in texts[i : i + batch_size]]
        batch_enc = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        if ort_session:
            logits = model.run(
                ["logits"],
                {
                    key: batch_enc[key].numpy()
                    for key in ("input_ids", "attention_mask")
                },
            )[0]
        else:
            with torch.no_grad():
                logits = model(**batch_enc).logits.detach().numpy()
        probs_list.append(logits)
    return np.argmax(np.concatenate(probs_list, axis=0), axis=-1)


def _per_class(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list[int], names: list[str]
) -> dict[str, dict]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    precision, recall, f1 = (
        np.atleast_1d(precision),
        np.atleast_1d(recall),
        np.atleast_1d(f1),
    )
    support = (
        np.atleast_1d(support)
        if support is not None
        else np.zeros_like(precision, dtype=int)
    )
    result = {}
    for i, name in enumerate(names):
        result[name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
    return result


def evaluate_on_split(
    df: pd.DataFrame,
    model: Any,
    tokenizer: Any,
    model_name: str,
    max_length: int,
) -> dict[str, Any]:
    id2label = getattr(getattr(model, "config", None), "id2label", None)
    raw_labels = df["label"].tolist()
    # Map string labels to indices via id2label if present, else textual order.
    if id2label and raw_labels and isinstance(raw_labels[0], str):
        id2label = {int(k): v for k, v in id2label.items()}
        label_to_id = {v: k for k, v in id2label.items()}
        y_true = np.array([label_to_id[lab] for lab in raw_labels])
        label_names = [id2label[i] for i in sorted(id2label)]
    else:
        labels_sorted = sorted(set(raw_labels))
        label_to_id = {lab: i for i, lab in enumerate(labels_sorted)}
        y_true = np.array([label_to_id[lab] for lab in raw_labels])
        label_names = labels_sorted
    label_ids = list(range(len(label_names)))

    y_pred = predict(model, tokenizer, df["text"].tolist(), model_name, max_length)

    cm = confusion_matrix(y_true, y_pred, labels=label_ids)
    per_class = _per_class(y_true, y_pred, label_ids, label_names)
    accuracy = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    return {
        "label_names": label_names,
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "sample_size": len(df),
    }


def dialect_breakdown(
    df: pd.DataFrame,
    model: Any,
    tokenizer: Any,
    model_name: str,
    max_length: int,
) -> dict[str, dict[str, float]]:
    if "dialect" not in df.columns:
        df = add_dialect_column(df, text_col="text")

    breakdown: dict[str, dict[str, float]] = {}
    for dialect, group in df.groupby("dialect"):
        group = group.copy()
        if group["label"].nunique() < 2 or len(group) < 2:
            # Degenerate slice: report count only.
            breakdown[str(dialect)] = {"count": len(group)}
            continue
        rep = evaluate_on_split(group, model, tokenizer, model_name, max_length)
        breakdown[str(dialect)] = {
            "count": len(group),
            "accuracy": rep["accuracy"],
            "f1_macro": rep["f1_macro"],
        }
    return breakdown
