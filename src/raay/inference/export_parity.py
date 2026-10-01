"""Model loading and parity utilities for ONNX export."""

from __future__ import annotations

from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_model(model_dir: str):
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    id2label = getattr(model.config, "id2label", None)
    return model, tokenizer, id2label


def export_to_onnx(
    model: torch.nn.Module,
    enc: dict[str, torch.Tensor],
    onnx_path: str,
    opset: int,
) -> None:
    """Freeze ``model`` to ``onnx_path`` with dynamic batch/sequence axes."""
    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model,
            (enc["input_ids"], enc["attention_mask"]),
            onnx_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "logits": {0: "batch"},
            },
            opset_version=opset,
            do_constant_folding=True,
        )
    logger.info(f"Exported {onnx_path} (opset={opset})")


def run_ort(session: ort.InferenceSession, enc: dict[str, torch.Tensor]) -> np.ndarray:
    """Run the exported graph and return the raw logits as a numpy array."""
    feed = {
        "input_ids": enc["input_ids"].numpy(),
        "attention_mask": enc["attention_mask"].numpy(),
    }
    return session.run(["logits"], feed)[0]


def torch_logits(model: torch.nn.Module, enc: dict[str, torch.Tensor]) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(**enc).logits.numpy()


def verify_parity(
    pt_logits: np.ndarray, ort_logits: np.ndarray, tolerance: float
) -> dict[str, Any]:
    """Compare PyTorch vs ONNX Runtime logits; return report + pass flag."""
    diff = np.abs(pt_logits - ort_logits)
    argmax_match = bool(
        np.array_equal(np.argmax(pt_logits, axis=-1), np.argmax(ort_logits, axis=-1))
    )
    result = {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "argmax_match": argmax_match,
        "tolerance": tolerance,
        "passed": bool(diff.max() <= tolerance) and argmax_match,
    }
    return result
