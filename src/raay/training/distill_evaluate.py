"""Test-split scoring for the distilled student."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, f1_score

from raay.training.distill_teacher import _preprocess_batch
from raay.training.training_common import _preprocess_fn, inverse_label_map, label_map


def _evaluate_test(
    student: Any,
    tokenizer: Any,
    test_df: pd.DataFrame,
    cfg: DictConfig,
    save_hard_predictions: bool = True,
) -> dict[str, float]:
    """Return test accuracy / f1 metrics for the student using the eval split.

    Uses the same preprocessing + batching path as the teacher forward pass so
    the student's test numbers are directly comparable to the baseline harness.
    """
    id_to_label = inverse_label_map(cfg)
    label_to_id = label_map(cfg)
    texts = test_df["text"].map(_preprocess_fn).tolist()
    labels = test_df["label"].map(label_to_id).tolist()

    student.eval()
    device = next(student.parameters()).device
    preds_list: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(texts), cfg.batch_size):
            batch = _preprocess_batch(texts[i : i + cfg.batch_size])
            enc = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=cfg.max_length,
                return_tensors="pt",
            ).to(device)
            preds = torch.argmax(student(**enc).logits, dim=-1).detach().cpu().numpy()
            preds_list.append(preds)
    preds = np.concatenate(preds_list, axis=0)

    metrics: dict[str, float] = {
        "test_accuracy": float(accuracy_score(labels, preds)),
        "test_f1_macro": float(f1_score(labels, preds, average="macro")),
    }

    if save_hard_predictions:
        test_df = test_df.copy()
        test_df["predicted_label"] = [id_to_label[p] for p in preds]
        pred_path = Path(cfg.output_dir) / "test_predictions.csv"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        test_df.to_csv(pred_path, index=False)

    return metrics
