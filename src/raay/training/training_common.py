"""Shared pieces of the baseline and distillation training jobs.

Both entry points (``raay.training.train`` and ``raay.training.distill``) build
the same AraBERT pipeline: identical MLflow pip-pinning, identical label maps
read from ``cfg.labels``, the same ``ArabertPreprocessor`` normalization, the
same ``TrainingArguments`` sentinel handling, the same eval metrics, and the
same "flatten the resolved config into scalar MLflow params" helper. Keeping one
copy is what stops the two run families from drifting apart.
"""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, f1_score
from transformers import EvalPrediction

# pyarabic (a transitive dep of `arabert`) emits noisy SyntaxWarnings on import.
warnings.filterwarnings("ignore", category=SyntaxWarning)
logging.getLogger("transformers").setLevel(logging.WARNING)

try:
    from arabert.preprocess import ArabertPreprocessor
except ImportError:  # pragma: no cover - import path guard
    ArabertPreprocessor = None  # type: ignore[assignment]

# Module-level slot so the preprocessor is built once after config is known.
_PREPROCESSOR: Any = None


def set_preprocessor(model_name: str) -> None:
    """Build the shared ArabertPreprocessor once the config's model name is known."""
    global _PREPROCESSOR
    if ArabertPreprocessor is not None:
        _PREPROCESSOR = ArabertPreprocessor(model_name=model_name)


def _preprocess_fn(text: str) -> str:
    if ArabertPreprocessor is not None:
        return _PREPROCESSOR.preprocess(text)  # type: ignore[attr-defined]
    return str(text)


def pinned_pip_requirements() -> list[str]:
    """Pin the framework deps MLflow would infer, skipping any that are missing.

    ``mlflow.transformers.get_default_pip_requirements`` hardcodes ``torchvision``
    whenever torch is present and then imports it to read its version, which crashes
    on Kaggle (no torchvision installed). ``importlib.metadata`` reads versions
    without importing, so missing packages are simply skipped.
    """
    from importlib.metadata import PackageNotFoundError, version

    pinned: list[str] = []
    for package in ("mlflow", "transformers", "torch", "torchvision", "accelerate"):
        try:
            pinned.append(f"{package}=={version(package)}")
        except PackageNotFoundError:
            continue
    return pinned


def label_map(cfg: DictConfig) -> dict[str, int]:
    return {label: i for i, label in enumerate(cfg.labels)}


def inverse_label_map(cfg: DictConfig) -> dict[int, str]:
    return {i: label for i, label in enumerate(cfg.labels)}


def resolve_output_dir(cfg: DictConfig) -> str:
    relative_out = Path(cfg.output_dir)
    if not relative_out.is_absolute():
        relative_out = Path(os.getcwd()) / relative_out
    return str(relative_out)


def compute_metrics(eval_pred: EvalPrediction, cfg: DictConfig) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    id_to_label = inverse_label_map(cfg)
    target_names = [id_to_label[i] for i in range(cfg.num_labels)]

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro")),
        "f1_weighted": float(f1_score(labels, preds, average="weighted")),
    }

    per_class: np.ndarray = np.asarray(
        f1_score(
            labels,
            preds,
            average=None,
            labels=list(range(cfg.num_labels)),
        )
    )

    for name, value in zip(target_names, per_class):
        metrics[f"f1_{name}"] = float(value)

    return metrics


def loggable_params(cfg: DictConfig, skip: tuple[str, ...] = ()) -> dict[str, Any]:
    # Flatten the resolved config to scalar params for MLflow, skipping the
    # hydra meta keys, nested groups, and any caller-supplied key.
    params: dict[str, Any] = {}
    container: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]
    for key, value in container.items():
        key_str = str(key)
        if key_str.startswith("hydra") or key_str in skip:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            params[key_str] = value
    return params
