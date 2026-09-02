"""Fine-tune the AraBERT baseline sentiment classifier.

Run from the repo root (DVC / Kaggle launch it there and every path below is
relative to the repo root):

    uv run python -m raay.training.train                        # Hydra defaults
    uv run python -m raay.training.train learning_rate=3e-5     # override

Behaviour
    - Loads ``data/processed/{train,val,test}.csv``.
    - Applies ``ArabertPreprocessor`` to ``text`` before tokenization.
    - Fine-tunes `aubmindlab/bert-base-arabertv02` with the HuggingFace
      Trainer (multiclass sentiment).
    - Logs params / metrics / tokenizer version / model artifacts to MLflow
      (experiment ``raay_training``) and checkpoints every ``save_steps``.

The GPU training happens on Kaggle; this script is GPU-agnostic (uses whatever
accelerator Trainer finds, CPU included for smoke tests).
"""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    set_seed,
)

import mlflow
from raay.data.dialect import add_dialect_column

# pyarabic (a transitive dep of `arabert`) emits noisy SyntaxWarnings on import.
warnings.filterwarnings("ignore", category=SyntaxWarning)
logging.getLogger("transformers").setLevel(logging.WARNING)

try:
    from arabert.preprocess import ArabertPreprocessor
except ImportError:  # pragma: no cover - import path guard
    ArabertPreprocessor = None  # type: ignore[assignment]


def _pinned_pip_requirements() -> list[str]:
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


def _label_map(cfg: DictConfig) -> dict[str, int]:
    return {label: i for i, label in enumerate(cfg.labels)}


def _inverse_label_map(cfg: DictConfig) -> dict[int, str]:
    return {i: label for i, label in enumerate(cfg.labels)}


def load_data(cfg: DictConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(cfg.train_file)
    val = pd.read_csv(cfg.val_file)
    test = pd.read_csv(cfg.test_file)
    return train, val, test


def _preprocess_fn(text: str) -> str:
    if ArabertPreprocessor is not None:
        return _PREPROCESSOR.preprocess(text)  # type: ignore[attr-defined]
    return str(text)


# Module-level slot so the preprocessor is built once after config is known.
_PREPROCESSOR: Any = None


def tokenize_and_encode(
    df: pd.DataFrame, tokenizer: Any, label_map: dict[str, int], cfg: DictConfig
) -> Dataset:
    texts = df["text"].map(_preprocess_fn).tolist()
    labels = df["label"].map(label_map).tolist()

    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=cfg.max_length,
    )
    dataset = Dataset.from_dict(
        {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": labels,
        }
    )
    return dataset


def compute_metrics(eval_pred: EvalPrediction, cfg: DictConfig) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    id_to_label = _inverse_label_map(cfg)
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


def _resolve_output_dir(cfg: DictConfig) -> str:
    relative_out = Path(cfg.output_dir)
    if not relative_out.is_absolute():
        relative_out = Path(os.getcwd()) / relative_out
    return str(relative_out)


def build_training_args(cfg: DictConfig) -> TrainingArguments:
    # transformers uses a *negative* max_steps as the sentinel for
    # "derive total steps from num_train_epochs". A 0 is treated as a real
    # step budget of zero (not epoch-based), which silently truncates training
    # to ~1 step. Map our "use epochs" default (max_steps == 0) to -1.
    max_steps = cfg.max_steps if int(cfg.max_steps) > 0 else -1
    save_model = bool(cfg.save_model)
    return TrainingArguments(
        output_dir=_resolve_output_dir(cfg),
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        num_train_epochs=cfg.epochs,
        max_steps=max_steps,
        weight_decay=cfg.weight_decay,
        warmup_steps=cfg.warmup_steps,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps" if save_model else "no",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        logging_strategy="steps",
        logging_steps=cfg.logging_steps,
        load_best_model_at_end=bool(cfg.load_best_model_at_end) and save_model,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=True,
        seed=cfg.seed,
        report_to=["mlflow"],
    )


@hydra.main(version_base=None, config_path="../../../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    global _PREPROCESSOR

    # Disable Hydra's own MLflow autologging to avoid double-logging; we log
    # explicitly below. Use a file store unless overridden (Kaggle sessions
    # track to a local file store; local dev may use sqlite://).
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
    if tracking_uri.startswith("file:"):
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    if "mlflow.autolog" in str(OmegaConf.to_container(cfg, resolve=True)):
        mlflow.autolog(disable=True)

    # ---- Guard rail: never silently train on the wrong hardware ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if cfg.require_gpu and device != "cuda":
        raise RuntimeError(
            f"require_gpu=true but no CUDA GPU detected (device={device}). "
            "Runs must happen on a Kaggle GPU session; refusing to train on CPU. "
            "Set require_gpu=false ONLY for deliberate local CPU smoke tests."
        )

    # Resolve output_dir to an absolute path exactly once so MLflow sees a
    # single value (our own log_params AND the HF MLflow callback both log
    # output_dir; a relative/absolute mismatch would collide on an immutable
    # param and be rejected as a second write).
    OmegaConf.set_struct(cfg, False)
    cfg.output_dir = _resolve_output_dir(cfg)
    OmegaConf.set_struct(cfg, True)

    save_model = bool(cfg.save_model)

    set_seed(cfg.seed)
    logger.info(OmegaConf.to_yaml(cfg))

    label_map = _label_map(cfg)
    logger.info(f"Label map: {label_map}")

    if ArabertPreprocessor is not None:
        _PREPROCESSOR = ArabertPreprocessor(model_name=cfg.model_name)

    train_df, val_df, test_df = load_data(cfg)
    logger.info(
        f"Loaded splits: train={len(train_df)} val={len(val_df)} test={len(test_df)}"
    )

    # Dialect-tag the test set so the evaluation harness can break down by
    # dialect without re-reading/relabelling later.
    test_df = add_dialect_column(test_df, text_col="text")
    # Save to a writable location: on Kaggle the input dir is read-only, so
    # always write next to the model output_dir (inside the working copy).
    dialect_test_path = Path(cfg.output_dir) / "test_dialect.csv"
    dialect_test_path.parent.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(dialect_test_path, index=False)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    logger.info(f"Tokenizer version: {getattr(tokenizer, 'vocab_size', 'N/A')}")

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=cfg.num_labels, id2label=_inverse_label_map(cfg)
    )

    train_ds = tokenize_and_encode(train_df, tokenizer, label_map, cfg)
    val_ds = tokenize_and_encode(val_df, tokenizer, label_map, cfg)

    args = build_training_args(cfg)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=lambda p: compute_metrics(p, cfg),
    )

    mlflow.set_experiment(cfg.experiment_name)
    with mlflow.start_run(run_name=f"baseline-{cfg.model_name.split('/')[-1]}") as run:
        mlflow.log_params(_loggable_params(cfg))

        run_id = run.info.run_id
        logger.info(f"MLflow run ID: {run_id}")

        # Sanity check: max_steps must be -1 (epoch-driven) unless a fixed
        # step budget was explicitly requested.
        print(
            f"training_args.max_steps={args.max_steps} "
            f"num_train_epochs={args.num_train_epochs}"
        )

        train_result = trainer.train()
        out_dir = cfg.output_dir

        metrics = train_result.metrics
        eval_metrics = trainer.evaluate()
        merged = {**metrics, **eval_metrics}
        logger.info(f"Final metrics: {merged}")

        for key, value in merged.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, float(value))

        # Tokenizer version / vocab size as an artifact tag.
        mlflow.log_param(
            "tokenizer_vocab_size", getattr(tokenizer, "vocab_size", "N/A")
        )
        mlflow.log_param("model_name", cfg.model_name)
        mlflow.log_param("run_tag", str(cfg.run_tag))

        # Persist weights only for the final (winner) run: during the sweep
        # save_model=false so candidates log metrics/params exclusively and skip
        # the (large) checkpoints + mlruns artifacts. Log just the final/
        # subfolder, not Trainer's rotated checkpoint-* dirs.
        if save_model:
            final_dir = Path(out_dir) / "final"
            trainer.save_model(str(final_dir))
            # Log a proper MLflow Model (MLmodel + weights) so it can be
            # registered from runs:/<id>/model; log_artifacts wouldn't create
            # an MLmodel file and register_model would fail.
            mlflow.transformers.log_model(
                transformers_model={
                    "model": trainer.model,
                    "tokenizer": tokenizer,
                },
                task="text-classification",
                artifact_path="model",
                pip_requirements=_pinned_pip_requirements(),
            )


def _loggable_params(cfg: DictConfig) -> dict[str, Any]:
    # Flatten the resolved config to scalar params for MLflow, skipping the
    # hydra meta keys and nested groups.
    params: dict[str, Any] = {}
    container: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]
    for key, value in container.items():
        key_str = str(key)
        if key_str.startswith("hydra"):
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            params[key_str] = value
    return params


if __name__ == "__main__":
    main()
