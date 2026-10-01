"""The baseline fine-tuning run (Hydra entry point).

Runs on the repo root, GPU or CPU:

    uv run python -m raay.training.train                        # Hydra defaults
    uv run python -m raay.training.train learning_rate=3e-5     # override

Behaviour
    - Loads ``data/processed/{train,val,test}.csv`` plus any validated
      customer-service overrides (train only).
    - Applies ``ArabertPreprocessor`` to ``text`` before tokenization.
    - Fine-tunes `aubmindlab/bert-base-arabertv02` with the HuggingFace
      Trainer (multiclass sentiment).
    - Logs params / metrics / tokenizer version / model artifacts to MLflow
      (experiment ``raay_training``) and checkpoints every ``save_steps``.

The GPU training happens on Kaggle; this script is GPU-agnostic (uses whatever
accelerator Trainer finds, CPU included for smoke tests).
"""

from __future__ import annotations

from pathlib import Path

import hydra
import mlflow
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    set_seed,
)

from raay.config.env import load_environment, mlflow_tracking_uri
from raay.data.dialect import add_dialect_column
from raay.training.train_args import build_training_args
from raay.training.train_data import load_data, tokenize_and_encode
from raay.training.training_common import (
    compute_metrics,
    inverse_label_map,
    label_map,
    loggable_params,
    pinned_pip_requirements,
    resolve_output_dir,
    set_preprocessor,
)


def _require_expected_device(cfg: DictConfig) -> str:
    """Guard rail: never silently train on the wrong hardware."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if cfg.require_gpu and device != "cuda":
        raise RuntimeError(
            f"require_gpu=true but no CUDA GPU detected (device={device}). "
            "Runs must happen on a Kaggle GPU session; refusing to train on CPU. "
            "Set require_gpu=false ONLY for deliberate local CPU smoke tests."
        )
    return device


def _absolute_output_dir(cfg: DictConfig) -> None:
    # Resolve output_dir to an absolute path exactly once so MLflow sees a
    # single value (our own log_params AND the HF MLflow callback both log
    # output_dir; a relative/absolute mismatch would collide on an immutable
    # param and be rejected as a second write).
    OmegaConf.set_struct(cfg, False)
    cfg.output_dir = resolve_output_dir(cfg)
    OmegaConf.set_struct(cfg, True)


@hydra.main(version_base=None, config_path="../../../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    load_environment()

    # Disable Hydra's own MLflow autologging to avoid double-logging; we log
    # explicitly below. Track to the configured store (Kaggle sessions use a
    # local file store; local dev may use sqlite://); file stores get the
    # MLFLOW_ALLOW_FILE_STORE workaround automatically.
    mlflow_tracking_uri(default="file:./mlruns")
    if "mlflow.autolog" in str(OmegaConf.to_container(cfg, resolve=True)):
        mlflow.autolog(disable=True)

    _require_expected_device(cfg)
    _absolute_output_dir(cfg)
    save_model = bool(cfg.save_model)

    set_seed(cfg.seed)
    logger.info(OmegaConf.to_yaml(cfg))

    id_to_label = inverse_label_map(cfg)
    label_ids = label_map(cfg)
    logger.info(f"Label map: {label_ids}")

    set_preprocessor(cfg.model_name)

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
        cfg.model_name, num_labels=cfg.num_labels, id2label=id_to_label
    )

    train_ds = tokenize_and_encode(train_df, tokenizer, label_ids, cfg)
    val_ds = tokenize_and_encode(val_df, tokenizer, label_ids, cfg)

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
        mlflow.log_params(loggable_params(cfg))

        logger.info(f"MLflow run ID: {run.info.run_id}")

        # Sanity check: max_steps must be -1 (epoch-driven) unless a fixed
        # step budget was explicitly requested.
        print(
            f"training_args.max_steps={args.max_steps} "
            f"num_train_epochs={args.num_train_epochs}"
        )

        train_result = trainer.train()
        merged = {**train_result.metrics, **trainer.evaluate()}
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
            final_dir = Path(cfg.output_dir) / "final"
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
                pip_requirements=pinned_pip_requirements(),
            )
