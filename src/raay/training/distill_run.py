"""The distillation run (Hydra entry point).

Run from the repo root (DVC / Kaggle launch it there):

    uv run python -m raay.training.distill                        # Hydra defaults
    uv run python -m raay.training.distill alpha=0.4 temperature=4.0

The teacher is the frozen baseline checkpoint; the student is the same
tokenizer/vocab with fewer encoder layers, trained on ``alpha * CE +
(1 - alpha) * T^2 * KL`` against precomputed teacher logits.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import mlflow
import pandas as pd
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from transformers import AutoConfig, AutoTokenizer, set_seed

from raay.config.env import load_environment, mlflow_tracking_uri
from raay.data.dialect import add_dialect_column
from raay.training.distill_data import (
    _BestStepLogger,
    build_train_dataset,
    build_training_args,
)
from raay.training.distill_evaluate import _evaluate_test
from raay.training.distill_models import build_student, load_teacher
from raay.training.distill_objective import DistillationTrainer, DistillCollator
from raay.training.distill_teacher import _cached_teacher_logits
from raay.training.training_common import (
    compute_metrics,
    inverse_label_map,
    label_map,
    loggable_params,
    pinned_pip_requirements,
    resolve_output_dir,
    set_preprocessor,
)

# The distiller logs ``model_name`` separately (as ``distilled``) so the
# baseline/distill sweep filter can tell the two run families apart; the config
# value is the *teacher's* repo id and must not collide.
_LOGGABLE_SKIP = ("model_name",)


def _require_expected_device(cfg: DictConfig) -> str:
    """Guard rail: never silently train on the wrong hardware."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if cfg.require_gpu and device != "cuda":
        raise RuntimeError(
            f"require_gpu=true but no CUDA GPU detected (device={device}). "
            "Distillation must run on a Kaggle GPU session; refusing to train on "
            "CPU. Set require_gpu=false ONLY for deliberate local CPU smoke tests."
        )
    return device


def _absolute_output_dir(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    cfg.output_dir = resolve_output_dir(cfg)
    OmegaConf.set_struct(cfg, True)


def _load_data(cfg: DictConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(cfg.train_file)
    val = pd.read_csv(cfg.val_file)
    test = pd.read_csv(cfg.test_file)
    logger.info(f"Loaded splits: train={len(train)} val={len(val)} test={len(test)}")
    test = add_dialect_column(test, text_col="text")
    dialect_test_path = Path(cfg.output_dir) / "test_dialect.csv"
    dialect_test_path.parent.mkdir(parents=True, exist_ok=True)
    test.to_csv(dialect_test_path, index=False)
    return train, val, test


@hydra.main(version_base=None, config_path="../../../configs", config_name="distill")
def main(cfg: DictConfig) -> None:
    load_environment()

    mlflow_tracking_uri(default="file:./mlruns")
    if "mlflow.autolog" in str(OmegaConf.to_container(cfg, resolve=True)):
        mlflow.autolog(disable=True)

    device = _require_expected_device(cfg)
    _absolute_output_dir(cfg)
    save_model = bool(cfg.save_model)

    set_seed(cfg.seed)
    logger.info(OmegaConf.to_yaml(cfg))

    id_to_label = inverse_label_map(cfg)
    label_ids = label_map(cfg)
    logger.info(f"Label map: {label_ids}")

    set_preprocessor(cfg.model_name)

    train_df, val_df, test_df = _load_data(cfg)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # ---- Teacher: frozen AraBERT baseline checkpoint ----
    teacher = load_teacher(cfg, device, id_to_label)
    teacher_config = AutoConfig.from_pretrained(cfg.teacher_model)

    # ---- Student: same tokenizer/vocab, fewer encoder layers ----
    student = build_student(cfg, teacher_config, tokenizer, id_to_label)

    # ---- Precompute teacher logits so the student objective is a pure function ----
    logger.info("Precomputing / loading teacher logits over train/val splits...")
    train_teacher_logits = _cached_teacher_logits(
        teacher, tokenizer, train_df["text"].tolist(), "train", cfg
    )
    val_teacher_logits = _cached_teacher_logits(
        teacher, tokenizer, val_df["text"].tolist(), "val", cfg
    )

    train_ds = build_train_dataset(
        train_df, tokenizer, label_ids, cfg, train_teacher_logits
    )
    val_ds = build_train_dataset(val_df, tokenizer, label_ids, cfg, val_teacher_logits)

    args = build_training_args(cfg)
    collator = DistillCollator(tokenizer)

    trainer = DistillationTrainer(
        alpha=float(cfg.alpha),
        temperature=float(cfg.temperature),
        model=student,
        args=args,
        data_collator=collator,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=lambda p: compute_metrics(p, cfg),
        callbacks=[_BestStepLogger()],
    )

    mlflow.set_experiment(cfg.experiment_name)
    with mlflow.start_run(run_name="distilled") as run:
        mlflow.log_params(loggable_params(cfg, skip=_LOGGABLE_SKIP))
        mlflow.log_param("model_name", "distilled")
        mlflow.log_param("layers", int(cfg.student_layers))
        mlflow.log_param("temperature", float(cfg.temperature))
        mlflow.log_param("alpha", float(cfg.alpha))

        logger.info(f"MLflow run ID: {run.info.run_id}")

        train_result = trainer.train()
        merged = {**train_result.metrics, **trainer.evaluate()}
        logger.info(f"Final validation metrics: {merged}")

        for key, value in merged.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, float(value))

        # Test-set accuracy / f1_macro (comparable to the baseline harness).
        test_metrics = _evaluate_test(
            student, tokenizer, test_df, cfg, save_hard_predictions=True
        )
        for key, value in test_metrics.items():
            mlflow.log_metric(key, float(value))
        logger.info(f"Test metrics: {test_metrics}")

        mlflow.log_param(
            "tokenizer_vocab_size", getattr(tokenizer, "vocab_size", "N/A")
        )
        mlflow.log_param("run_tag", str(cfg.run_tag))

        if save_model:
            final_dir = Path(cfg.output_dir) / "final"
            trainer.save_model(str(final_dir))
            mlflow.transformers.log_model(
                transformers_model={"model": student, "tokenizer": tokenizer},
                task="text-classification",
                artifact_path="model",
                pip_requirements=pinned_pip_requirements(),
            )
            logger.info(f"Saved student to {final_dir}")
