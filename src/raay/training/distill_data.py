"""Dataset, ``TrainingArguments`` and callback for the distillation run."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from datasets import Dataset
from loguru import logger
from omegaconf import DictConfig
from transformers import TrainerCallback, TrainingArguments

from raay.training.training_common import _preprocess_fn, resolve_output_dir


def build_train_dataset(
    df: pd.DataFrame,
    tokenizer: Any,
    label_map: dict[str, int],
    cfg: DictConfig,
    teacher_logits: np.ndarray,
) -> Dataset:
    texts = df["text"].map(_preprocess_fn).tolist()
    labels = df["label"].map(label_map).tolist()

    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=cfg.max_length,
    )
    return Dataset.from_dict(
        {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": labels,
            "teacher_logits": [row.tolist() for row in teacher_logits],
        }
    )


def build_training_args(cfg: DictConfig) -> TrainingArguments:
    # transformers uses a *negative* max_steps as the sentinel for
    # "derive total steps from num_train_epochs". Map our "use epochs" default
    # (max_steps == 0) to -1, as in the baseline trainer.
    max_steps = cfg.max_steps if int(cfg.max_steps) > 0 else -1
    save_model = bool(cfg.save_model)
    return TrainingArguments(
        output_dir=resolve_output_dir(cfg),
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
        # Keep the precomputed `teacher_logits` column through the training
        # dataloader: with the default remove_unused_columns=True the Trainer
        # strips any column missing from the model's forward() signature, which
        # would silently drop the teacher logits our DistillCollator/loss need.
        remove_unused_columns=False,
    )


class _BestStepLogger(TrainerCallback):
    """Report the step at which the best eval checkpoint was found."""

    def on_evaluate(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if state.best_metric is not None:
            logger.info(
                f"Best step: {state.best_model_checkpoint} "
                f"(metric={state.best_metric:.4f})"
            )
