"""``TrainingArguments`` for the baseline fine-tuning job."""

from __future__ import annotations

from omegaconf import DictConfig
from transformers import TrainingArguments

from raay.training.training_common import resolve_output_dir


def build_training_args(cfg: DictConfig) -> TrainingArguments:
    # transformers uses a *negative* max_steps as the sentinel for
    # "derive total steps from num_train_epochs". A 0 is treated as a real
    # step budget of zero (not epoch-based), which silently truncates training
    # to ~1 step. Map our "use epochs" default (max_steps == 0) to -1.
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
    )
