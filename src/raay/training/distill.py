"""Distil a smaller AraBERT student against the frozen baseline teacher.

Run from the repo root (DVC / Kaggle launch it there and every path below is
relative to the repo root):

    uv run python -m raay.training.distill                        # Hydra defaults
    uv run python -m raay.training.distill alpha=0.4 temperature=4.0

The teacher is the fine-tuned baseline checkpoint
(``models/baseline/final`` by default) and the student is the same tokenizer /
vocabulary with ``student_layers`` encoder layers, trained on

    loss = alpha * CE(student, labels)
         + (1 - alpha) * T^2 * KL(softmax(student/T), softmax(teacher/T))

Teacher logits depend only on the teacher and the data, not on the distillation
knobs, so they are precomputed once and cached under
``cfg.teacher_logits_cache`` and reused across a hyperparameter sweep.

Implementation lives in siblings -- ``distill_objective`` (loss, trainer,
collator), ``distill_teacher`` (teacher forward + logit cache), ``distill_data``
(dataset, ``TrainingArguments``, best-step callback), ``distill_models``
(teacher/student construction), ``distill_evaluate`` (test-split scoring),
``distill_run`` (the Hydra entry point), plus the shared ``training_common``
-- re-exported here so ``raay.training.distill.<name>`` keeps one import path.
"""

from __future__ import annotations

import pandas as pd
from omegaconf import DictConfig

from raay.training.distill_data import (
    _BestStepLogger,
    build_train_dataset,
    build_training_args,
)
from raay.training.distill_evaluate import _evaluate_test
from raay.training.distill_objective import (
    DistillationTrainer,
    DistillCollator,
    distill_loss,
)
from raay.training.distill_run import main
from raay.training.distill_teacher import (
    _cached_teacher_logits,
    _preprocess_batch,
    _resolve_cache_path,
    _teacher_logits,
)
from raay.training.training_common import (
    compute_metrics,
    inverse_label_map,
    label_map,
    loggable_params,
    pinned_pip_requirements,
    resolve_output_dir,
)

__all__ = [
    "DistillCollator",
    "DistillationTrainer",
    "build_train_dataset",
    "build_training_args",
    "compute_metrics",
    "distill_loss",
    "main",
]

# Kept importable from here: the cache tests address these by name. The tests
# monkeypatch ``_teacher_logits`` on ``raay.training.distill_teacher`` -- the
# module that actually calls it -- so patching this façade attribute would
# silently leave the real teacher forward in place.
_PRIVATE_REEXPORTS = (
    _BestStepLogger,
    _cached_teacher_logits,
    _evaluate_test,
    _preprocess_batch,
    _resolve_cache_path,
    _teacher_logits,
)


def load_data(cfg: DictConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Plain split loading: the distiller never trains on feedback overrides."""
    return (
        pd.read_csv(cfg.train_file),
        pd.read_csv(cfg.val_file),
        pd.read_csv(cfg.test_file),
    )


# Underscored aliases the original module exposed, kept importable under the same
# names as a plain-import façade.
_label_map = label_map
_inverse_label_map = inverse_label_map
_loggable_params = loggable_params
_pinned_pip_requirements = pinned_pip_requirements
_resolve_output_dir = resolve_output_dir


if __name__ == "__main__":
    main()
