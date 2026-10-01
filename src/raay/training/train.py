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

The implementation lives in siblings -- ``training_common`` (preprocessor slot,
label maps, eval metrics, MLflow param/pip helpers shared with the distiller),
``train_data`` (split loading + feedback merge, tokenization), ``train_args``
(``TrainingArguments``), and ``train_run`` (the Hydra entry point) --
re-exported here so ``raay.training.train.load_data`` and friends keep one
import path.
"""

from __future__ import annotations

from raay.training.train_args import build_training_args
from raay.training.train_data import load_data, tokenize_and_encode
from raay.training.train_run import main
from raay.training.training_common import (
    compute_metrics,
    inverse_label_map,
    label_map,
    loggable_params,
    pinned_pip_requirements,
    resolve_output_dir,
)

__all__ = [
    "build_training_args",
    "compute_metrics",
    "inverse_label_map",
    "label_map",
    "load_data",
    "loggable_params",
    "main",
    "pinned_pip_requirements",
    "resolve_output_dir",
    "tokenize_and_encode",
]

# Underscored aliases the original module exposed; kept importable because the
# sweep driver and tests address them by those names.
_label_map = label_map
_inverse_label_map = inverse_label_map
_loggable_params = loggable_params
_pinned_pip_requirements = pinned_pip_requirements
_resolve_output_dir = resolve_output_dir


if __name__ == "__main__":
    main()
