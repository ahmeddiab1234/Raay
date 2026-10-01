"""Kaggle GPU sweep driver: run N hyperparameter trials, retrain the best, register.

Real AraBERT fine-tuning only happens on a Kaggle GPU (this box has no CUDA), so
this script runs *on Kaggle* against the Dataset snapshot of ``data/processed/``.
After the session, copy ``mlruns/`` back and merge it with ``merge_mlflow.py``.

    DATA_ROOT=/kaggle/input/raay-splits/data/processed \
    KAGGLE_TEACHER_DIR=/kaggle/input/raay-teacher/final \
    MLFLOW_TRACKING_URI=file:/kaggle/working/mlruns \
    uv run python scripts/kaggle_train_runs.py --module distill --n 6

The sweep definitions and the command builders live in ``kaggle_runs``; run
selection/registration lives in ``kaggle_registry``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from kaggle_registry import (
    find_best,
    register_best,
    resolve_model_name,
    retrain_best,
    run_sweep,
)
from kaggle_runs import SWEEP, data_overrides, teacher_dir

from raay.config.env import load_environment
from raay.enums.constants import Experiments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=len(SWEEP), help="Number of runs")
    parser.add_argument("--experiment", default=Experiments.TRAINING.value)
    parser.add_argument(
        "--module",
        choices=["train", "distill"],
        default="train",
        help="train = baseline fine-tune sweep; distill = KD from the teacher",
    )
    parser.add_argument(
        "--teacher-dir",
        default=None,
        help="Dir containing the trained teacher checkpoint "
        "(distill only); overrides KAGGLE_TEACHER_DIR. Defaults to the Hydra "
        "config value (models/baseline/final).",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Registered model name; default ArabicSentiment (train) or "
        "ArabicSentimentDistilled (distill).",
    )
    parser.add_argument("--stage", default="Production")
    parser.add_argument(
        "--no-train", action="store_true", help="Skip sweep+retrain, only register"
    )
    args = parser.parse_args()

    load_environment()

    repo_root = Path(__file__).resolve().parent.parent
    mode = args.module
    teacher = teacher_dir(args.teacher_dir)
    model_name = resolve_model_name(args.model_name, mode)

    if not args.no_train:
        run_sweep(args.n, mode, teacher, repo_root, data_overrides())
        best = find_best(args.experiment, mode)
        retrain_best(best, repo_root, mode, teacher)

    register_best(args.experiment, model_name, args.stage, mode)


if __name__ == "__main__":
    main()
