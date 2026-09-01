"""Kaggle GPU driver: run the baseline fine-tuning sweep and register the best.

Designed to be pasted into a Kaggle notebook (GPU accelerator) footer, or run
standalone against a local snapshot. It:

    1. Reads the processed splits from ``DATA_ROOT``
       (defaults to ``data/processed`` at the repo root; on Kaggle set
       ``DATA_ROOT=/kaggle/input/<dataset>/data/processed``).
    2. Launches >= ``N`` independent train runs varying learning_rate /
       batch_size (each a fresh subprocess so Hydra recomposes cleanly).
    3. Artifacts/metrics land in the session-local MLflow file store
       (``file:<MLFLOW_TRACKING_URI>``), one experiment ``raay_training``.
    4. Selects the best run by ``eval_f1_macro`` and registers it as the
       ``ArabicSentiment`` model in stage ``Production``.

After the run, copy the session ``mlruns/`` into the local repo and run
``scripts/merge_mlflow.py`` to merge them into ``mlflow/mlruns``.

Usage:
    DATA_ROOT=/kaggle/input/raay-splits/data/processed \
    MLFLOW_TRACKING_URI=file:/kaggle/working/mlruns \
    uv run python scripts/kaggle_train_runs.py --n 6
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from loguru import logger

import mlflow

SWEEP = [
    # (learning_rate, batch_size)
    (2e-5, 16),
    (3e-5, 16),
    (5e-5, 16),
    (2e-5, 32),
    (3e-5, 32),
    (5e-5, 8),
]


def _tracking_uri() -> str:
    uri = os.environ.get("MLFLOW_TRACKING_URI", "file:" + str(Path("mlruns").resolve()))
    if uri.startswith("file:"):
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    return uri


def _data_overrides() -> list[str]:
    """Hydra overrides pointing at the processed-DataFrame snapshot.

    On Kaggle the processed splits arrive as a Dataset snapshot mounted under
    ``/kaggle/input/<dataset>/``; set ``KAGGLE_DATA_DIR`` (or ``DATA_ROOT``) to
    that directory so the runs read from there instead of the repo-relative
    defaults. Locally, unset -> default to ``data/processed``.
    """
    root = os.environ.get("KAGGLE_DATA_DIR") or os.environ.get("DATA_ROOT")
    if not root:
        return []
    root_path = Path(root)
    for name in ("train", "val", "test"):
        cand = root_path / f"{name}.csv"
        if not cand.exists():
            raise FileNotFoundError(
                f"{cand} not found. Point DATA_ROOT at the dir containing the "
                "processed train/val/test.csv snapshot."
            )
    return [
        f"train_file={root_path / 'train.csv'}",
        f"val_file={root_path / 'val.csv'}",
        f"test_file={root_path / 'test.csv'}",
    ]


def run_sweep(n: int) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    combos = SWEEP[:n]
    logger.info(f"Will run {len(combos)} training runs")
    data_overrides = _data_overrides()
    logger.info(f"Data overrides: {data_overrides or 'repo-default data/processed'}")

    for lr, bs in combos:
        cmd = [
            sys.executable,
            "-m",
            "raay.training.train",
            f"learning_rate={lr}",
            f"batch_size={bs}",
            "hydra.run.dir=./outputs/train",
            *data_overrides,
        ]
        logger.info(f"Launching: {cmd}")
        result = subprocess.run(cmd, cwd=repo_root, check=False)
        if result.returncode != 0:
            logger.error(f"Run failed (lr={lr}, bs={bs}) rc={result.returncode}")
        else:
            logger.info(f"Run finished (lr={lr}, bs={bs})")


def register_best(experiment_name: str, model_name: str, stage: str) -> str:
    mlflow.set_tracking_uri(_tracking_uri())
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise RuntimeError(
            f"Experiment {experiment_name!r} not found; cannot register."
        )

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["metrics.eval_f1_macro DESC"],
    )
    if not runs:
        raise RuntimeError("No finished runs with eval_f1_macro to register.")

    best = runs[0]
    best_f1 = best.data.metrics.get("eval_f1_macro", best.data.metrics.get("f1_macro"))
    logger.info(
        f"Best run {best.info.run_id} (eval_f1_macro={best_f1}) "
        f"lr={best.data.params.get('learning_rate')} "
        f"bs={best.data.params.get('batch_size')}"
    )

    model_uri = f"runs:/{best.info.run_id}/model"
    registered = mlflow.register_model(model_uri=model_uri, name=model_name)
    client.transition_model_version_stage(
        name=model_name,
        version=registered.version,
        stage=stage,
        archive_existing_versions=True,
    )
    logger.info(f"Registered {model_name} v{registered.version} -> {stage}")
    return registered.version


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=len(SWEEP), help="Number of runs")
    parser.add_argument("--experiment", default="raay_training")
    parser.add_argument("--model-name", default="ArabicSentiment")
    parser.add_argument("--stage", default="Production")
    parser.add_argument(
        "--no-train", action="store_true", help="Skip training, only register"
    )
    args = parser.parse_args()

    if not args.no_train:
        run_sweep(args.n)

    register_best(args.experiment, args.model_name, args.stage)


if __name__ == "__main__":
    main()
