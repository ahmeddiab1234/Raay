"""Kaggle GPU driver: run the fine-tuning / distillation sweep and register best.

Designed to be pasted into a Kaggle notebook (GPU accelerator) footer, or run
standalone against a local snapshot. It:

    1. Reads the processed splits from ``DATA_ROOT``
       (defaults to ``data/processed`` at the repo root; on Kaggle set
       ``DATA_ROOT=/kaggle/input/<dataset>/data/processed``).
    2. Launches >= ``N`` independent runs varying the sweep hyper-parameters
       (each a fresh subprocess so Hydra recomposes cleanly). Sweep candidates
       run with ``save_model=false`` so they only log metrics/params (no
       checkpoints, no model artifacts) — keeps the Kaggle working disk from
       filling up.
    3. Artifacts/metrics land in the session-local MLflow file store
       (``file:<MLFLOW_TRACKING_URI>``), one experiment ``raay_training``.
    4. Selects the best run by ``eval_f1_macro``, retrains that exact config
       with ``save_model=true``, and registers its model artifacts in stage
       ``Production``.

``--module train`` (default) sweeps the **baseline** fine-tune over
learning_rate / batch_size and registers ``ArabicSentiment``. ``--module
distill`` sweeps **knowledge distillation** over learning_rate / batch_size /
alpha / temperature (a 6-layer AraBERT student trained against the frozen
teacher) and registers ``ArabicSentimentDistilled``. Distill needs the teacher
checkpoint: upload the trained ``models/baseline/final`` dir as a Kaggle dataset
and point ``KAGGLE_TEACHER_DIR`` (or ``--teacher-dir``) at its mounted path.

After the run, copy the session ``mlruns/`` into the local repo and run
``scripts/merge_mlflow.py`` to merge them into ``mlflow/mlruns``.

Usage:
    DATA_ROOT=/kaggle/input/raay-splits/data/processed \
    KAGGLE_TEACHER_DIR=/kaggle/input/raay-teacher/final \
    MLFLOW_TRACKING_URI=file:/kaggle/working/mlruns \
    uv run python scripts/kaggle_train_runs.py --module distill --n 6
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import cast

from loguru import logger

import mlflow
from raay.config.env import env_str, load_environment, mlflow_tracking_uri
from raay.enums.constants import DefaultPaths, EnvVar, Experiments, Models

SWEEP = [
    # (learning_rate, batch_size)
    (2e-5, 16),
    (3e-5, 16),
    (5e-5, 16),
    (2e-5, 32),
    (3e-5, 32),
    (5e-5, 8),
]

DISTILL_SWEEP = [
    # (learning_rate, batch_size, alpha, temperature)
    (2e-5, 16, 0.4, 4.0),
    (3e-5, 16, 0.4, 4.0),
    (5e-5, 16, 0.4, 4.0),
    (3e-5, 16, 0.3, 3.0),
    (3e-5, 16, 0.5, 4.0),
    (3e-5, 32, 0.4, 4.0),
]


def _tracking_uri() -> str:
    uri = mlflow_tracking_uri(
        default="file:" + str(Path(DefaultPaths.LOCAL_MLRUNS.value).resolve())
    )
    return uri


def _data_overrides() -> list[str]:
    """Hydra overrides pointing at the processed-DataFrame snapshot.

    On Kaggle the processed splits arrive as a Dataset snapshot mounted under
    ``/kaggle/input/<dataset>/``; set ``DATA_ROOT`` (legacy ``KAGGLE_DATA_DIR``)
    to that directory so the runs read from there instead of the repo-relative
    defaults. Locally, unset -> default to ``data/processed``.
    """
    root = env_str(EnvVar.KAGGLE_DATA_DIR) or env_str(EnvVar.DATA_ROOT)
    if not root:
        return []
    root_path = Path(root)
    for name in ("train", "val", "test"):
        cand = root_path / f"{name}.csv"
        if not cand.exists():
            raise FileNotFoundError(
                f"{cand} not found. Point {EnvVar.DATA_ROOT.value} at the dir "
                "containing the processed train/val/test.csv snapshot."
            )
    return [
        f"train_file={root_path / 'train.csv'}",
        f"val_file={root_path / 'val.csv'}",
        f"test_file={root_path / 'test.csv'}",
    ]


def _teacher_dir(args: argparse.Namespace) -> str | None:
    """Resolve the teacher checkpoint dir (distill only).

    ``--teacher-dir`` wins over ``KAGGLE_TEACHER_DIR``; if neither is set the
    Hydra config default (``models/baseline/final``) is used, which is the
    correct value for standalone/local runs.
    """
    return args.teacher_dir or env_str(EnvVar.KAGGLE_TEACHER_DIR) or None


def _train_command(
    repo_root: Path, lr: float, bs: int, *, save_model: bool, run_tag: str
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "raay.training.train",
        f"learning_rate={lr}",
        f"batch_size={bs}",
        f"save_model={'true' if save_model else 'false'}",
        f"run_tag={run_tag}",
        f"hydra.run.dir=./{DefaultPaths.OUTPUT_DIR_TRAIN.value}",
        *_data_overrides(),
    ]


def _distill_command(
    repo_root: Path,
    lr: float,
    bs: int,
    alpha: float,
    temperature: float,
    *,
    save_model: bool,
    run_tag: str,
    teacher_dir: str | None,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "raay.training.distill",
        f"learning_rate={lr}",
        f"batch_size={bs}",
        f"alpha={alpha}",
        f"temperature={temperature}",
        f"save_model={'true' if save_model else 'false'}",
        f"run_tag={run_tag}",
        f"hydra.run.dir=./{DefaultPaths.OUTPUT_DIR_DISTILL.value}",
        *_data_overrides(),
    ]
    if teacher_dir:
        cmd.append(f"teacher_model={teacher_dir}")
    return cmd


def _run_filter(mode: str, run_tag: str | None = None) -> str:
    """MLflow search filter scoping runs to the requested module.

    Baseline and distillation runs share the ``raay_training`` experiment, so
    when selecting the best / registering we must not let a run from the other
    module leak in. Distill runs log ``model_name=distilled`` (baseline logs the
    teacher repo id).
    """
    base = "attributes.status = 'FINISHED'"
    if mode == "distill":
        base += " and params.model_name = 'distilled'"
    if run_tag:
        base += f" and params.run_tag = '{run_tag}'"
    return base


def run_sweep(n: int, mode: str, teacher_dir: str | None) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    combos = cast(list, SWEEP[:n] if mode == "train" else DISTILL_SWEEP[:n])
    logger.info(f"Will run {len(combos)} {mode} runs")
    logger.info(f"Data overrides: {_data_overrides() or 'repo-default data/processed'}")
    if mode == "distill":
        logger.info(
            f"Teacher: {teacher_dir or 'config default (models/baseline/final)'}"
        )

    for combo in combos:
        if mode == "train":
            lr, bs = combo
            cmd = _train_command(repo_root, lr, bs, save_model=False, run_tag="sweep")
        else:
            lr, bs, alpha, temperature = combo
            cmd = _distill_command(
                repo_root,
                lr,
                bs,
                alpha,
                temperature,
                save_model=False,
                run_tag="sweep",
                teacher_dir=teacher_dir,
            )
        logger.info(f"Launching: {cmd}")
        result = subprocess.run(cmd, cwd=repo_root, check=False)
        if result.returncode != 0:
            logger.error(f"Run failed ({combo}) rc={result.returncode}")
        else:
            logger.info(f"Run finished ({combo})")


def find_best(experiment_name: str, mode: str) -> mlflow.entities.Run:
    mlflow.set_tracking_uri(_tracking_uri())
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise RuntimeError(
            f"Experiment {experiment_name!r} not found; cannot select best."
        )

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=_run_filter(mode),
        order_by=["metrics.eval_f1_macro DESC"],
    )
    if not runs:
        raise RuntimeError("No finished runs with eval_f1_macro to select best.")
    return runs[0]


def retrain_best(
    best: mlflow.entities.Run,
    repo_root: Path,
    mode: str,
    teacher_dir: str | None,
) -> None:
    lr = best.data.params.get("learning_rate")
    bs = best.data.params.get("batch_size")
    if lr is None or bs is None:
        raise RuntimeError(
            f"Best run {best.info.run_id} missing lr/bs params; cannot retrain."
        )
    lr, bs = float(lr), int(bs)
    logger.info(f"Retraining best config (lr={lr}, bs={bs}) with save_model=true")

    if mode == "train":
        cmd = _train_command(repo_root, lr, bs, save_model=True, run_tag="final")
    else:
        alpha = float(best.data.params.get("alpha", 0.4))
        temperature = float(best.data.params.get("temperature", 4.0))
        logger.info(f"Distill best config (alpha={alpha}, temperature={temperature})")
        cmd = _distill_command(
            repo_root,
            lr,
            bs,
            alpha,
            temperature,
            save_model=True,
            run_tag="final",
            teacher_dir=teacher_dir,
        )

    result = subprocess.run(cmd, cwd=repo_root, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Retrain of best config failed (lr={lr}, bs={bs}) rc={result.returncode}"
        )


def register_best(experiment_name: str, model_name: str, stage: str, mode: str) -> str:
    mlflow.set_tracking_uri(_tracking_uri())
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise RuntimeError(
            f"Experiment {experiment_name!r} not found; cannot register."
        )

    # Register only the retrained `final` run (save_model=true); sweep
    # candidates carry no model artifacts.
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=_run_filter(mode, run_tag="final"),
        order_by=["metrics.eval_f1_macro DESC"],
    )
    if not runs:
        raise RuntimeError("No 'final' run (save_model=true) to register.")

    best = runs[0]
    best_f1 = best.data.metrics.get("eval_f1_macro", best.data.metrics.get("f1_macro"))
    logger.info(
        f"Registering run {best.info.run_id} (eval_f1_macro={best_f1}) "
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
    teacher_dir = _teacher_dir(args)
    model_name = args.model_name or (
        Models.REGISTERED_DISTILLED.value
        if mode == "distill"
        else Models.REGISTERED_BASELINE.value
    )

    if not args.no_train:
        run_sweep(args.n, mode, teacher_dir)
        best = find_best(args.experiment, mode)
        retrain_best(best, repo_root, mode, teacher_dir)

    register_best(args.experiment, model_name, args.stage, mode)


if __name__ == "__main__":
    main()
