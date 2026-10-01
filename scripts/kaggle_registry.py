"""Run selection + registration for the Kaggle sweep.

Baseline and distill runs share the ``raay_training`` experiment, so every search
is scoped by a filter: distill runs log ``model_name=distilled`` (baseline logs
the teacher repo id), and registration only considers ``run_tag=final`` runs
(``save_model=true``) because sweep candidates carry no model artifacts.

The sweep driver itself is here too: it launches the commands from
``kaggle_runs`` as subprocesses and logs failures without aborting the rest of
the sweep.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import cast

import mlflow
from kaggle_runs import DISTILL_SWEEP, SWEEP, distill_command, train_command
from loguru import logger

from raay.config.env import mlflow_tracking_uri
from raay.enums.constants import DefaultPaths, Models


def tracking_uri() -> str:
    return mlflow_tracking_uri(
        default="file:" + str(Path(DefaultPaths.LOCAL_MLRUNS.value).resolve())
    )


def run_filter(mode: str, run_tag: str | None = None) -> str:
    """MLflow search filter scoping runs to the requested module."""
    base = "attributes.status = 'FINISHED'"
    if mode == "distill":
        base += " and params.model_name = 'distilled'"
    if run_tag:
        base += f" and params.run_tag = '{run_tag}'"
    return base


def run_sweep(
    n: int,
    mode: str,
    teacher: str | None,
    repo_root: Path,
    overrides: list[str],
) -> None:
    combos = cast(list, SWEEP[:n] if mode == "train" else DISTILL_SWEEP[:n])
    logger.info(f"Will run {len(combos)} {mode} runs")
    logger.info(f"Data overrides: {overrides or 'repo-default data/processed'}")
    if mode == "distill":
        logger.info(f"Teacher: {teacher or 'config default (models/baseline/final)'}")

    for combo in combos:
        if mode == "train":
            lr, bs = combo
            cmd = train_command(lr, bs, save_model=False, run_tag="sweep")
        else:
            lr, bs, alpha, temperature = combo
            cmd = distill_command(
                lr,
                bs,
                alpha,
                temperature,
                save_model=False,
                run_tag="sweep",
                teacher=teacher,
            )
        logger.info(f"Launching: {cmd}")
        result = subprocess.run(cmd, cwd=repo_root, check=False)
        if result.returncode != 0:
            logger.error(f"Run failed ({combo}) rc={result.returncode}")
        else:
            logger.info(f"Run finished ({combo})")


def find_best(experiment_name: str, mode: str) -> mlflow.entities.Run:
    mlflow.set_tracking_uri(tracking_uri())
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise RuntimeError(
            f"Experiment {experiment_name!r} not found; cannot select best."
        )

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=run_filter(mode),
        order_by=["metrics.eval_f1_macro DESC"],
    )
    if not runs:
        raise RuntimeError("No finished runs with eval_f1_macro to select best.")
    return runs[0]


def retrain_best(
    best: mlflow.entities.Run,
    repo_root: Path,
    mode: str,
    teacher: str | None,
) -> None:
    """Re-run the winning config with ``save_model=true``."""
    lr = best.data.params.get("learning_rate")
    bs = best.data.params.get("batch_size")
    if lr is None or bs is None:
        raise RuntimeError(
            f"Best run {best.info.run_id} missing lr/bs params; cannot retrain."
        )
    lr, bs = float(lr), int(bs)
    logger.info(f"Retraining best config (lr={lr}, bs={bs}) with save_model=true")

    if mode == "train":
        cmd = train_command(lr, bs, save_model=True, run_tag="final")
    else:
        alpha = float(best.data.params.get("alpha", 0.4))
        temperature = float(best.data.params.get("temperature", 4.0))
        logger.info(f"Distill best config (alpha={alpha}, temperature={temperature})")
        cmd = distill_command(
            lr,
            bs,
            alpha,
            temperature,
            save_model=True,
            run_tag="final",
            teacher=teacher,
        )

    result = subprocess.run(cmd, cwd=repo_root, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Retrain of best config failed (lr={lr}, bs={bs}) rc={result.returncode}"
        )


def register_best(experiment_name: str, model_name: str, stage: str, mode: str) -> str:
    mlflow.set_tracking_uri(tracking_uri())
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise RuntimeError(
            f"Experiment {experiment_name!r} not found; cannot register."
        )

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=run_filter(mode, run_tag="final"),
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


def resolve_model_name(requested: str | None, mode: str) -> str:
    """Default registered name per module when ``--model-name`` is not given."""
    return requested or (
        Models.REGISTERED_DISTILLED.value
        if mode == "distill"
        else Models.REGISTERED_BASELINE.value
    )
