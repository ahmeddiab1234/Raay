"""Per-mode runs for the nightly batch job: score, reference, drift.

Split out of ``batch_score`` so this module is only the argparse/MLflow shell and
each mode's body reads on its own.

Two of these modes are **report-only and exit 0** by design (``drift``,
``predict-drift``): a FAIL verdict lands in the JSON report, and the Airflow task
stays green so a quiet night does not page anyone. Only ``score`` enforces the
``--min-samples`` floor.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from raay.enums.constants import DefaultPaths
from raay.inference.batch_reference import init_reference, reference_artifacts
from raay.inference.batch_scoring import Scorer, score_input
from raay.inference.drift_engine import EngineeredDriftSpec, drift_check
from raay.inference.prediction_drift import (
    PredictionDriftSpec,
    mlflow_metrics,
    predict_drift_check,
)


def write_json(path: str, payload: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def run_init_reference(
    args: argparse.Namespace,
    scorer: Scorer,
    engineered: EngineeredDriftSpec | None,
) -> tuple[dict[str, float], list[tuple[str, str]]]:
    """Build the reference panel + its engineered/PCA artifacts; log metrics."""
    frame = init_reference(
        args.pool, args.samples, args.reference, scorer, engineered=engineered
    )
    if args.no_mlflow:
        return {}, []
    return reference_artifacts(frame, args.reference, engineered)


def run_score(
    args: argparse.Namespace, day: str, paths: dict[str, str], scorer: Scorer
) -> tuple[dict[str, float], list[tuple[str, str]]]:
    """Score the day's panel in large in-process batches; log metrics."""
    stats = score_input(
        paths["input"], paths["output"], scorer, min_samples=args.min_samples
    )
    write_json(paths["stats"], stats)
    if args.no_mlflow:
        return {}, []
    metrics = {
        "n_reviews": float(stats["n_reviews"]),
        "elapsed_sec": stats["elapsed_sec"],
        "mean_predicted_score": float(
            pd.read_csv(paths["output"])["predicted_score"].mean()
        ),
    }
    artifacts = [
        (f"score/{day}", paths["output"]),
        ("stats_today", paths["stats"]),
    ]
    return metrics, artifacts


def run_predict_drift(
    args: argparse.Namespace, day: str, paths: dict[str, str]
) -> tuple[dict[str, float], list[tuple[str, str]]]:
    """Phase 6 step 2: watch the model's outputs, end in a triage verdict."""
    report = predict_drift_check(
        PredictionDriftSpec(
            current_csv=args.current or paths["output"],
            reference_csv=args.reference,
            train_prior_csv=args.train_prior,
            input_drift_report=(
                args.input_drift_report
                if args.input_drift_report is not None
                else str(Path(DefaultPaths.DRIFT_REPORTS.value) / f"{day}.json")
            ),
            history_dir=DefaultPaths.PREDICTION_DRIFT_REPORTS.value,
            day=day,
            window=args.history_window,
            min_days=args.min_history_days,
        )
    )
    write_json(paths["predict_drift"], report)
    logger.info(
        f"Prediction drift check {report['output_drift']['overall']}: "
        f"triage={report['triage']} "
        f"psi_vs_prior={report['class_distribution']['psi_vs_training_prior']['drift_score']} "
        f"mean_confidence={report['confidence']['mean']}"
    )
    if args.no_mlflow:
        return {}, []
    return mlflow_metrics(report), [("prediction_drift", paths["predict_drift"])]


def run_drift(
    args: argparse.Namespace,
    day: str,
    paths: dict[str, str],
    engineered: EngineeredDriftSpec | None,
) -> tuple[dict[str, float], list[tuple[str, str]]]:
    """Phase 6 step 1: PSI on the inputs (model outputs + engineered features)."""
    override = (
        tuple(c.strip() for c in args.drift_columns.split(",") if c.strip())
        if args.drift_columns
        else None
    )
    verdict = drift_check(
        args.reference,
        args.current or paths["output"],
        paths["drift"],
        day,
        drift_columns=override,
        engineered=engineered,
    )
    if args.no_mlflow:
        return {}, []
    return drift_metrics(verdict), [("drift", paths["drift"])]


def drift_metrics(verdict: dict[str, Any]) -> dict[str, float]:
    """Only columns that actually produced a score.

    An ERRORed column has ``drift_score`` None, and logging a placeholder 0.0
    for it would draw a healthy line on a chart for a check that never ran.
    The trendable input-feature means and mixes are appended so a rising OOV
    rate is a visible series in the ``raay_batch`` experiment rather than
    something you have to open each JSON to find.
    """
    metrics = {
        col: data["drift_score"]
        for col, data in verdict["columns"].items()
        if data["drift_score"] is not None
    }
    metrics["drift_share"] = verdict["drift_share"]
    metrics["n_columns_errored"] = float(len(verdict["errored_columns"]))
    metrics["n_current"] = float(verdict["n_current"])
    metrics |= {
        f"mean_{col}": value for col, value in verdict.get("feature_means", {}).items()
    }
    metrics |= {
        f"oov_share_{bucket}": share
        for bucket, share in verdict.get("oov_bucket_mix", {}).items()
    }
    metrics |= {
        f"dialect_share_{dialect}": share
        for dialect, share in verdict.get("dialect_mix", {}).get("current", {}).items()
    }
    return metrics
