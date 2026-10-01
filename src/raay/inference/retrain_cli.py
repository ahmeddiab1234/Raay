"""The retrain-trigger CLI: evaluate, report, dispatch, log, exit."""

from __future__ import annotations

import json
import os
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from loguru import logger

from raay.enums.constants import DefaultPaths
from raay.inference.retrain_args import build_dispatch_payload, parse_args
from raay.inference.retrain_calendar import load_calendar
from raay.inference.retrain_decision import _read_json, evaluate_trigger
from raay.inference.retrain_dispatch import (
    _DEFAULT_REPOSITORY,
    dispatch_retrain,
    resolve_token,
)
from raay.inference.retrain_report import build_report, mlflow_metrics, mlflow_tags


def _log_run(
    run_name: str,
    metrics: dict[str, float],
    tags: dict[str, str],
    report_path: str,
) -> None:
    """Log the trigger to ``raay_batch`` with the reason as a queryable tag.

    Reuses batch_score's logger shape (idempotent experiment creation, no
    ``run_type`` collision) but adds ``set_tag`` calls, because the reason is
    the one field a human will filter the experiment by.
    """
    import mlflow

    from raay.config.env import mlflow_tracking_uri
    from raay.enums.constants import Experiments

    mlflow.set_tracking_uri(mlflow_tracking_uri())
    experiment = mlflow.get_experiment_by_name(Experiments.BATCH.value)
    if experiment is None:
        experiment_id = mlflow.create_experiment(Experiments.BATCH.value)
    else:
        experiment_id = experiment.experiment_id
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
        for tag_key, tag_value in tags.items():
            mlflow.set_tag(tag_key, tag_value)
        for metric_key, metric_value in metrics.items():
            mlflow.log_metric(metric_key, float(metric_value))
        mlflow.log_artifact(report_path, artifact_path="retrain_trigger")
        logger.info(f"Logged raay_batch run {run_name}")


def _attempt_dispatch(
    decision: Any,
    predictions: dict[str, Any] | None,
    args: Any,
    repository: str,
) -> dict[str, Any]:
    """Resolve the token and POST, or say why no POST happened."""
    token = None if args.no_dispatch else resolve_token(args.token_file)
    if not decision.triggered:
        return {
            "attempted": False,
            "reason_not_attempted": f"nothing triggered (reason={decision.reason})",
        }
    if token is None:
        return {
            "attempted": False,
            "reason_not_attempted": (
                "no dispatch token provisioned; the decision and its MLflow "
                "provenance are still recorded"
            ),
        }
    return dispatch_retrain(
        build_dispatch_payload(decision, predictions, args.fail_threshold),
        token,
        repository=repository,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # UTC, not naive local: the nightly DAG runs at 03:00 UTC and the trigger
    # report is keyed by that date, so a local-time date would file tonight's run
    # under the wrong day whenever the host's offset crosses midnight.
    day = date.fromisoformat(args.date) if args.date else datetime.now(tz=UTC).date()

    input_report_path = args.input_drift_report or str(
        Path(DefaultPaths.DRIFT_REPORTS.value) / f"{day.isoformat()}.json"
    )
    prediction_report_path = args.prediction_drift_report or str(
        Path(DefaultPaths.PREDICTION_DRIFT_REPORTS.value) / f"{day.isoformat()}.json"
    )
    report_path = args.report_out or str(
        Path(DefaultPaths.RETRAIN_TRIGGER_REPORTS.value) / f"{day.isoformat()}.json"
    )

    input_drift = _read_json(input_report_path)
    predictions = _read_json(prediction_report_path)
    events = load_calendar(args.calendar)
    decision = evaluate_trigger(
        day=day,
        input_report=input_drift,
        prediction_report=predictions,
        events=events,
        thresholds=(args.warn_threshold, args.fail_threshold),
        lead_days=args.lead_days,
        escalate_fires=args.escalate_fires,
        force_reason=args.force_reason,
    )

    repository = (
        args.repository
        or os.environ.get("RAAY_GITHUB_REPOSITORY")
        or _DEFAULT_REPOSITORY
    )
    dispatch = _attempt_dispatch(decision, predictions, args, repository)

    report = build_report(decision, input_report_path, prediction_report_path, dispatch)

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(report, indent=2, ensure_ascii=False))

    if not args.no_mlflow:
        _log_run(
            f"retrain-trigger-{day.isoformat()}",
            mlflow_metrics(decision),
            mlflow_tags(decision),
            report_path,
        )

    logger.info(
        f"Retrain trigger {day}: {decision.reason} "
        f"(triggered={decision.triggered}, psi={decision.psi.fired}, "
        f"calendar={decision.calendar.fired}) -> {report_path}"
    )
    # Report-only, like --mode drift and --mode predict-drift: a decision to
    # retrain is a recommendation to a human, not a job outcome, and no token
    # simply skips the dispatch. The exception is a dispatch that was *tried*
    # and failed -- a breach the operator asked to be notified about and wasn't
    # is an infrastructure fault, so it turns the task red (and Airflow retries)
    # rather than passing silently. The report is already on disk either way.
    if dispatch.get("attempted") and not dispatch.get("ok"):
        logger.error(f"GitHub dispatch failed: {dispatch.get('error')}")
        return 1
    return 0
