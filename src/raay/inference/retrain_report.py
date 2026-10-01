"""Report and MLflow provenance for the retrain decision."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from raay.inference.retrain_decision import (
    _REASON_SEASONAL,
    TRIGGER_REASONS,
    RetrainDecision,
)


def mlflow_metrics(decision: RetrainDecision) -> dict[str, float]:
    """Metrics for the ``raay_batch`` run.

    Only real numbers are logged. A missing worst-column score would otherwise
    be logged as 0.0 and draw a healthy line on the retrain chart for a night
    that never breached anything.
    """
    metrics: dict[str, float] = {
        "retrain_triggered": float(decision.triggered),
        "retrain_psi_escalate": float(decision.psi.detail.get("escalate", False)),
    }
    worst = decision.psi.detail.get("worst") or {}
    if worst.get("drift_score") is not None:
        metrics["retrain_worst_psi"] = float(worst["drift_score"])
    if decision.reason == _REASON_SEASONAL:
        event = decision.calendar.detail.get("event") or {}
        if event.get("days_until") is not None:
            metrics["retrain_event_days_until"] = float(event["days_until"])
    return metrics


def mlflow_tags(decision: RetrainDecision) -> dict[str, str]:
    """Provenance tags. Strings, so the reason is queryable in the MLflow UI."""
    worst = decision.psi.detail.get("worst") or {}
    tags = {
        "trigger_reason": decision.reason,
        "run_type": "retrain-trigger",
        "psi_triggered": str(decision.psi.fired).lower(),
        "calendar_triggered": str(decision.calendar.fired).lower(),
        "escalate": str(decision.psi.detail.get("escalate", False)).lower(),
    }
    if worst.get("column"):
        tags["psi_worst_column"] = str(worst["column"])
    if worst.get("drift_score") is not None:
        tags["psi_worst_score"] = str(worst["drift_score"])
    if worst.get("source"):
        tags["psi_source"] = str(worst["source"])
    event = decision.calendar.detail.get("event") or {}
    if event.get("name"):
        tags["seasonal_event"] = str(event["name"])
    return tags


def build_report(
    decision: RetrainDecision,
    input_path: str | Path | None,
    prediction_path: str | Path | None,
    dispatch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The on-disk report, in the shape of the other drift report families."""
    return {
        **decision.as_dict(),
        "reasons_available": list(TRIGGER_REASONS),
        "inputs": {
            "input_drift_report": str(input_path) if input_path else None,
            "prediction_drift_report": str(prediction_path)
            if prediction_path
            else None,
        },
        "dispatch": dispatch
        or {
            "attempted": False,
            "reason_not_attempted": "no dispatch was requested",
        },
        "handoff": (
            "This job does not fine-tune. Retraining is scripts/kaggle_train_runs.py "
            "on a Kaggle GPU; scripts/promote_model.py is the only code allowed to "
            "move the Production alias."
        ),
        "caveat": (
            "Both drift panels are seeded draws from data/processed/test.csv, so a "
            "breach here reflects the test split, not production traffic. The "
            "seasonal signal is calendar-only and cannot be validated: no row in "
            "this corpus carries a timestamp."
        ),
    }
