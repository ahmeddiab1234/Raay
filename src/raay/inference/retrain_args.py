"""CLI argument parsing and the GitHub dispatch payload for the retrain trigger."""

from __future__ import annotations

import argparse
from typing import Any

from raay.enums.constants import DefaultPaths
from raay.inference.retrain_decision import _REASON_MANUAL, RetrainDecision
from raay.inference.retrain_dispatch import _DEFAULT_REPOSITORY


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--date", default=None, help="YYYY-MM-DD (default today)")
    parser.add_argument(
        "--input-drift-report",
        default=None,
        help="Phase 6 step 1 report (default reports/drift/{date}.json)",
    )
    parser.add_argument(
        "--prediction-drift-report",
        default=None,
        help="Phase 6 step 2 report (default reports/prediction_drift/{date}.json)",
    )
    parser.add_argument(
        "--calendar",
        default=DefaultPaths.CONFIG_SEASONAL_EVENTS.value,
        help="Seasonal event config (configs/seasonal_events.yaml).",
    )
    parser.add_argument("--report-out", default=None)
    parser.add_argument(
        "--lead-days",
        type=int,
        default=21,
        help="How far ahead a confirmed seasonal event arms the trigger.",
    )
    parser.add_argument("--warn-threshold", type=float, default=0.1)
    parser.add_argument("--fail-threshold", type=float, default=0.2)
    parser.add_argument(
        "--escalate-fires",
        action="store_true",
        help=(
            "Also fire on prediction_drift's `escalate` (falling confidence AND "
            "rising class PSI). Off by default: escalate is a coupled signal "
            "deliberately kept out of its own `overall` verdict."
        ),
    )
    parser.add_argument(
        "--force-reason",
        choices=[_REASON_MANUAL],
        default=None,
        help="Fire regardless of the gates (manual operator retrain).",
    )
    parser.add_argument(
        "--token-file",
        default=None,
        help=(
            "Path to the GitHub dispatch token. Defaults to "
            "RAAY_GITHUB_DISPATCH_TOKEN_FILE, then RAAY_GITHUB_DISPATCH_TOKEN. "
            "The token is read from a file, never argv."
        ),
    )
    parser.add_argument(
        "--repository",
        default=None,
        help=(
            "owner/repo to dispatch to (default "
            f"{_DEFAULT_REPOSITORY}, overridable via RAAY_GITHUB_REPOSITORY)."
        ),
    )
    parser.add_argument(
        "--no-dispatch",
        action="store_true",
        help="Evaluate and report the decision without POSTing to GitHub.",
    )
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging.")
    return parser.parse_args(argv)


def build_dispatch_payload(
    decision: RetrainDecision,
    prediction_report: dict[str, Any] | None,
    threshold: float,
) -> dict[str, Any]:
    """The ``client_payload`` the receiver's ``notify`` job reads.

    Keys mirror ``.github/workflows/retrain.yml``: ``reason`` is echoed in the
    summary, ``threshold`` overrides the refresh gate, and the rest is rendered
    into the issue body.
    """
    worst = decision.psi.detail.get("worst") or {}
    event = decision.calendar.detail.get("event") or {}
    return {
        "reason": decision.reason,
        "trigger_date": decision.day.isoformat(),
        "threshold": threshold,
        "psi": worst.get("drift_score"),
        "psi_column": worst.get("column"),
        "triage": (prediction_report or {}).get("triage"),
        "seasonal_event": event.get("name"),
    }
