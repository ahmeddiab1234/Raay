"""The retrain verdict: reason precedence and its evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from loguru import logger

from raay.inference.retrain_calendar import SeasonalEvent, evaluate_calendar_trigger
from raay.inference.retrain_signals import SignalEvidence, evaluate_psi_trigger

#: The reasons a retrain can be triggered, in the order they are evaluated.
#: Exposed as a tuple (not an Enum) because the value is written straight into
#: the report and onto an MLflow run tag, where a plain string reads better.
TRIGGER_REASONS: tuple[str, ...] = ("manual", "psi_breach", "scheduled", "none")

_REASON_MANUAL = "manual"
_REASON_PSI = "psi_breach"
_REASON_SEASONAL = "scheduled"
_REASON_NONE = "none"


@dataclass(frozen=True)
class RetrainDecision:
    """The verdict for one night, and the evidence behind it."""

    triggered: bool
    reason: str
    psi: SignalEvidence
    calendar: SignalEvidence
    day: date
    forced: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "date": self.day.isoformat(),
            "triggered": self.triggered,
            "reason": self.reason,
            "forced": self.forced,
            "psi": self.psi.as_dict(),
            "calendar": self.calendar.as_dict(),
        }


def decide(
    psi: SignalEvidence,
    calendar: SignalEvidence,
    day: date,
    force_reason: str | None = None,
) -> RetrainDecision:
    """Apply the precedence: manual > psi_breach > scheduled > none.

    ``force_reason`` is the manual path. It bypasses detection but still
    reports both signals' evidence, so an operator-forced retrain is auditable
    against what the gates actually said that night.
    """
    if force_reason == _REASON_MANUAL:
        return RetrainDecision(
            triggered=True,
            reason=_REASON_MANUAL,
            psi=psi,
            calendar=calendar,
            day=day,
            forced=True,
        )
    if psi.fired:
        return RetrainDecision(
            triggered=True, reason=_REASON_PSI, psi=psi, calendar=calendar, day=day
        )
    if calendar.fired:
        return RetrainDecision(
            triggered=True,
            reason=_REASON_SEASONAL,
            psi=psi,
            calendar=calendar,
            day=day,
        )
    return RetrainDecision(
        triggered=False, reason=_REASON_NONE, psi=psi, calendar=calendar, day=day
    )


def _read_json(path: str | Path | None) -> dict[str, Any] | None:
    """Read a report, tolerating absence.

    A missing report is "no signal", never a breach: the trigger must not fire
    because a report failed to be written.
    """
    if path is None:
        return None
    file = Path(path)
    if not file.exists():
        logger.warning(f"No report at {file}; treating as no signal.")
        return None
    try:
        return json.loads(file.read_text())
    except json.JSONDecodeError as error:
        logger.warning(f"Report {file} is not valid JSON ({error}); no signal.")
        return None


def evaluate_trigger(
    day: date,
    input_report: dict[str, Any] | None,
    prediction_report: dict[str, Any] | None,
    events: list[SeasonalEvent],
    thresholds: tuple[float, float] = (0.1, 0.2),
    lead_days: int = 21,
    escalate_fires: bool = False,
    force_reason: str | None = None,
) -> RetrainDecision:
    """The whole decision, from parsed reports to a verdict."""
    psi = evaluate_psi_trigger(
        input_report, prediction_report, thresholds, escalate_fires
    )
    calendar = evaluate_calendar_trigger(events, day, lead_days)
    return decide(psi, calendar, day, force_reason)
