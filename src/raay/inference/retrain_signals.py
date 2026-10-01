"""The PSI signal: does either drift report cross the FAIL threshold?"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SignalEvidence:
    """Why one signal did or did not fire."""

    fired: bool
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"fired": self.fired, **self.detail}


def evaluate_psi_trigger(
    input_report: dict[str, Any] | None,
    prediction_report: dict[str, Any] | None,
    thresholds: tuple[float, float] = (0.1, 0.2),
    escalate_fires: bool = False,
) -> SignalEvidence:
    """Has anything crossed the FAIL threshold in either drift report?

    Fires on the *worst* observed column, and only ever on an explicit FAIL:
    ``WARN`` sits between the thresholds by construction, so re-deriving it
    here would fire on a band this function does not claim.
    """
    _, fail = thresholds
    worst_score: float | None = None
    worst: dict[str, Any] = {}
    counts = {"FAIL": 0, "WARN": 0, "SKIPPED": 0, "ERROR": 0}
    columns_seen = 0

    if input_report:
        for name, column in (input_report.get("columns") or {}).items():
            decision = str(column.get("decision", "ERROR")).upper()
            counts[decision] = counts.get(decision, 0) + 1
            columns_seen += 1
            score = column.get("drift_score")
            if (
                decision == "FAIL"
                and score is not None
                and score > (worst_score if worst_score is not None else -1.0)
            ):
                worst_score = float(score)
                worst = {
                    "source": "input_drift",
                    "column": name,
                    "drift_score": float(score),
                    "threshold": fail,
                    "source_report": input_report.get("date"),
                }

    # The output side carries no per-column map, just the verdict its class-mix
    # PSI produced, so the comparison name is recorded instead of a column.
    if prediction_report:
        output = prediction_report.get("output_drift") or {}
        decision = str(output.get("overall", "ERROR")).upper()
        counts[decision] = counts.get(decision, 0) + 1
        columns_seen += 1
        psi_block = (prediction_report.get("class_distribution") or {}).get(
            "psi_vs_training_prior"
        ) or {}
        score = psi_block.get("drift_score")
        if (
            decision == "FAIL"
            and score is not None
            and score > (worst_score if worst_score is not None else -1.0)
        ):
            worst_score = float(score)
            worst = {
                "source": "prediction_drift",
                "column": "class_distribution_vs_training_prior",
                "drift_score": float(score),
                "threshold": fail,
                "source_report": prediction_report.get("date"),
            }

    escalate = bool((prediction_report or {}).get("escalate", False))
    fired = bool(worst)
    detail: dict[str, Any] = {
        "worst": worst,
        "decisions": counts,
        "n_signals_checked": columns_seen,
        "thresholds": {"warn": thresholds[0], "fail": fail},
        "escalate": escalate,
        # Recorded, never used to fire by default: escalate is prediction_drift's
        # coupled signal, deliberately kept out of its own `overall`.
        "escalate_ignored": escalate and not escalate_fires,
    }
    if escalate_fires and escalate and not fired:
        detail["escalate_would_fire"] = True
    return SignalEvidence(fired=fired or (escalate and escalate_fires), detail=detail)
