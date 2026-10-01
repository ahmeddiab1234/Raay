"""The prediction-drift verdict: PSI of the mix, confidence series, triage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from raay.inference.drift_psi import _psi_per_column, _uncomparable_reason
from raay.inference.prediction_series import (
    COL_PREDICTED_LABEL,
    class_distribution,
    collect_confidence_history,
    confidence_z_score,
    mean_confidence,
    rolling_baseline,
    share_delta_pp,
    training_label_reference,
)
from raay.inference.prediction_triage import _read_input_verdict, classify_triage


@dataclass
class PredictionDriftSpec:
    """Everything ``predict_drift_check`` reads. All paths injectable for tests."""

    current_csv: str
    reference_csv: str
    train_prior_csv: str
    input_drift_report: str | None = None
    history_dir: str | None = None
    day: str | None = None
    thresholds: tuple[float, float] = (0.1, 0.2)
    window: int = 14
    min_days: int = 7


def _verdict(score: float | None, thresholds: tuple[float, float]) -> str:
    warn, fail = thresholds
    if score is None:
        return "ERROR"
    if score >= fail:
        return "FAIL"
    if score >= warn:
        return "WARN"
    return "PASS"


def _psi_against(
    reference: pd.DataFrame, current: pd.DataFrame, thresholds: tuple[float, float]
) -> dict[str, Any]:
    """PSI of one panel's predicted mix against a reference frame.

    Reuses ``_psi_per_column`` so this step cannot drift numerically from step
    1 -- same helper, same thresholds, same SKIPPED/ERROR handling.
    """
    if COL_PREDICTED_LABEL not in reference.columns:
        return {
            "drift_score": None,
            "decision": "ERROR",
            "reason": (
                f"reference is missing {COL_PREDICTED_LABEL!r}; has "
                f"{list(reference.columns)}"
            ),
        }
    degenerate = _uncomparable_reason(reference[COL_PREDICTED_LABEL])
    if degenerate is not None:
        return {
            "drift_score": None,
            "decision": "SKIPPED",
            "reason": degenerate["reason"],
            "reference_std": degenerate["reference_std"],
        }
    # No try/except: _psi_per_column already turns a failing column into
    # decision ERROR and returns it, so wrapping it again would be dead code.
    verdict = _psi_per_column([COL_PREDICTED_LABEL], reference, current, thresholds)[0][
        COL_PREDICTED_LABEL
    ]
    return {
        "drift_score": verdict["drift_score"],
        "decision": verdict["decision"],
        "stattest": verdict.get("stattest", "PSI"),
        "stattest_threshold": verdict.get("stattest_threshold"),
    }


def predict_drift_check(spec: PredictionDriftSpec) -> dict[str, Any]:
    """Build ``reports/prediction_drift/{date}.json``."""
    current = pd.read_csv(spec.current_csv)
    reference = pd.read_csv(spec.reference_csv)
    prior = training_label_reference(spec.train_prior_csv)

    current_mix = class_distribution(current)
    reference_mix = class_distribution(reference)
    prior_mix = class_distribution(prior)

    psi_prior = _psi_against(prior, current, spec.thresholds)
    psi_reference = _psi_against(reference, current, spec.thresholds)

    # The headline signal is the training-prior comparison; the reference-panel
    # one is supporting context. Taking the worse of the two would let the
    # better-calibrated measure drag a genuine shift down.
    overall = psi_prior["decision"]

    today_confidence = mean_confidence(current)
    reference_confidence = mean_confidence(reference)
    history = (
        collect_confidence_history(spec.history_dir, exclude=spec.day)
        if spec.history_dir
        else []
    )
    baseline = rolling_baseline(history, spec.window, spec.min_days)
    z = confidence_z_score(today_confidence, baseline)

    delta = round(today_confidence - reference_confidence, 6)
    confidence_falling = delta < 0
    class_psi_rising = psi_prior["decision"] in ("WARN", "FAIL")

    input_verdict, input_source = _read_input_verdict(spec.input_drift_report)

    return {
        "date": spec.day,
        "n_current": len(current),
        "n_reference": len(reference),
        "thresholds": {"warn": spec.thresholds[0], "fail": spec.thresholds[1]},
        "class_distribution": {
            "current": current_mix,
            "reference_panel": reference_mix,
            "training_prior": prior_mix,
            "training_prior_source": spec.train_prior_csv,
            "share_delta_pp_vs_prior": share_delta_pp(current_mix, prior_mix),
            "share_delta_pp_vs_reference": share_delta_pp(current_mix, reference_mix),
            "psi_vs_training_prior": psi_prior,
            "psi_vs_reference_panel": psi_reference,
        },
        "confidence": {
            "mean": round(today_confidence, 6),
            "reference_mean": round(reference_confidence, 6),
            "delta_vs_reference": delta,
            "falling": confidence_falling,
            "z_score": z,
            "rolling": baseline,
            "history_n_days": len(history),
        },
        "output_drift": {
            "overall": overall,
            "basis": "class distribution vs the training label prior",
        },
        "input_drift": {
            "overall": input_verdict,
            "source": input_source,
        },
        "triage": classify_triage(input_verdict, overall),
        # The brief's coupled signal, kept as its own field: it is the reason
        # to look, not a verdict, and folding it into `overall` would make the
        # two indistinguishable in the report.
        "escalate": bool(confidence_falling and class_psi_rising),
        "caveat": (
            "Both panels are seeded draws from data/processed/test.csv, so the "
            "input/output attribution is unverified against production traffic."
        ),
    }


def mlflow_metrics(report: dict[str, Any]) -> dict[str, float]:
    """Metrics for the ``raay_batch`` experiment, skipping anything None.

    A None is a check that did not run (cold-start z-score, SKIPPED PSI);
    logging 0.0 for it would draw a healthy line for a measurement nobody took.
    """
    cd = report["class_distribution"]
    confidence = report["confidence"]
    raw: dict[str, Any] = {
        "pred_class_share_positive": cd["current"]["positive"],
        "pred_class_share_negative": cd["current"]["negative"],
        "pred_class_share_neutral": cd["current"]["neutral"],
        "pred_psi_vs_train_prior": cd["psi_vs_training_prior"]["drift_score"],
        "pred_psi_vs_reference_panel": cd["psi_vs_reference_panel"]["drift_score"],
        "pred_mean_confidence": confidence["mean"],
        "pred_confidence_delta_vs_reference": confidence["delta_vs_reference"],
        "pred_confidence_z_score": confidence["z_score"],
        "pred_confidence_history_days": float(confidence["history_n_days"]),
        "pred_escalate": float(report["escalate"]),
    }
    return {key: float(value) for key, value in raw.items() if value is not None}
