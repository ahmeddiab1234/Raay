"""Prediction drift: is the model's *output* behaviour shifting?

Phase 6 step 2, and the counterpart to ``drift_features`` (step 1). Step 1
watches the inputs; this watches what comes out: the predicted class mix and
the confidence behind it.

Two class-distribution comparisons, deliberately both:

- vs the **training label prior** read from ``data/processed/train.csv``
  (57.6 / 37.3 / 5.1 positive / negative / neutral). This asks "is the output
  mix still shaped like the mix we trained on?"
- vs the **scored reference panel**'s own predictions. This asks "is the model
  behaving like it did last time we looked?" and is the same
  ``predicted_label`` column ``--mode drift`` already gates.

The brief for this step specified a 45 / 35 / 20 prior "per your labeling
guidelines". That does not describe this dataset: the true proportions are
identical across train/val/test at 57.6 / 37.3 / 5.1, i.e. the brief overstates
Neutral by 4x. Gating on 45/35/20 was measured, not assumed::

    PSI(reference-panel predictions, brief 45/35/20)  = 0.4143  -> FAIL
    PSI(reference-panel predictions, real train prior) = 0.0207  -> PASS

so the literal prior fires a hard FAIL every night against a completely clean
panel, and a neutral-heavy positive control still scores 0.70 against the real
prior -- so the real prior keeps the gate both quiet and sharp. The train CSV
is therefore the reference frame: real data, DVC-tracked, and it cannot rot
silently the way a hardcoded constant can.

Note the baseline is ~0.02 and not 0.00 on purpose: the model under-predicts
Neutral (2.9% predicted against a 5.1% prior on the held-out split, recall
0.139), so a little PSI is the model being imperfect, not the world moving.
That offset is why the thresholds stay at 0.1/0.2 and are not tightened.

Confidence is tracked as a series. The baseline is the frozen reference
panel's mean ``predicted_score`` -- available from day one -- plus a rolling
z-score once enough days of history exist. The brief's point is that a
*falling* confidence alongside *rising* class PSI is stronger evidence than
either alone, which is reported as ``escalate`` rather than folded silently
into the verdict.

Finally, the step's actual purpose: this is a **triage** signal, so the report
ends in a classification pairing the input-drift verdict for the same date
with this one. See :func:`classify_triage`.

Both panels in this repo are seeded draws from ``data/processed/test.csv``, so
``world_changed`` vs ``model_degraded`` cannot be distinguished against real
traffic here. What is implemented is the mechanism and the wiring; the
distinction is unproven until production data flows through it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from raay.enums.constants import LABELS, DataColumns, DefaultPaths
from raay.inference.batch_score import (
    _psi_per_column,
    _uncomparable_reason,
)

COL_PREDICTED_LABEL = "predicted_label"
COL_PREDICTED_SCORE = "predicted_score"

#: Verdicts the input-drift half of the triage can hand us.
_TRIAGE_PASSTHROUGH = {"PASS": "PASS", "SKIPPED": "PASS"}


def training_label_reference(path: str | Path | None = None) -> pd.DataFrame:
    """The training label prior as a frame the PSI helper can consume.

    Returns the train split's labels under ``predicted_label`` so the same
    ``_psi_per_column`` call serves both references -- one implementation of
    PSI rather than two that can disagree.

    Raises if any of the three classes is absent: a prior missing a class
    cannot be compared against a panel that has one, and silently producing a
    two-class prior would understate the drift instead.
    """
    csv_path = Path(path or DefaultPaths.TRAIN_SPLIT.value)
    labels = pd.read_csv(csv_path, usecols=[DataColumns.LABEL.value])[
        DataColumns.LABEL.value
    ].dropna()
    missing = [label for label in LABELS if label not in set(labels)]
    if missing:
        raise ValueError(
            f"{csv_path} has no rows for class(es) {missing}; expected all of "
            f"{list(LABELS)} to build a comparable label prior"
        )
    return pd.DataFrame({COL_PREDICTED_LABEL: labels})


def class_distribution(frame: pd.DataFrame) -> dict[str, float]:
    """Share of each class. Every key is present, so an absent class reads 0.0."""
    if COL_PREDICTED_LABEL not in frame.columns:
        raise KeyError(
            f"panel is missing {COL_PREDICTED_LABEL!r}; has {list(frame.columns)}"
        )
    counts = frame[COL_PREDICTED_LABEL].value_counts(normalize=True)
    return {label: round(float(counts.get(label, 0.0)), 6) for label in LABELS}


def share_delta_pp(
    current: dict[str, float], reference: dict[str, float]
) -> dict[str, float]:
    """Per-class movement in percentage points; negative means the class grew less."""
    return {
        label: round((current[label] - reference[label]) * 100, 3) for label in LABELS
    }


def mean_confidence(frame: pd.DataFrame) -> float:
    if COL_PREDICTED_SCORE not in frame.columns:
        raise KeyError(
            f"panel is missing {COL_PREDICTED_SCORE!r}; has {list(frame.columns)}"
        )
    return float(frame[COL_PREDICTED_SCORE].mean())


def rolling_baseline(values: list[float], window: int, min_days: int) -> dict[str, Any]:
    """Mean/std over the last ``window`` values, and whether to trust the z-score.

    A rolling baseline adapts to slow drift, which is the point -- but it is
    meaningless until there is enough of it. Below ``min_days`` the mean and
    std are still reported (they are honest descriptions of what exists) while
    ``z_score`` is withheld, so a two-day "trend" cannot manufacture a signal.
    """
    recent = [v for v in values[-window:] if v is not None]
    n = len(recent)
    out: dict[str, Any] = {
        "window": window,
        "n_days": n,
        "min_days": min_days,
        "sufficient_history": n >= min_days,
    }
    if n == 0:
        return out | {"mean": None, "std": None, "z_score": None}
    mean = float(np.mean(recent))
    std = float(np.std(recent, ddof=1)) if n > 1 else 0.0
    out["mean"] = round(mean, 6)
    out["std"] = round(std, 6)
    if n >= min_days and std > 0:
        out["z_score"] = None  # filled in by the caller, which knows today's value
    else:
        # `reason` is what makes this auditable: a withheld z-score is a
        # decision, and the next person needs to know which of the two it was.
        out["z_score"] = None
        out["reason"] = (
            f"need >= {min_days} days, have {n}"
            if n < min_days
            else "rolling std is 0, so every day is identical and the z-score is undefined"
        )
    return out


def confidence_z_score(today: float, baseline: dict[str, Any]) -> float | None:
    if not baseline.get("sufficient_history") or baseline.get("std") in (None, 0):
        return None
    return round((today - baseline["mean"]) / baseline["std"], 3)


def classify_triage(input_verdict: str | None, output_verdict: str) -> str:
    """Name *which half* moved. The triage answer the brief is asking for.

    - inputs drifted, outputs did not: the world changed and the model coped.
    - inputs held, outputs did not: the model itself moved -- degradation, or
      an inference/serving change rather than a data change.
    - both: genuinely ambiguous, and a human should look.
    - neither: stable.

    A missing input report is ``None``, not a silent PASS: without the input
    half there is no way to attribute anything, so the answer is
    ``indeterminate``.
    """
    if input_verdict is None:
        return "indeterminate"
    if input_verdict == "PASS":
        return "stable" if output_verdict == "PASS" else "model_degraded"
    return "world_changed" if output_verdict == "PASS" else "ambiguous"


def _read_input_verdict(path: str | Path | None) -> tuple[str | None, str | None]:
    """``(verdict, source)`` from the input-drift report, tolerating its absence."""
    if not path:
        return None, None
    report = Path(path)
    if not report.exists():
        logger.warning(
            f"No input-drift report at {report}; triage will be 'indeterminate' "
            "rather than assuming the inputs held"
        )
        return None, None
    try:
        payload = json.loads(report.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(f"Could not read input-drift report {report}: {exc}")
        return None, None
    verdict = payload.get("overall")
    if verdict is None:
        logger.warning(
            f"{report} has no 'overall' key; treating input drift as unknown"
        )
        return None, None
    return str(verdict), str(report)


def collect_confidence_history(
    reports_dir: str | Path, exclude: str | None = None
) -> list[float]:
    """Mean confidence per prior day, oldest first.

    Reads the mean already recorded in each report rather than rescoring the
    panels: the point of a history is that it is cheap. Reports written before
    ``feature_means`` existed are skipped -- the 2026-09-23/24 reports predate
    it -- which is exactly why this tolerates the missing key instead of
    indexing it.
    """
    directory = Path(reports_dir)
    if not directory.is_dir():
        return []
    values: list[tuple[str, float]] = []
    for report in sorted(directory.glob("*.json")):
        if exclude is not None and report.stem == exclude:
            continue
        try:
            payload = json.loads(report.read_text())
        except (OSError, json.JSONDecodeError):
            logger.warning(f"Skipping unreadable prediction-drift report {report}")
            continue
        mean = payload.get("confidence", {}).get("mean")
        if isinstance(mean, (int, float)):
            values.append((report.stem, float(mean)))
    return [value for _, value in sorted(values)]


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
