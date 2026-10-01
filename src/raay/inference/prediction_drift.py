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

Implementation lives in three siblings -- ``prediction_series`` (mix/confidence
summaries), ``prediction_triage`` (input/output pairing) and
``prediction_report`` (the verdict) -- re-exported here so callers keep the one
import path.
"""

from __future__ import annotations

from raay.enums.constants import LABELS
from raay.inference.prediction_report import (
    PredictionDriftSpec,
    mlflow_metrics,
    predict_drift_check,
)
from raay.inference.prediction_series import (
    class_distribution,
    collect_confidence_history,
    confidence_z_score,
    mean_confidence,
    rolling_baseline,
    share_delta_pp,
    training_label_reference,
)
from raay.inference.prediction_triage import classify_triage

__all__ = [
    "LABELS",
    "PredictionDriftSpec",
    "class_distribution",
    "classify_triage",
    "collect_confidence_history",
    "confidence_z_score",
    "mean_confidence",
    "mlflow_metrics",
    "predict_drift_check",
    "rolling_baseline",
    "share_delta_pp",
    "training_label_reference",
]
