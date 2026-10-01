"""Prediction-drift verdict: reference-prior decision, escalation, triage, MLflow."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from prediction_drift_helpers import BRIEF_PRIOR, REAL_PRIOR, panel, spec, train_csv

from raay.inference.prediction_drift import (
    classify_triage,
    mlflow_metrics,
    predict_drift_check,
)


def test_clean_panel_passes_against_the_real_prior(tmp_path: Path) -> None:
    """A panel shaped like the model actually predicts must PASS.

    Guards the measured ~0.02 baseline: the model under-predicts Neutral, so
    a little PSI is expected, and the threshold must stay above it.
    """
    # What the model really emits on a clean panel (59.5/37.6/2.9), not the
    # ideal prior -- using the ideal here would not reproduce the floor.
    clean = panel({"positive": 0.595, "negative": 0.376, "neutral": 0.029})
    report = predict_drift_check(spec(tmp_path, clean))
    cd = report["class_distribution"]
    assert cd["psi_vs_training_prior"]["decision"] == "PASS"
    assert cd["psi_vs_training_prior"]["drift_score"] < 0.1
    assert report["output_drift"]["overall"] == "PASS"


def test_brief_prior_45_35_20_would_fail_a_clean_panel(tmp_path: Path) -> None:
    """Documents *why* the brief's prior is not used.

    If a future edit reintroduces 45/35/20 as the reference, this pins the
    consequence: the same clean panel that PASSes above reads 0.41 and fails,
    every night, with nothing actually wrong.
    """
    clean = panel({"positive": 0.595, "negative": 0.376, "neutral": 0.029})
    result = predict_drift_check(
        spec(
            tmp_path,
            clean,
            train_prior_csv=train_csv(tmp_path, BRIEF_PRIOR, name="train_brief"),
        )
    )
    score = result["class_distribution"]["psi_vs_training_prior"]["drift_score"]
    assert score >= 0.2
    assert result["output_drift"]["overall"] == "FAIL"


def test_neutral_heavy_panel_fails_against_the_real_prior(tmp_path: Path) -> None:
    """The gate must still have teeth with the corrected prior."""
    shifted = panel({"positive": 0.35, "negative": 0.30, "neutral": 0.35})
    report = predict_drift_check(spec(tmp_path, shifted))
    assert report["class_distribution"]["psi_vs_training_prior"]["decision"] == "FAIL"


def test_reference_panel_comparison_is_reported_alongside(tmp_path: Path) -> None:
    clean = panel({"positive": 0.595, "negative": 0.376, "neutral": 0.029})
    report = predict_drift_check(spec(tmp_path, clean))
    cd = report["class_distribution"]
    assert cd["psi_vs_reference_panel"]["decision"] == "PASS"
    assert cd["training_prior_source"].endswith("train.csv")


def test_confidence_falling_is_flagged_and_escalation_needs_both(
    tmp_path: Path,
) -> None:
    """`escalate` is a conjunction: a falling mean alone is not escalation."""
    lower = panel(REAL_PRIOR, score=0.70)
    report = predict_drift_check(spec(tmp_path, lower))
    assert report["confidence"]["falling"] is True
    assert report["escalate"] is False  # class PSI is still PASS

    shifted = panel({"positive": 0.35, "negative": 0.30, "neutral": 0.35}, score=0.70)
    report = predict_drift_check(spec(tmp_path, shifted))
    assert report["confidence"]["falling"] is True
    assert report["escalate"] is True  # both halves present


@pytest.mark.parametrize(
    ("input_verdict", "output_verdict", "expected"),
    [
        ("PASS", "PASS", "stable"),
        ("FAIL", "PASS", "world_changed"),
        ("PASS", "FAIL", "model_degraded"),
        ("FAIL", "FAIL", "ambiguous"),
        ("WARN", "PASS", "world_changed"),
        ("PASS", "WARN", "model_degraded"),
        (None, "PASS", "indeterminate"),
        (None, "FAIL", "indeterminate"),
    ],
)
def test_classify_triage_matrix(
    input_verdict: str | None, output_verdict: str, expected: str
) -> None:
    assert classify_triage(input_verdict, output_verdict) == expected


def test_triage_pairs_with_the_same_days_input_drift_report(
    tmp_path: Path,
) -> None:
    drift_report = tmp_path / "reports" / "drift" / "2026-01-01.json"
    drift_report.parent.mkdir(parents=True)
    drift_report.write_text(json.dumps({"overall": "FAIL"}))
    clean = panel({"positive": 0.595, "negative": 0.376, "neutral": 0.029})
    report = predict_drift_check(
        spec(tmp_path, clean, input_drift_report=str(drift_report))
    )
    assert report["input_drift"]["overall"] == "FAIL"
    assert report["triage"] == "world_changed"


def test_missing_input_report_is_indeterminate_not_a_silent_pass(
    tmp_path: Path,
) -> None:
    """Attribution needs both halves; a missing half must not read as PASS."""
    clean = panel({"positive": 0.595, "negative": 0.376, "neutral": 0.029})
    report = predict_drift_check(
        spec(tmp_path, clean, input_drift_report=str(tmp_path / "nope.json"))
    )
    assert report["input_drift"]["overall"] is None
    assert report["triage"] == "indeterminate"


def test_unreadable_input_report_is_indeterminate(tmp_path: Path) -> None:
    bad = tmp_path / "drift.json"
    bad.write_text("{oops")
    clean = panel({"positive": 0.595, "negative": 0.376, "neutral": 0.029})
    report = predict_drift_check(spec(tmp_path, clean, input_drift_report=str(bad)))
    assert report["triage"] == "indeterminate"


def test_report_carries_the_caveat(tmp_path: Path) -> None:
    clean = panel(REAL_PRIOR)
    report = predict_drift_check(spec(tmp_path, clean))
    assert "unverified against production traffic" in report["caveat"]


def test_mlflow_metrics_skip_none_rather_than_logging_zero(
    tmp_path: Path,
) -> None:
    """A cold-start z-score must not chart as a healthy 0.0."""
    clean = panel(REAL_PRIOR)
    report = predict_drift_check(spec(tmp_path, clean))
    metrics = mlflow_metrics(report)
    assert "pred_confidence_z_score" not in metrics
    assert metrics["pred_mean_confidence"] == pytest.approx(0.9)


def test_mlflow_metrics_include_class_shares(tmp_path: Path) -> None:
    clean = panel({"positive": 0.595, "negative": 0.376, "neutral": 0.029})
    metrics = mlflow_metrics(predict_drift_check(spec(tmp_path, clean)))
    assert metrics["pred_class_share_neutral"] == pytest.approx(0.029, abs=1e-3)
    assert metrics["pred_psi_vs_train_prior"] > 0
