"""Shared in-memory report builders for the retrain-trigger tests."""

from __future__ import annotations

from datetime import date

DAY = date(2026, 10, 1)


def input_report(**overrides) -> dict:
    base = {
        "date": "2026-10-01",
        "thresholds": {"warn": 0.1, "fail": 0.2},
        "columns": {
            "predicted_label": {"drift_score": 0.002, "decision": "PASS"},
            "positive": {"drift_score": 0.011, "decision": "PASS"},
            "confidence_score": {"drift_score": 0.021, "decision": "PASS"},
        },
        "overall": "PASS",
    }
    base.update(overrides)
    return base


def breach_input_report() -> dict:
    report = input_report()
    report["columns"]["positive"] = {"drift_score": 0.44, "decision": "FAIL"}
    return report


def prediction_report(**overrides) -> dict:
    base = {
        "date": "2026-10-01",
        "class_distribution": {
            "psi_vs_training_prior": {"drift_score": 0.021, "decision": "PASS"}
        },
        "output_drift": {"overall": "PASS"},
        "triage": "stable",
        "escalate": False,
    }
    base.update(overrides)
    return base
