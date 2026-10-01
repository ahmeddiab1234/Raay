"""Retrain trigger: the PSI signal (hermetic, in-memory reports)."""

from __future__ import annotations

import pytest
from retrain_helpers import input_report, prediction_report

from raay.inference.retrain_trigger import evaluate_psi_trigger


def test_clean_reports_do_not_fire():
    ev = evaluate_psi_trigger(input_report(), prediction_report())
    assert ev.fired is False
    assert ev.detail["worst"] == {}


def test_input_column_fail_fires():
    report = input_report()
    report["columns"]["positive"] = {"drift_score": 0.31, "decision": "FAIL"}
    ev = evaluate_psi_trigger(report, prediction_report())
    assert ev.fired is True
    assert ev.detail["worst"]["column"] == "positive"
    assert ev.detail["worst"]["source"] == "input_drift"


def test_prediction_output_fail_fires():
    ev = evaluate_psi_trigger(
        input_report(),
        prediction_report(
            output_drift={"overall": "FAIL"},
            class_distribution={
                "psi_vs_training_prior": {"drift_score": 0.47, "decision": "FAIL"}
            },
        ),
    )
    assert ev.fired is True
    assert ev.detail["worst"]["source"] == "prediction_drift"
    assert ev.detail["worst"]["column"] == "class_distribution_vs_training_prior"


def test_warn_does_not_fire():
    report = input_report()
    report["columns"]["positive"] = {"drift_score": 0.15, "decision": "WARN"}
    assert evaluate_psi_trigger(report, prediction_report()).fired is False


def test_skipped_column_does_not_fire():
    report = input_report()
    report["columns"]["oov_rate"] = {"drift_score": None, "decision": "SKIPPED"}
    ev = evaluate_psi_trigger(report, prediction_report())
    assert ev.fired is False
    assert ev.detail["decisions"]["SKIPPED"] == 1


def test_error_column_does_not_fire():
    report = input_report()
    report["columns"]["positive"] = {"drift_score": None, "decision": "ERROR"}
    assert evaluate_psi_trigger(report, prediction_report()).fired is False


def test_worst_column_wins():
    report = input_report()
    report["columns"]["positive"] = {"drift_score": 0.21, "decision": "FAIL"}
    report["columns"]["confidence_score"] = {"drift_score": 0.42, "decision": "FAIL"}
    ev = evaluate_psi_trigger(report, prediction_report())
    assert ev.detail["worst"]["column"] == "confidence_score"
    assert ev.detail["worst"]["drift_score"] == pytest.approx(0.42)


def test_prediction_worse_than_input_wins():
    report = input_report()
    report["columns"]["positive"] = {"drift_score": 0.25, "decision": "FAIL"}
    ev = evaluate_psi_trigger(
        report,
        prediction_report(
            output_drift={"overall": "FAIL"},
            class_distribution={
                "psi_vs_training_prior": {"drift_score": 0.60, "decision": "FAIL"}
            },
        ),
    )
    assert ev.detail["worst"]["source"] == "prediction_drift"


def test_missing_reports_are_no_signal_not_a_breach():
    ev = evaluate_psi_trigger(None, None)
    assert ev.fired is False
    assert ev.detail["n_signals_checked"] == 0


def test_escalate_recorded_but_does_not_fire_by_default():
    ev = evaluate_psi_trigger(input_report(), prediction_report(escalate=True))
    assert ev.fired is False
    assert ev.detail["escalate"] is True
    assert ev.detail["escalate_ignored"] is True


def test_escalate_fires_when_opted_in():
    ev = evaluate_psi_trigger(
        input_report(), prediction_report(escalate=True), escalate_fires=True
    )
    assert ev.fired is True
