"""Retrain trigger: reason precedence, end-to-end decision, report, MLflow."""

from __future__ import annotations

from retrain_helpers import DAY, input_report, prediction_report

from raay.inference.retrain_trigger import (
    SignalEvidence,
    build_report,
    decide,
    evaluate_trigger,
    mlflow_metrics,
    mlflow_tags,
)


def test_psi_beats_calendar():
    psi = SignalEvidence(fired=True, detail={"worst": {}})
    calendar = SignalEvidence(fired=True, detail={"event": {}})
    d = decide(psi, calendar, DAY)
    assert d.reason == "psi_breach"


def test_manual_beats_everything():
    d = decide(SignalEvidence(fired=True), SignalEvidence(fired=True), DAY, "manual")
    assert d.reason == "manual"
    assert d.forced is True


def test_no_signal_is_none():
    d = decide(SignalEvidence(fired=False), SignalEvidence(fired=False), DAY)
    assert d.reason == "none"
    assert d.triggered is False


def test_calendar_only_is_scheduled():
    d = decide(SignalEvidence(fired=False), SignalEvidence(fired=True), DAY)
    assert d.reason == "scheduled"


def test_end_to_end_clean_day_is_none():
    d = evaluate_trigger(DAY, input_report(), prediction_report(), events=[])
    assert d.triggered is False
    assert d.reason == "none"


def test_end_to_end_breach_is_psi_breach():
    report = input_report()
    report["columns"]["positive"] = {"drift_score": 0.44, "decision": "FAIL"}
    d = evaluate_trigger(DAY, report, prediction_report(), events=[])
    assert d.reason == "psi_breach"


def test_mlflow_metrics_omit_missing_worst_score():
    d = decide(SignalEvidence(fired=False), SignalEvidence(fired=False), DAY)
    metrics = mlflow_metrics(d)
    assert "retrain_worst_psi" not in metrics
    assert metrics["retrain_triggered"] == 0.0


def test_mlflow_tags_carry_the_reason():
    report = input_report()
    report["columns"]["positive"] = {"drift_score": 0.44, "decision": "FAIL"}
    d = evaluate_trigger(DAY, report, prediction_report(), events=[])
    tags = mlflow_tags(d)
    assert tags["trigger_reason"] == "psi_breach"
    assert tags["psi_worst_column"] == "positive"


def test_report_has_dispatch_and_handoff():
    d = decide(SignalEvidence(fired=False), SignalEvidence(fired=False), DAY)
    report = build_report(d, "in.json", "pred.json")
    assert report["dispatch"]["attempted"] is False
    assert "kaggle_train_runs" in report["handoff"]
