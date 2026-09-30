"""Phase 6 step 3: the retrain trigger's decision logic (hermetic).

Every test drives the real entry points against in-memory dicts or tiny YAML
files. No drift reports are read from ``reports/``, no model is loaded, and no
network is touched -- the module deliberately keeps all I/O in ``main()`` so
the decision surface stays a pure function.
"""

from __future__ import annotations

import io
import json
import urllib.error
from datetime import date, timedelta
from typing import Self

import pytest

import raay.inference.retrain_trigger as rt
from raay.enums.constants import DefaultPaths
from raay.inference.retrain_trigger import (
    SeasonalEvent,
    SignalEvidence,
    build_report,
    decide,
    dispatch_retrain,
    evaluate_calendar_trigger,
    evaluate_psi_trigger,
    evaluate_trigger,
    load_calendar,
    mlflow_metrics,
    mlflow_tags,
    resolve_event_date,
    resolve_token,
)

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


# --------------------------------------------------------------------------
# PSI signal
# --------------------------------------------------------------------------


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


# --------------------------------------------------------------------------
# Calendar signal
# --------------------------------------------------------------------------


def test_black_friday_rule_is_not_last_friday():
    black_friday = SeasonalEvent(name="bf", rule="black_friday")
    last_friday = SeasonalEvent(name="lf", rule="last_friday_of_november")
    assert resolve_event_date(black_friday, 2026) == date(2026, 11, 27)
    # In 2027 the two rules land on different days, which is the point.
    assert resolve_event_date(black_friday, 2027) == date(2027, 11, 26)
    assert resolve_event_date(last_friday, 2027) == date(2027, 11, 26)
    assert resolve_event_date(last_friday, 2026) == date(2026, 11, 27)


def test_confirmed_event_in_window_fires():
    events = [SeasonalEvent(name="bf", rule="black_friday", confirmed=True)]
    ev = evaluate_calendar_trigger(events, date(2026, 11, 10))
    assert ev.fired is True
    assert ev.detail["event"]["name"] == "bf"
    assert ev.detail["event"]["days_until"] == 17


def test_confirmed_event_outside_window_does_not_fire():
    events = [SeasonalEvent(name="bf", rule="black_friday", confirmed=True)]
    ev = evaluate_calendar_trigger(events, date(2026, 9, 30))
    assert ev.fired is False


def test_per_event_lead_days_overrides_the_default():
    black_friday = date(2026, 11, 27)
    # 40 days out: outside the default 21-day window, inside a 45-day one.
    day = black_friday - timedelta(days=40)
    wide = SeasonalEvent(
        name="white_friday", rule="black_friday", confirmed=True, lead_days=45
    )
    narrow = SeasonalEvent(name="black_friday", rule="black_friday", confirmed=True)
    assert evaluate_calendar_trigger([wide], day, lead_days=21).fired is True
    assert evaluate_calendar_trigger([narrow], day, lead_days=21).fired is False


def test_unconfirmed_event_never_fires():
    events = [SeasonalEvent(name="ramadan", date=date(2026, 10, 5), confirmed=False)]
    ev = evaluate_calendar_trigger(events, date(2026, 9, 30))
    assert ev.fired is False
    assert "ramadan" in ev.detail["unconfirmed"]


def test_unresolvable_event_is_reported_not_guessed():
    events = [SeasonalEvent(name="mystery", rule="not_a_rule", confirmed=True)]
    ev = evaluate_calendar_trigger(events, DAY)
    assert ev.fired is False
    assert "mystery" in ev.detail["unresolvable"]


def test_load_calendar_skips_malformed_entries(tmp_path):
    path = tmp_path / "cal.yaml"
    path.write_text(
        "events:\n"
        "  - name: good\n    rule: black_friday\n    confirmed: true\n"
        "  - rule: black_friday\n"  # no name -> skipped
    )
    events = load_calendar(path)
    assert [e.name for e in events] == ["good"]


def test_shipped_calendar_lunar_events_are_unconfirmed():
    events = load_calendar(DefaultPaths.CONFIG_SEASONAL_EVENTS.value)
    by_name = {e.name: e for e in events}
    assert by_name["black_friday"].confirmed is True
    assert by_name["ramadan"].confirmed is False


# --------------------------------------------------------------------------
# Precedence
# --------------------------------------------------------------------------


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


# --------------------------------------------------------------------------
# End-to-end + reports + tokens
# --------------------------------------------------------------------------


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


def test_resolve_token_reads_file(tmp_path):
    path = tmp_path / "tok"
    path.write_text("ghp_secret\n")
    assert resolve_token(str(path)) == "ghp_secret"


def test_resolve_token_absent_returns_none(monkeypatch):
    monkeypatch.delenv("RAAY_GITHUB_DISPATCH_TOKEN", raising=False)
    monkeypatch.delenv("RAAY_GITHUB_DISPATCH_TOKEN_FILE", raising=False)
    assert resolve_token() is None


def test_resolve_token_missing_file_is_none_not_an_error(tmp_path):
    missing = tmp_path / "nope"
    assert resolve_token(str(missing)) is None


class _FakeResponse:
    def __init__(self, status: int) -> None:
        self.status = status

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def test_dispatch_posts_the_event_with_a_bearer_header():
    captured: dict = {}

    def fake_opener(request, timeout=None):
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["headers"] = {k.lower(): v for k, v in request.header_items()}
        captured["body"] = json.loads(request.data)
        return _FakeResponse(204)

    result = dispatch_retrain(
        {"reason": "psi_breach", "trigger_date": "2026-10-01"},
        "github_pat_secret",
        repository="ahmeddiab1234/Raay",
        opener=fake_opener,
    )

    assert result["ok"] is True
    assert result["status"] == 204
    assert captured["method"] == "POST"
    assert captured["url"] == (
        "https://api.github.com/repos/ahmeddiab1234/Raay/dispatches"
    )
    assert captured["headers"]["authorization"] == "Bearer github_pat_secret"
    assert captured["body"]["event_type"] == "retrain"
    assert captured["body"]["client_payload"]["reason"] == "psi_breach"


def test_dispatch_reports_http_error_body_and_does_not_raise():
    def fake_opener(request, timeout=None):
        raise urllib.error.HTTPError(
            request.full_url,
            401,
            "Unauthorized",
            {},
            io.BytesIO(b'{"message":"Bad credentials"}'),
        )

    result = dispatch_retrain({}, "bad", opener=fake_opener)

    assert result["attempted"] is True
    assert result["ok"] is False
    assert result["status"] == 401
    assert "Bad credentials" in result["error"]


def test_dispatch_reports_network_error_and_does_not_raise():
    def fake_opener(request, timeout=None):
        raise urllib.error.URLError("name resolution failed")

    result = dispatch_retrain({}, "tok", opener=fake_opener)

    assert result["ok"] is False
    assert "name resolution failed" in result["error"]


def test_main_skips_dispatch_without_token(tmp_path, monkeypatch):
    monkeypatch.delenv("RAAY_GITHUB_DISPATCH_TOKEN", raising=False)
    monkeypatch.delenv("RAAY_GITHUB_DISPATCH_TOKEN_FILE", raising=False)
    inp = tmp_path / "in.json"
    pred = tmp_path / "pred.json"
    inp.write_text(json.dumps(breach_input_report()))
    pred.write_text(json.dumps({"triage": "world_changed"}))
    out = tmp_path / "trigger.json"

    code = rt.main(
        [
            "--date",
            "2026-10-01",
            "--input-drift-report",
            str(inp),
            "--prediction-drift-report",
            str(pred),
            "--report-out",
            str(out),
            "--no-mlflow",
        ]
    )

    report = json.loads(out.read_text())
    assert report["triggered"] is True
    assert report["dispatch"]["attempted"] is False
    assert "no dispatch token" in report["dispatch"]["reason_not_attempted"]
    assert code == 0


def test_main_returns_nonzero_when_a_dispatch_attempt_fails(tmp_path, monkeypatch):
    tok = tmp_path / "tok"
    tok.write_text("github_pat_secret\n")
    inp = tmp_path / "in.json"
    pred = tmp_path / "pred.json"
    inp.write_text(json.dumps(breach_input_report()))
    pred.write_text(json.dumps({"triage": "world_changed"}))
    out = tmp_path / "trigger.json"
    monkeypatch.setattr(
        rt,
        "dispatch_retrain",
        lambda *a, **k: {"attempted": True, "ok": False, "error": "HTTP 403"},
    )

    code = rt.main(
        [
            "--date",
            "2026-10-01",
            "--input-drift-report",
            str(inp),
            "--prediction-drift-report",
            str(pred),
            "--report-out",
            str(out),
            "--token-file",
            str(tok),
            "--no-mlflow",
        ]
    )

    assert code == 1
    assert json.loads(out.read_text())["dispatch"]["ok"] is False


def test_main_no_dispatch_flag_never_reads_the_token(tmp_path, monkeypatch):
    inp = tmp_path / "in.json"
    pred = tmp_path / "pred.json"
    inp.write_text(json.dumps(breach_input_report()))
    pred.write_text(json.dumps({"triage": "world_changed"}))
    out = tmp_path / "trigger.json"
    monkeypatch.setattr(
        rt, "resolve_token", lambda *a, **k: pytest.fail("token must not be read")
    )

    code = rt.main(
        [
            "--date",
            "2026-10-01",
            "--input-drift-report",
            str(inp),
            "--prediction-drift-report",
            str(pred),
            "--report-out",
            str(out),
            "--no-dispatch",
            "--no-mlflow",
        ]
    )

    assert code == 0
