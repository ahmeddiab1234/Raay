"""Retrain trigger: seasonal calendar resolution and the lead window."""

from __future__ import annotations

from datetime import date, timedelta

from retrain_helpers import DAY

from raay.enums.constants import DefaultPaths
from raay.inference.retrain_trigger import (
    SeasonalEvent,
    evaluate_calendar_trigger,
    load_calendar,
    resolve_event_date,
)


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
