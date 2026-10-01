"""Seasonal-event calendar: loading, date resolution, and the lead-window check.

Lunar events carry a literal ``date`` and ``confirmed: false`` because they are
set by moon sighting; recurring Gregorian events carry a ``rule`` computed here.
An unconfirmed or unresolvable event is reported, never fired.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from raay.inference.retrain_signals import SignalEvidence


@dataclass(frozen=True)
class SeasonalEvent:
    """One calendar event that can justify a retrain.

    Exactly one of ``date`` (a literal ``YYYY-MM-DD``) or ``rule`` (a computed
    Gregorian date) is used. ``confirmed`` is the arming switch: see the module
    docstring -- a lunar date nobody has re-announced must not fire.
    """

    name: str
    confirmed: bool = False
    date: date | None = None
    rule: str | None = None
    lead_days: int | None = None
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "confirmed": self.confirmed,
            "date": self.date.isoformat() if self.date else None,
            "rule": self.rule,
            "lead_days": self.lead_days,
            "note": self.note,
        }


def _resolve_black_friday(year: int) -> date:
    """Black Friday = the Friday after the 4th Thursday in November.

    Deliberately not "the last Friday of November": those differ in most years
    (Nov 2026 gives 27th either way, but 2027 gives 26th vs 27th), and the
    commercial date is fixed by Thanksgiving, not by the calendar's last
    remaining Friday.
    """
    thursdays = [
        date(year, 11, day)
        for day in range(1, 31)
        if date(year, 11, day).weekday() == 3  # Monday=0, so Thursday=3
    ]
    return thursdays[3] + timedelta(days=1)


def _resolve_last_friday_of_november(year: int) -> date:
    """A separate rule kept only so the Black Friday rule is testable against it."""
    fridays = [
        date(year, 11, day)
        for day in range(1, 31)
        if date(year, 11, day).weekday() == 4  # Friday=4
    ]
    return fridays[-1]


#: Computable Gregorian rules. Lunar events are never here -- they carry a
#: literal ``date`` because they are set by moon sighting, not by a calendar.
_RULES = {
    "black_friday": _resolve_black_friday,
    "last_friday_of_november": _resolve_last_friday_of_november,
}


def resolve_event_date(event: SeasonalEvent, year: int) -> date | None:
    """The event's date in ``year``.

    Returns ``None`` for an event with neither a literal date nor a known rule,
    or one whose rule is not implemented -- an unresolvable date is reported
    rather than guessed, because guessing a date is how a phantom retrain
    happens.
    """
    if event.date is not None:
        return event.date.replace(year=year)
    if event.rule is None:
        return None
    resolver = _RULES.get(event.rule)
    if resolver is None:
        return None
    return resolver(year)


def load_calendar(path: str | Path) -> list[SeasonalEvent]:
    """Read ``configs/seasonal_events.yaml``.

    Malformed entries are skipped with a warning rather than raising: a typo in
    one event must not take the nightly job down and suppress every other
    trigger.
    """
    events: list[SeasonalEvent] = []
    raw = yaml.safe_load(Path(path).read_text()) or {}
    for entry in raw.get("events") or []:
        if not isinstance(entry, dict) or not entry.get("name"):
            logger.warning(f"Skipping malformed seasonal event: {entry!r}")
            continue
        literal = entry.get("date")
        parsed: date | None = None
        if literal:
            try:
                parsed = date.fromisoformat(str(literal))
            except ValueError:
                logger.warning(
                    f"Seasonal event {entry['name']!r} has an unparseable date "
                    f"{literal!r}; treating it as unresolvable."
                )
        events.append(
            SeasonalEvent(
                name=str(entry["name"]),
                confirmed=bool(entry.get("confirmed", False)),
                date=parsed,
                rule=entry.get("rule"),
                lead_days=entry.get("lead_days"),
                note=str(entry.get("note", "")),
            )
        )
    return events


def evaluate_calendar_trigger(
    events: list[SeasonalEvent],
    day: date,
    lead_days: int = 21,
) -> SignalEvidence:
    """Is a confirmed seasonal event inside the lead window?

    The window opens *now* rather than on the event date, which is the point:
    the retrain has to be finished and promoted before demand shifts, not
    noticed on the morning it does.
    """
    armed: list[str] = []
    unconfirmed: list[str] = []
    unresolvable: list[str] = []
    hit: dict[str, Any] | None = None
    widest_horizon = day + timedelta(days=lead_days)

    for event in events:
        resolved = resolve_event_date(event, day.year)
        if resolved is None:
            unresolvable.append(event.name)
            continue
        # A per-event lead_days overrides the default. Without this the YAML's
        # `lead_days: 35` on white_friday would be parsed and then silently
        # ignored, arming it on the same 21-day window as everything else.
        event_lead = event.lead_days if event.lead_days is not None else lead_days
        horizon = day + timedelta(days=event_lead)
        widest_horizon = max(widest_horizon, horizon)
        in_window = day <= resolved <= horizon
        if not event.confirmed:
            unconfirmed.append(event.name)
            continue
        if in_window and hit is None:
            hit = {
                "name": event.name,
                "event_date": resolved.isoformat(),
                "days_until": (resolved - day).days,
                "lead_days": event_lead,
                "note": event.note,
            }
        elif in_window:
            armed.append(event.name)

    return SignalEvidence(
        fired=hit is not None,
        detail={
            "event": hit,
            "other_in_window": armed,
            "unconfirmed": unconfirmed,
            "unresolvable": unresolvable,
            "window": {
                "from": day.isoformat(),
                "to": widest_horizon.isoformat(),
                "lead_days": lead_days,
            },
        },
    )
