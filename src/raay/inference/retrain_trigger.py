"""Should tonight's batch trigger a retrain? (Phase 6 step 3)

Steps 1 and 2 watch the world and the model. Neither of them *acts*: both are
report-only and exit 0 (see ``batch_score --mode drift`` / ``--mode
predict-drift``). This step is the seam they were pointing at -- the nightly
job now asks a question and records the answer, so "the gate fired and nobody
looked" stops being possible.

The decision is deliberately small and boring, because the cost of being wrong
is asymmetric: a needless retrain burns a Kaggle GPU sweep and a human
promotion, while a missed one is caught by the next night's gate.

    manual        an operator asked for it; bypasses detection entirely
    psi_breach    a gated column PSI >= 0.2 in either drift report
    scheduled     a confirmed seasonal event falls in the lead window
    none          nothing fired

Three deliberate non-firers, each of which would otherwise be a false alarm:

- **WARN does not fire.** The brief puts action at 0.2-0.25; WARN (0.1-0.2) is
  a heads-up, not a retrain. A model sitting permanently at ~0.02 (its Neutral
  weakness) must never reach a trigger.
- **SKIPPED and ERROR columns do not fire.** ``oov_rate`` is structurally 0.0
  on this corpus and reports SKIPPED with a reason; treating a null score as a
  breach would retrain nightly over a check that never ran.
- **Unconfirmed calendar dates do not fire.** Ramadan and Eid dates are set by
  moon sighting. A stale or guessed date that fired would be a retrain nobody
  asked for, so an event must be explicitly ``confirmed: true`` to arm.

The one signal deliberately *excluded* from firing is ``escalate`` (falling
confidence AND rising class PSI), which ``prediction_drift`` reports as its own
field. It is stronger evidence than either half alone, but it is a coupled
signal deliberately kept out of ``overall``; letting it trigger on its own would
invent a threshold below the agreed 0.2. It is recorded as evidence and can be
opted into with ``--escalate-fires``.

**This step does not retrain anything.** Fine-tuning needs a GPU and happens in
``scripts/kaggle_train_runs.py`` on Kaggle; ``promote_model.py`` is the only
code allowed to move the Production alias. What happens here is a decision, a
report, and MLflow provenance.

The seasonal trigger is the one part that cannot be validated. Nothing in this
corpus carries a timestamp, so no retrain this job triggers can ever be checked
after the fact for whether firing before Ramadan helped. It is calendar-only and
is documented as untested rather than presented as a feedback loop.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from raay.enums.constants import DefaultPaths

#: GitHub REST endpoint for a repository_dispatch. The token goes in an
#: Authorization header, never argv (same discipline as deploy_staging.py).
_GITHUB_API = "https://api.github.com"

#: Default receiver. This is the *git remote* slug, which is what dispatch
#: needs -- not the container-registry name the CD workflow publishes under.
_DEFAULT_REPOSITORY = "ahmeddiab1234/Raay"

#: The reasons a retrain can be triggered, in the order they are evaluated.
#: Exposed as a tuple (not an Enum) because the value is written straight into
#: the report and onto an MLflow run tag, where a plain string reads better.
TRIGGER_REASONS: tuple[str, ...] = ("manual", "psi_breach", "scheduled", "none")

_REASON_MANUAL = "manual"
_REASON_PSI = "psi_breach"
_REASON_SEASONAL = "scheduled"
_REASON_NONE = "none"

#: Report decisions that cannot fire a retrain, with the worst one still
#: recorded for the evidence block so the reason a column was ignored is
#: visible rather than implied by its absence.
_NON_FIRING_DECISIONS: tuple[str, ...] = ("PASS", "WARN", "SKIPPED", "ERROR")


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


@dataclass
class SignalEvidence:
    """Why one signal did or did not fire."""

    fired: bool
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"fired": self.fired, **self.detail}


def evaluate_psi_trigger(
    input_report: dict[str, Any] | None,
    prediction_report: dict[str, Any] | None,
    thresholds: tuple[float, float] = (0.1, 0.2),
    escalate_fires: bool = False,
) -> SignalEvidence:
    """Has anything crossed the FAIL threshold in either drift report?

    Fires on the *worst* observed column, and only ever on an explicit FAIL:
    ``WARN`` sits between the thresholds by construction, so re-deriving it
    here would fire on a band this function does not claim.
    """
    _, fail = thresholds
    worst_score: float | None = None
    worst: dict[str, Any] = {}
    counts = {"FAIL": 0, "WARN": 0, "SKIPPED": 0, "ERROR": 0}
    columns_seen = 0

    if input_report:
        for name, column in (input_report.get("columns") or {}).items():
            decision = str(column.get("decision", "ERROR")).upper()
            counts[decision] = counts.get(decision, 0) + 1
            columns_seen += 1
            score = column.get("drift_score")
            if (
                decision == "FAIL"
                and score is not None
                and score > (worst_score if worst_score is not None else -1.0)
            ):
                worst_score = float(score)
                worst = {
                    "source": "input_drift",
                    "column": name,
                    "drift_score": float(score),
                    "threshold": fail,
                    "source_report": input_report.get("date"),
                }

    # The output side carries no per-column map, just the verdict its class-mix
    # PSI produced, so the comparison name is recorded instead of a column.
    if prediction_report:
        output = prediction_report.get("output_drift") or {}
        decision = str(output.get("overall", "ERROR")).upper()
        counts[decision] = counts.get(decision, 0) + 1
        columns_seen += 1
        psi_block = (prediction_report.get("class_distribution") or {}).get(
            "psi_vs_training_prior"
        ) or {}
        score = psi_block.get("drift_score")
        if (
            decision == "FAIL"
            and score is not None
            and score > (worst_score if worst_score is not None else -1.0)
        ):
            worst_score = float(score)
            worst = {
                "source": "prediction_drift",
                "column": "class_distribution_vs_training_prior",
                "drift_score": float(score),
                "threshold": fail,
                "source_report": prediction_report.get("date"),
            }

    escalate = bool((prediction_report or {}).get("escalate", False))
    fired = bool(worst)
    detail: dict[str, Any] = {
        "worst": worst,
        "decisions": counts,
        "n_signals_checked": columns_seen,
        "thresholds": {"warn": thresholds[0], "fail": fail},
        "escalate": escalate,
        # Recorded, never used to fire by default: escalate is prediction_drift's
        # coupled signal, deliberately kept out of its own `overall`.
        "escalate_ignored": escalate and not escalate_fires,
    }
    if escalate_fires and escalate and not fired:
        detail["escalate_would_fire"] = True
    return SignalEvidence(fired=fired or (escalate and escalate_fires), detail=detail)


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


@dataclass(frozen=True)
class RetrainDecision:
    """The verdict for one night, and the evidence behind it."""

    triggered: bool
    reason: str
    psi: SignalEvidence
    calendar: SignalEvidence
    day: date
    forced: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "date": self.day.isoformat(),
            "triggered": self.triggered,
            "reason": self.reason,
            "forced": self.forced,
            "psi": self.psi.as_dict(),
            "calendar": self.calendar.as_dict(),
        }


def decide(
    psi: SignalEvidence,
    calendar: SignalEvidence,
    day: date,
    force_reason: str | None = None,
) -> RetrainDecision:
    """Apply the precedence: manual > psi_breach > scheduled > none.

    ``force_reason`` is the manual path. It bypasses detection but still
    reports both signals' evidence, so an operator-forced retrain is auditable
    against what the gates actually said that night.
    """
    if force_reason == _REASON_MANUAL:
        return RetrainDecision(
            triggered=True,
            reason=_REASON_MANUAL,
            psi=psi,
            calendar=calendar,
            day=day,
            forced=True,
        )
    if psi.fired:
        return RetrainDecision(
            triggered=True, reason=_REASON_PSI, psi=psi, calendar=calendar, day=day
        )
    if calendar.fired:
        return RetrainDecision(
            triggered=True,
            reason=_REASON_SEASONAL,
            psi=psi,
            calendar=calendar,
            day=day,
        )
    return RetrainDecision(
        triggered=False, reason=_REASON_NONE, psi=psi, calendar=calendar, day=day
    )


def _read_json(path: str | Path | None) -> dict[str, Any] | None:
    """Read a report, tolerating absence.

    A missing report is "no signal", never a breach: the trigger must not fire
    because a report failed to be written.
    """
    if path is None:
        return None
    file = Path(path)
    if not file.exists():
        logger.warning(f"No report at {file}; treating as no signal.")
        return None
    try:
        return json.loads(file.read_text())
    except json.JSONDecodeError as error:
        logger.warning(f"Report {file} is not valid JSON ({error}); no signal.")
        return None


def evaluate_trigger(
    day: date,
    input_report: dict[str, Any] | None,
    prediction_report: dict[str, Any] | None,
    events: list[SeasonalEvent],
    thresholds: tuple[float, float] = (0.1, 0.2),
    lead_days: int = 21,
    escalate_fires: bool = False,
    force_reason: str | None = None,
) -> RetrainDecision:
    """The whole decision, from parsed reports to a verdict."""
    psi = evaluate_psi_trigger(
        input_report, prediction_report, thresholds, escalate_fires
    )
    calendar = evaluate_calendar_trigger(events, day, lead_days)
    return decide(psi, calendar, day, force_reason)


def mlflow_metrics(decision: RetrainDecision) -> dict[str, float]:
    """Metrics for the ``raay_batch`` run.

    Only real numbers are logged. A missing worst-column score would otherwise
    be logged as 0.0 and draw a healthy line on the retrain chart for a night
    that never breached anything.
    """
    metrics: dict[str, float] = {
        "retrain_triggered": float(decision.triggered),
        "retrain_psi_escalate": float(decision.psi.detail.get("escalate", False)),
    }
    worst = decision.psi.detail.get("worst") or {}
    if worst.get("drift_score") is not None:
        metrics["retrain_worst_psi"] = float(worst["drift_score"])
    if decision.reason == _REASON_SEASONAL:
        event = decision.calendar.detail.get("event") or {}
        if event.get("days_until") is not None:
            metrics["retrain_event_days_until"] = float(event["days_until"])
    return metrics


def mlflow_tags(decision: RetrainDecision) -> dict[str, str]:
    """Provenance tags. Strings, so the reason is queryable in the MLflow UI."""
    worst = decision.psi.detail.get("worst") or {}
    tags = {
        "trigger_reason": decision.reason,
        "run_type": "retrain-trigger",
        "psi_triggered": str(decision.psi.fired).lower(),
        "calendar_triggered": str(decision.calendar.fired).lower(),
        "escalate": str(decision.psi.detail.get("escalate", False)).lower(),
    }
    if worst.get("column"):
        tags["psi_worst_column"] = str(worst["column"])
    if worst.get("drift_score") is not None:
        tags["psi_worst_score"] = str(worst["drift_score"])
    if worst.get("source"):
        tags["psi_source"] = str(worst["source"])
    event = decision.calendar.detail.get("event") or {}
    if event.get("name"):
        tags["seasonal_event"] = str(event["name"])
    return tags


def build_report(
    decision: RetrainDecision,
    input_path: str | Path | None,
    prediction_path: str | Path | None,
    dispatch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The on-disk report, in the shape of the other drift report families."""
    return {
        **decision.as_dict(),
        "reasons_available": list(TRIGGER_REASONS),
        "inputs": {
            "input_drift_report": str(input_path) if input_path else None,
            "prediction_drift_report": str(prediction_path)
            if prediction_path
            else None,
        },
        "dispatch": dispatch
        or {
            "attempted": False,
            "reason_not_attempted": "no dispatch was requested",
        },
        "handoff": (
            "This job does not fine-tune. Retraining is scripts/kaggle_train_runs.py "
            "on a Kaggle GPU; scripts/promote_model.py is the only code allowed to "
            "move the Production alias."
        ),
        "caveat": (
            "Both drift panels are seeded draws from data/processed/test.csv, so a "
            "breach here reflects the test split, not production traffic. The "
            "seasonal signal is calendar-only and cannot be validated: no row in "
            "this corpus carries a timestamp."
        ),
    }


def read_token_file(path: str | Path) -> str:
    """Read the dispatch token from a file.

    Mirrors ``scripts/deploy_staging.py --registry-token-file``: the token
    travels through a path, never argv, so it cannot land in ``ps`` output.
    """
    return Path(path).read_text().strip()


def resolve_token(explicit: str | None = None) -> str | None:
    """Token from ``--token-file``/env, or ``None`` when unprovisioned.

    The trigger is fully functional without one: the decision, the report and
    the MLflow provenance are all produced, and the nightly job still exits 0.
    Only the GitHub dispatch is skipped. That is deliberate -- a missing secret
    must not take the nightly pipeline down or hide a real breach.
    """
    path = explicit or os.environ.get("RAAY_GITHUB_DISPATCH_TOKEN_FILE", "")
    if path:
        token_path = Path(path)
        if not token_path.exists():
            logger.warning(
                f"No dispatch token at {token_path}; GitHub dispatch will be skipped."
            )
            return None
        return read_token_file(token_path)
    inline = os.environ.get("RAAY_GITHUB_DISPATCH_TOKEN", "")
    return inline.strip() or None


def dispatch_retrain(
    payload: dict[str, Any],
    token: str,
    repository: str = _DEFAULT_REPOSITORY,
    event_type: str = "retrain",
    api_url: str = _GITHUB_API,
    timeout: float = 30.0,
    opener: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """POST the retrain event to GitHub. Never raises; returns a result dict.

    ``urllib.request`` rather than ``requests``: the latter is only a
    transitive dependency here, and the sibling tools (``deploy_staging``,
    ``canary_promote``) are stdlib-only for the same reason. The token travels
    in an Authorization header, never argv.

    A 204 with no body is success. 401/403 (bad or under-scoped token) and 404
    (wrong repo) are reported with the API's own message, so the failure is
    diagnosable from the report rather than a bare "dispatch failed".
    """
    url = f"{api_url.rstrip('/')}/repos/{repository}/dispatches"
    body = json.dumps({"event_type": event_type, "client_payload": payload}).encode()
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
            "User-Agent": "raay-retrain-trigger",
        },
    )
    open_url = opener or urllib.request.urlopen
    result: dict[str, Any] = {
        "attempted": True,
        "ok": False,
        "status": None,
        "repository": repository,
        "event_type": event_type,
        "error": None,
    }
    try:
        with open_url(request, timeout=timeout) as response:
            status = int(getattr(response, "status", 0) or 0)
            result["status"] = status
            result["ok"] = status in (200, 202, 204)
            if not result["ok"]:
                result["error"] = f"unexpected status {status}"
    except urllib.error.HTTPError as error:
        result["status"] = int(error.code)
        detail = ""
        try:
            detail = error.read().decode("utf-8", "replace")
        except OSError:
            detail = ""
        result["error"] = f"HTTP {error.code}: {detail[:300]}"
    except urllib.error.URLError as error:
        result["error"] = f"URLError: {error.reason}"
    except OSError as error:
        result["error"] = f"OSError: {error}"
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--date", default=None, help="YYYY-MM-DD (default today)")
    parser.add_argument(
        "--input-drift-report",
        default=None,
        help="Phase 6 step 1 report (default reports/drift/{date}.json)",
    )
    parser.add_argument(
        "--prediction-drift-report",
        default=None,
        help="Phase 6 step 2 report (default reports/prediction_drift/{date}.json)",
    )
    parser.add_argument(
        "--calendar",
        default=DefaultPaths.CONFIG_SEASONAL_EVENTS.value,
        help="Seasonal event config (configs/seasonal_events.yaml).",
    )
    parser.add_argument("--report-out", default=None)
    parser.add_argument(
        "--lead-days",
        type=int,
        default=21,
        help="How far ahead a confirmed seasonal event arms the trigger.",
    )
    parser.add_argument("--warn-threshold", type=float, default=0.1)
    parser.add_argument("--fail-threshold", type=float, default=0.2)
    parser.add_argument(
        "--escalate-fires",
        action="store_true",
        help=(
            "Also fire on prediction_drift's `escalate` (falling confidence AND "
            "rising class PSI). Off by default: escalate is a coupled signal "
            "deliberately kept out of its own `overall` verdict."
        ),
    )
    parser.add_argument(
        "--force-reason",
        choices=[_REASON_MANUAL],
        default=None,
        help="Fire regardless of the gates (manual operator retrain).",
    )
    parser.add_argument(
        "--token-file",
        default=None,
        help=(
            "Path to the GitHub dispatch token. Defaults to "
            "RAAY_GITHUB_DISPATCH_TOKEN_FILE, then RAAY_GITHUB_DISPATCH_TOKEN. "
            "The token is read from a file, never argv."
        ),
    )
    parser.add_argument(
        "--repository",
        default=None,
        help=(
            "owner/repo to dispatch to (default "
            f"{_DEFAULT_REPOSITORY}, overridable via RAAY_GITHUB_REPOSITORY)."
        ),
    )
    parser.add_argument(
        "--no-dispatch",
        action="store_true",
        help="Evaluate and report the decision without POSTing to GitHub.",
    )
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging.")
    return parser.parse_args(argv)


def _dispatch_payload(
    decision: RetrainDecision,
    prediction_report: dict[str, Any] | None,
    threshold: float,
) -> dict[str, Any]:
    """The ``client_payload`` the receiver's ``notify`` job reads.

    Keys mirror ``.github/workflows/retrain.yml``: ``reason`` is echoed in the
    summary, ``threshold`` overrides the refresh gate, and the rest is rendered
    into the issue body.
    """
    worst = decision.psi.detail.get("worst") or {}
    event = decision.calendar.detail.get("event") or {}
    return {
        "reason": decision.reason,
        "trigger_date": decision.day.isoformat(),
        "threshold": threshold,
        "psi": worst.get("drift_score"),
        "psi_column": worst.get("column"),
        "triage": (prediction_report or {}).get("triage"),
        "seasonal_event": event.get("name"),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # UTC, not naive local: the nightly DAG runs at 03:00 UTC and the trigger
    # report is keyed by that date, so a local-time date would file tonight's run
    # under the wrong day whenever the host's offset crosses midnight.
    day = date.fromisoformat(args.date) if args.date else datetime.now(tz=UTC).date()

    input_report_path = args.input_drift_report or str(
        Path(DefaultPaths.DRIFT_REPORTS.value) / f"{day.isoformat()}.json"
    )
    prediction_report_path = args.prediction_drift_report or str(
        Path(DefaultPaths.PREDICTION_DRIFT_REPORTS.value) / f"{day.isoformat()}.json"
    )
    report_path = args.report_out or str(
        Path(DefaultPaths.RETRAIN_TRIGGER_REPORTS.value) / f"{day.isoformat()}.json"
    )

    input_drift = _read_json(input_report_path)
    predictions = _read_json(prediction_report_path)
    events = load_calendar(args.calendar)
    decision = evaluate_trigger(
        day=day,
        input_report=input_drift,
        prediction_report=predictions,
        events=events,
        thresholds=(args.warn_threshold, args.fail_threshold),
        lead_days=args.lead_days,
        escalate_fires=args.escalate_fires,
        force_reason=args.force_reason,
    )

    repository = (
        args.repository
        or os.environ.get("RAAY_GITHUB_REPOSITORY")
        or _DEFAULT_REPOSITORY
    )
    token = None if args.no_dispatch else resolve_token(args.token_file)
    if not decision.triggered:
        dispatch: dict[str, Any] = {
            "attempted": False,
            "reason_not_attempted": f"nothing triggered (reason={decision.reason})",
        }
    elif token is None:
        dispatch = {
            "attempted": False,
            "reason_not_attempted": (
                "no dispatch token provisioned; the decision and its MLflow "
                "provenance are still recorded"
            ),
        }
    else:
        dispatch = dispatch_retrain(
            _dispatch_payload(decision, predictions, args.fail_threshold),
            token,
            repository=repository,
        )

    report = build_report(decision, input_report_path, prediction_report_path, dispatch)

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(report, indent=2, ensure_ascii=False))

    if not args.no_mlflow:
        _log_run(
            f"retrain-trigger-{day.isoformat()}",
            mlflow_metrics(decision),
            mlflow_tags(decision),
            report_path,
        )

    logger.info(
        f"Retrain trigger {day}: {decision.reason} "
        f"(triggered={decision.triggered}, psi={decision.psi.fired}, "
        f"calendar={decision.calendar.fired}) -> {report_path}"
    )
    # Report-only, like --mode drift and --mode predict-drift: a decision to
    # retrain is a recommendation to a human, not a job outcome, and no token
    # simply skips the dispatch. The exception is a dispatch that was *tried*
    # and failed -- a breach the operator asked to be notified about and wasn't
    # is an infrastructure fault, so it turns the task red (and Airflow retries)
    # rather than passing silently. The report is already on disk either way.
    if dispatch.get("attempted") and not dispatch.get("ok"):
        logger.error(f"GitHub dispatch failed: {dispatch.get('error')}")
        return 1
    return 0


def _log_run(
    run_name: str,
    metrics: dict[str, float],
    tags: dict[str, str],
    report_path: str,
) -> None:
    """Log the trigger to ``raay_batch`` with the reason as a queryable tag.

    Reuses batch_score's logger shape (idempotent experiment creation, no
    ``run_type`` collision) but adds ``set_tag`` calls, because the reason is
    the one field a human will filter the experiment by.
    """
    import mlflow

    from raay.config.env import mlflow_tracking_uri
    from raay.enums.constants import Experiments

    mlflow.set_tracking_uri(mlflow_tracking_uri())
    experiment = mlflow.get_experiment_by_name(Experiments.BATCH.value)
    if experiment is None:
        experiment_id = mlflow.create_experiment(Experiments.BATCH.value)
    else:
        experiment_id = experiment.experiment_id
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
        for tag_key, tag_value in tags.items():
            mlflow.set_tag(tag_key, tag_value)
        for metric_key, metric_value in metrics.items():
            mlflow.log_metric(metric_key, float(metric_value))
        mlflow.log_artifact(report_path, artifact_path="retrain_trigger")
        logger.info(f"Logged raay_batch run {run_name}")


if __name__ == "__main__":
    raise SystemExit(main())
