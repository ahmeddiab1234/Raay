"""Data types and status constants for the CI metrics gate.

A ``Row`` is one compared metric; a ``GateResult`` is the whole evaluation. Both
are frozen: this is a reporting tool, and mutating a verdict after it has been
rendered is how a gate ends up reporting something it did not decide.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

DEFAULT_THRESHOLD = 0.005
DEFAULT_BASE = "origin/dev"

PASS = "PASS"
FAIL = "FAIL"
NEW = "NEW"

REASON_DRIFT = "drift beyond threshold"
REASON_ADDED = "metric added"
REASON_REMOVED = "metric removed"
REASON_UNREADABLE = "unreadable baseline"
REASON_MALFORMED = "malformed diff entry"
REASON_NEW_FILE = "no baseline for this metrics file"


@dataclass(frozen=True)
class Row:
    """One compared metric."""

    path: str
    metric: str
    old: Any
    new: Any
    change: float | None
    status: str
    reason: str = ""

    @property
    def failed(self) -> bool:
        return self.status == FAIL


@dataclass(frozen=True)
class GateResult:
    """Outcome of one gate evaluation."""

    rows: list[Row]
    errors: list[str] = field(default_factory=list)
    ignored: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors and not any(row.failed for row in self.rows)

    @property
    def violations(self) -> list[Row]:
        return [row for row in self.rows if row.failed]
