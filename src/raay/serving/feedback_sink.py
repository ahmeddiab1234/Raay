"""Append-only per-day CSV sink for captured overrides.

Single writer by construction: one lock around the whole append, and the
existence check that decides who writes the header is taken *inside* it. Several
workers would mean a real database -- pretending a lock makes a CSV concurrent is
how rows get silently interleaved into unparseable lines.
"""

from __future__ import annotations

import csv
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from raay.enums.constants import DefaultPaths
from raay.serving.feedback_schema import RAW_COLUMNS, FeedbackOverride


class FeedbackSink:
    """Append-only CSV per UTC day, single writer.

    ``base_dir`` and ``clock`` are injected so tests never touch ``data/``.
    """

    def __init__(
        self,
        base_dir: str = DefaultPaths.FEEDBACK_RAW.value,
        clock: Any = None,
    ) -> None:
        self._base_dir = Path(base_dir)
        self._clock = clock or (lambda: datetime.now(UTC))
        self._lock = threading.Lock()
        self.rows_written = 0

    def now(self) -> datetime:
        """The sink's clock, always timezone-aware UTC."""
        stamp = self._clock()
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=UTC)
        return stamp.astimezone(UTC)

    @staticmethod
    def day_of(stamp: datetime) -> str:
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=UTC)
        return stamp.astimezone(UTC).date().isoformat()

    def path_for(self, day: str) -> Path:
        return self._base_dir / f"{day}.csv"

    def ids_for(self, day: str) -> set[str]:
        """``override_id``s already persisted for ``day`` (empty when absent).

        Read from disk rather than kept in memory precisely because the process
        restarts: an in-memory set would let a replayed POST through as a new
        row after every deploy.
        """
        path = self.path_for(day)
        if not path.exists():
            return set()
        with open(path, newline="", encoding="utf-8") as handle:
            return {row.get("override_id", "") for row in csv.DictReader(handle)}

    def append(self, override: FeedbackOverride, override_id: str) -> None:
        captured_at = override.captured_at or self.now()
        row = {
            "override_id": override_id,
            "captured_at": captured_at.astimezone(UTC).isoformat(),
            "text": override.text,
            "model_label": override.model_label,
            "corrected_label": override.corrected_label,
            "agent_id": override.agent_id,
            "model_score": "" if override.model_score is None else override.model_score,
            "model_version": override.model_version,
            "company": override.company,
            "note": override.note,
            "guideline_version": override.guideline_version,
        }
        with self._lock:
            path = self.path_for(self.day_of(captured_at))
            path.parent.mkdir(parents=True, exist_ok=True)
            # Existence is tested inside the lock: two concurrent first-writes
            # would otherwise both emit a header.
            is_new_file = not path.exists()
            with open(path, "a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(RAW_COLUMNS))
                if is_new_file:
                    writer.writeheader()
                writer.writerow(row)
            self.rows_written += 1
