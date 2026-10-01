from __future__ import annotations

from pathlib import Path

import pandas as pd

from raay.data.feedback import FeedbackConfig

CONFIG = FeedbackConfig()


def _row(
    text: str,
    agent: str,
    model_label: str = "positive",
    corrected_label: str = "negative",
    *,
    override_id: str = "",
    captured_at: str = "2026-10-01T12:00:00+00:00",
    model_score: str = "",
    adjudicator_id: str = "",
    adjudicated_label: str = "",
    **extra: str,
) -> dict[str, str]:
    row = {
        "override_id": override_id or f"{agent}-{abs(hash((text, agent))) % 10**8}",
        "captured_at": captured_at,
        "text": text,
        "model_label": model_label,
        "corrected_label": corrected_label,
        "agent_id": agent,
        "model_score": model_score,
        "model_version": "int8-687d587004c6",
        "company": "",
        "note": "",
        "guideline_version": "v1.1",
        "adjudicator_id": adjudicator_id,
        "adjudicated_label": adjudicated_label,
    }
    row.update(extra)
    return row


def _corroborated(
    text: str = "المنتج ممتاز لكن التوصيل كان متأخر جدا",
    corrected_label: str = "negative",
) -> list[dict[str, str]]:
    """Two agents, same review, same corrected label."""
    return [
        _row(
            text,
            "agent.01",
            corrected_label=corrected_label,
            captured_at="2026-10-01T10:00:00+00:00",
        ),
        _row(
            text,
            "agent.02",
            corrected_label=corrected_label,
            captured_at="2026-10-01T11:00:00+00:00",
        ),
    ]


def _write_raw(
    rows: list[dict[str, str]], tmp_path: Path, name: str = "2026-10-01.csv"
) -> Path:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(raw_dir / name, index=False)
    return raw_dir


def _no_test_split(tmp_path: Path) -> str:
    """A test.csv with one unrelated row, so the leak guard has something to match."""
    path = tmp_path / "test.csv"
    pd.DataFrame({"text": ["a review that has nothing to do with feedback"]}).to_csv(
        path, index=False
    )
    return str(path)
