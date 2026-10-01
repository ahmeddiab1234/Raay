"""Class-mix and confidence summaries for the prediction-drift report.

Pure helpers over scored panels: no I/O beyond reading a labelled CSV and the
prior day reports. Kept separate from the verdict logic in
``prediction_report`` so each can be read and tested on its own.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from raay.enums.constants import LABELS, DataColumns, DefaultPaths

COL_PREDICTED_LABEL = "predicted_label"
COL_PREDICTED_SCORE = "predicted_score"


def training_label_reference(path: str | Path | None = None) -> pd.DataFrame:
    """The training label prior as a frame the PSI helper can consume.

    Returns the train split's labels under ``predicted_label`` so the same
    ``_psi_per_column`` call serves both references -- one implementation of
    PSI rather than two that can disagree.

    Raises if any of the three classes is absent: a prior missing a class
    cannot be compared against a panel that has one, and silently producing a
    two-class prior would understate the drift instead.
    """
    csv_path = Path(path or DefaultPaths.TRAIN_SPLIT.value)
    labels = pd.read_csv(csv_path, usecols=[DataColumns.LABEL.value])[
        DataColumns.LABEL.value
    ].dropna()
    missing = [label for label in LABELS if label not in set(labels)]
    if missing:
        raise ValueError(
            f"{csv_path} has no rows for class(es) {missing}; expected all of "
            f"{list(LABELS)} to build a comparable label prior"
        )
    return pd.DataFrame({COL_PREDICTED_LABEL: labels})


def class_distribution(frame: pd.DataFrame) -> dict[str, float]:
    """Share of each class. Every key is present, so an absent class reads 0.0."""
    if COL_PREDICTED_LABEL not in frame.columns:
        raise KeyError(
            f"panel is missing {COL_PREDICTED_LABEL!r}; has {list(frame.columns)}"
        )
    counts = frame[COL_PREDICTED_LABEL].value_counts(normalize=True)
    return {label: round(float(counts.get(label, 0.0)), 6) for label in LABELS}


def share_delta_pp(
    current: dict[str, float], reference: dict[str, float]
) -> dict[str, float]:
    """Per-class movement in percentage points; negative means the class grew less."""
    return {
        label: round((current[label] - reference[label]) * 100, 3) for label in LABELS
    }


def mean_confidence(frame: pd.DataFrame) -> float:
    if COL_PREDICTED_SCORE not in frame.columns:
        raise KeyError(
            f"panel is missing {COL_PREDICTED_SCORE!r}; has {list(frame.columns)}"
        )
    return float(frame[COL_PREDICTED_SCORE].mean())


def rolling_baseline(values: list[float], window: int, min_days: int) -> dict[str, Any]:
    """Mean/std over the last ``window`` values, and whether to trust the z-score.

    A rolling baseline adapts to slow drift, which is the point -- but it is
    meaningless until there is enough of it. Below ``min_days`` the mean and
    std are still reported (they are honest descriptions of what exists) while
    ``z_score`` is withheld, so a two-day "trend" cannot manufacture a signal.
    """
    recent = [v for v in values[-window:] if v is not None]
    n = len(recent)
    out: dict[str, Any] = {
        "window": window,
        "n_days": n,
        "min_days": min_days,
        "sufficient_history": n >= min_days,
    }
    if n == 0:
        return out | {"mean": None, "std": None, "z_score": None}
    mean = float(np.mean(recent))
    std = float(np.std(recent, ddof=1)) if n > 1 else 0.0
    out["mean"] = round(mean, 6)
    out["std"] = round(std, 6)
    if n >= min_days and std > 0:
        out["z_score"] = None  # filled in by the caller, which knows today's value
    else:
        # `reason` is what makes this auditable: a withheld z-score is a
        # decision, and the next person needs to know which of the two it was.
        out["z_score"] = None
        out["reason"] = (
            f"need >= {min_days} days, have {n}"
            if n < min_days
            else "rolling std is 0, so every day is identical and the z-score is undefined"
        )
    return out


def confidence_z_score(today: float, baseline: dict[str, Any]) -> float | None:
    if not baseline.get("sufficient_history") or baseline.get("std") in (None, 0):
        return None
    return round((today - baseline["mean"]) / baseline["std"], 3)


def collect_confidence_history(
    reports_dir: str | Path, exclude: str | None = None
) -> list[float]:
    """Mean confidence per prior day, oldest first.

    Reads the mean already recorded in each report rather than rescoring the
    panels: the point of a history is that it is cheap. Reports written before
    ``feature_means`` existed are skipped -- the 2026-09-23/24 reports predate
    it -- which is exactly why this tolerates the missing key instead of
    indexing it.
    """
    directory = Path(reports_dir)
    if not directory.is_dir():
        return []
    values: list[tuple[str, float]] = []
    for report in sorted(directory.glob("*.json")):
        if exclude is not None and report.stem == exclude:
            continue
        try:
            payload = json.loads(report.read_text())
        except (OSError, json.JSONDecodeError):
            logger.warning(f"Skipping unreadable prediction-drift report {report}")
            continue
        mean = payload.get("confidence", {}).get("mean")
        if isinstance(mean, (int, float)):
            values.append((report.stem, float(mean)))
    return [value for _, value in sorted(values)]
