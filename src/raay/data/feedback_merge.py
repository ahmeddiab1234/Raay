"""``--mode merge``: reviewed rows -> ``data/processed/train_feedback.csv``.

This is the DVC stage and the only piece that touches training data. It reads the
reviewed artifact **as found** and never re-reviews (see ``raay.data.feedback``),
and it appends rather than overwrites so no previously-validated hard negative is
ever dropped.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from raay.data.dialect import add_dialect_column
from raay.data.feedback_schema import (
    _PROPORTION_DP,
    MERGED_COLUMNS,
    REVIEWED_COLUMNS,
    TRAINABLE_STATUSES,
    FeedbackConfig,
)
from raay.data.feedback_text import _text_key_only, coerce_label
from raay.enums.constants import LABELS


def label_proportions(series: pd.Series) -> dict[str, float]:
    """Value counts as fractions of the total, zero-filled over :data:`LABELS`.

    Zero-filled so the report's shape is stable across days: an absent class is a
    fact about this batch, not a missing key.
    """
    counts = series.value_counts()
    total = float(counts.sum())
    if total == 0:
        return {label: 0.0 for label in LABELS}
    return {
        label: round(float(counts.get(label, 0)) / total, _PROPORTION_DP)
        for label in LABELS
    }


def read_merged(path: str) -> pd.DataFrame | None:
    """The current merged file, or ``None`` when it does not exist yet."""
    if not Path(path).exists():
        return None
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    return frame if "text" in frame.columns else None


def build_merged(
    reviewed: pd.DataFrame,
    config: FeedbackConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Turn reviewed rows into train-shaped rows.

    Returns the frame and a summary recording anything suppressed, so a cap that
    ate half the batch shows up in the report instead of being inferred from a
    mysteriously smaller file.
    """
    summary: dict[str, Any] = {
        "n_candidates": len(reviewed),
        "n_eligible": 0,
        "n_accepted": 0,
        "n_collapsed_to_one_row_per_review": 0,
        "n_suppressed_by_cap": 0,
        "neutral_cap": config.max_neutral_per_batch,
        "accepted_label_proportions": {label: 0.0 for label in LABELS},
    }
    if reviewed.empty:
        return pd.DataFrame(columns=list(MERGED_COLUMNS)), summary

    eligible = reviewed[reviewed["status"].isin(TRAINABLE_STATUSES)].copy()
    summary["n_eligible"] = len(eligible)
    if eligible.empty:
        return pd.DataFrame(columns=list(MERGED_COLUMNS)), summary

    # Resolve the adjudicated label *before* the cap, so the cap is applied to the
    # label the row will actually train with. Capping on `corrected_label` would let
    # adjudicated-to-neutral rows through uncapped and defeat the whole point.
    adjudicated = eligible["adjudicated_label"].map(coerce_label)
    eligible["_final_label"] = np.where(
        adjudicated != "", adjudicated, eligible["corrected_label"]
    )

    # One review, one training row.
    #
    # Two agents corroborating a single review produce two assertion rows, and both
    # carry the same text and the same label. Emitting both would put an identical
    # (text, label) pair into train.csv twice: the reviewer would weight that one
    # hard negative 2x for no informational reason, and a review five agents flagged
    # would weight 5x. The corroboration is *evidence for the label*, recorded in
    # `corroborating_agents` -- it is not extra training data. Deduping here also
    # keeps the Neutral cap counting reviews rather than votes.
    before = len(eligible)
    eligible = eligible.drop_duplicates(subset=["text"], keep="first")
    summary["n_collapsed_to_one_row_per_review"] = before - len(eligible)

    # Deterministic order so the cap takes the same rows on every re-run. A cap
    # that picked arbitrarily would make the output non-reproducible, which is
    # exactly what `git diff --exit-code dvc.lock` exists to catch.
    eligible = eligible.sort_values(["captured_at", "override_id"], kind="stable")

    if config.max_neutral_per_batch >= 0:
        neutral_idx = eligible.index[eligible["_final_label"] == "neutral"]
        overflow = neutral_idx[config.max_neutral_per_batch :]
        if len(overflow):
            summary["n_suppressed_by_cap"] = len(overflow)
            eligible = eligible.drop(index=overflow)

    if eligible.empty:
        return pd.DataFrame(columns=list(MERGED_COLUMNS)), summary

    merged = pd.DataFrame(
        {
            # The adjudicated label wins: it is the resolution of a dispute the
            # two agents could not settle. falls back to the agreed label.
            "label": eligible["_final_label"],
            "text": eligible["text"],
            "company": eligible["company"],
            "model_label": eligible["model_label"],
            "model_version": eligible["model_version"],
            "captured_at": eligible["captured_at"],
            "override_id": eligible["override_id"],
            "qa_status": eligible["status"],
            "corroborated_by": eligible["corroborating_agents"],
        }
    )
    merged["source"] = "customer_service_feedback"
    merged["guideline_version"] = eligible["guideline_version"]
    # Never true: review_rows excluded these rows. Carrying the column (rather
    # than dropping it) keeps the schema identical to train.csv.
    merged["is_near_empty"] = False

    merged = add_dialect_column(merged, text_col="text")
    ordered = merged[list(MERGED_COLUMNS)]
    summary["n_accepted"] = len(ordered)
    summary["accepted_label_proportions"] = label_proportions(ordered["label"])
    return ordered, summary


def merge_reviewed(
    reviewed_csv: str,
    out_csv: str,
    config: FeedbackConfig,
) -> dict[str, Any]:
    """Idempotent: the same reviewed file always yields the same merged output."""
    if Path(reviewed_csv).exists():
        reviewed = pd.read_csv(reviewed_csv, dtype=str, keep_default_na=False)
    else:
        logger.warning(
            f"No reviewed overrides at {reviewed_csv}; writing an empty merge."
        )
        reviewed = pd.DataFrame(columns=list(REVIEWED_COLUMNS))

    existing = read_merged(out_csv)
    # Drop rows already present, so re-running after a crash never double-counts
    # a correction that landed before the failure.
    if existing is not None and not existing.empty:
        seen = {_text_key_only(t) for t in existing["text"]}
        reviewed = reviewed[~reviewed["text"].map(_text_key_only).isin(seen)]

    fresh, summary = build_merged(reviewed, config)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)

    # Append to what is already merged, never replace it. The training sidecar is
    # cumulative on purpose: `train.py` concatenates this whole file onto train.csv,
    # so overwriting it with only the newest batch would silently *drop* every
    # previously-validated hard negative. Rows are ordered by a content hash so the
    # accumulated file stays byte-stable across re-runs.
    combined = fresh
    if existing is not None and not existing.empty:
        combined = pd.concat([existing, fresh], ignore_index=True)
    if not combined.empty:
        combined = combined.sort_values("override_id", kind="stable").reset_index(
            drop=True
        )
    combined = combined.reindex(columns=list(MERGED_COLUMNS))
    combined.to_csv(out_csv, index=False)

    summary["cumulative_rows"] = len(combined)
    summary["merged_path"] = out_csv
    return summary
