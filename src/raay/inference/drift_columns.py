"""Names and aggregate summaries for the engineered drift columns.

Kept separate from the code that *computes* the columns because three different
callers need to agree on the names without importing an encoder: the drift
column list, the verdict's ``feature_means`` block and the tests. Nothing here
touches a tokenizer, a model or a frame's ``text``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

# Engineered column names. Kept as module constants because the drift column
# list, the feature means in the verdict and the tests all have to agree.
COL_TEXT_LENGTH = "text_length"
COL_CONFIDENCE = "confidence_score"
COL_OOV_RATE = "oov_rate"
COL_OOV_BUCKET = "oov_bucket"
COL_DIALECT_LABEL = "dialect_label"
EMBEDDING_PREFIX = "embedding_pc"

# ``embedding_pc1`` .. ``embedding_pcN``.
PCS = 10


def embedding_pc_columns(n_components: int = PCS) -> tuple[str, ...]:
    """``("embedding_pc1", ..., "embedding_pcN")`` in ascending order."""
    return tuple(f"{EMBEDDING_PREFIX}{i}" for i in range(1, n_components + 1))


def default_drift_columns(n_components: int = PCS) -> tuple[str, ...]:
    """Every column the engineered drift check gates, in report order.

    ``positive`` stays in the list even though ``confidence_score`` is the row
    max: the max masks a drop in ``positive`` that leaves another class on top,
    so the two are not redundant.
    """
    return (
        "predicted_label",
        "positive",
        COL_CONFIDENCE,
        COL_TEXT_LENGTH,
        COL_OOV_BUCKET,
        COL_OOV_RATE,
        COL_DIALECT_LABEL,
        *embedding_pc_columns(n_components),
    )


def dialect_mix(
    frame: pd.DataFrame, column: str = COL_DIALECT_LABEL
) -> dict[str, float]:
    """Label -> share of rows, in every label seen (0.0 for the ones absent)."""
    if column not in frame.columns:
        raise ValueError(f"frame has no {column!r} column")
    counts = frame[column].astype(str).value_counts(normalize=True)
    return {str(k): round(float(v), 6) for k, v in counts.items()}


def dialect_total_variation(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    column: str = COL_DIALECT_LABEL,
) -> dict[str, Any]:
    """How far the dialect mix moved, as shares plus total variation distance.

    PSI on the categorical column is the gate; this is the human-readable
    version of the same movement ("5 points of share went from Egyptian to
    Gulf"), which is what you need to decide whether the shift matters.

    Labels present on only one side are counted as a zero share on the other
    rather than being left out of the union, so a dialect that *appears* for
    the first time moves this number -- the whole point of tracking the mix.
    """
    ref_mix = dialect_mix(reference, column)
    cur_mix = dialect_mix(current, column)
    labels = sorted(set(ref_mix) | set(cur_mix))
    union = {
        k: {"reference": ref_mix.get(k, 0.0), "current": cur_mix.get(k, 0.0)}
        for k in labels
    }
    tvd = 0.5 * sum(
        abs(entry["current"] - entry["reference"]) for entry in union.values()
    )
    return {
        "column": column,
        "reference": ref_mix,
        "current": cur_mix,
        "shares": union,
        "total_variation": round(float(tvd), 6),
    }


def feature_means(frame: pd.DataFrame, columns: Sequence[str]) -> dict[str, float]:
    """Mean of the numeric engineered features, for trending in MLflow."""
    means: dict[str, float] = {}
    for col in columns:
        if col in frame.columns and pd.api.types.is_numeric_dtype(frame[col]):
            means[col] = round(float(frame[col].mean()), 6)
    return means
