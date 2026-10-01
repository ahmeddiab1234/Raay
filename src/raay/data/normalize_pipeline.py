"""The cleaning pipeline itself: flag, strip, collapse, normalize, dedupe.

Split out of ``preprocess`` so the DVC stage's ``main`` is only the orchestration
(wire paths, open the MLflow run, write outputs) and the cleaning order lives in
one readable function.

The order is load-bearing and must not be rearranged:

1. ``is_near_empty`` is flagged **before** normalization, because stripping
   diacritics and collapsing elongation can turn a trivially-short review into
   something that looks long enough -- and a review too short to carry sentiment
   should stay flagged.
2. Diacritics, then elongation, then casing. Casing is last because
   ``normalize_casing`` only touches Latin runs and is safe on already-stripped
   Arabic; doing it first is equally safe but the pre-normalization percentage
   stats would then measure a different text than the one that gets scored.
3. Dedup runs last, on the final normalized text, so two reviews that differ only
   in diacritics or elongation collapse into one. Exact before fuzzy: exact is
   free and removes the pairs the O(n^2) ratio check would otherwise chew on.

**Emojis are deliberately kept** (an explicit product decision); step 5 in the
original numbering is intentionally a no-op, and ``has_emoji`` exists only to
report the share in the metrics file.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from loguru import logger

from raay.data.text_clean import (
    collapse_elongation,
    flag_near_empty,
    fuzzy_deduplicate,
    has_diacritics,
    has_elongation,
    has_emoji,
    has_latin,
    normalize_casing,
    remove_diacritics,
)


def pre_normalization_stats(df: pd.DataFrame) -> dict[str, float]:
    """Percentage of reviews containing each noisy character class."""
    return {
        "latin_chars": float(df["text"].apply(has_latin).mean() * 100),
        "diacritics": float(df["text"].apply(has_diacritics).mean() * 100),
        "elongated": float(df["text"].apply(has_elongation).mean() * 100),
        "emoji": float(df["text"].apply(has_emoji).mean() * 100),
    }


def normalize_frame(
    df: pd.DataFrame, prep_config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the cleaning pipeline; returns the cleaned frame and its metrics."""
    min_char_length = prep_config.get("min_char_length", 10)
    elongation_max_repeat = prep_config.get("elongation_max_repeat", 2)
    dedup_similarity_threshold = prep_config.get("dedup_similarity_threshold", 0.9)

    raw_row_count = len(df)
    logger.info(f"Raw row count: {raw_row_count}")

    percentages = pre_normalization_stats(df)
    logger.info(
        f"Pre-normalization stats: {percentages['latin_chars']:.2f}% Latin, "
        f"{percentages['diacritics']:.2f}% Diacritics, "
        f"{percentages['elongated']:.2f}% Elongated, "
        f"{percentages['emoji']:.2f}% Emoji"
    )

    df["is_near_empty"] = df["text"].apply(
        lambda x: flag_near_empty(x, min_char_length)
    )
    near_empty_count = int(df["is_near_empty"].sum())
    logger.info(f"Near-empty flagged count: {near_empty_count}")

    logger.info("Stripping Arabic diacritics...")
    df["text"] = df["text"].apply(remove_diacritics)

    logger.info(f"Collapsing elongated characters to max {elongation_max_repeat}...")
    df["text"] = df["text"].apply(
        lambda x: collapse_elongation(x, elongation_max_repeat)
    )

    logger.info("Normalizing Latin casing...")
    df["text"] = df["text"].apply(normalize_casing)

    pre_exact_count = len(df)
    df = df.drop_duplicates(subset=["text"])
    exact_dupes_removed = pre_exact_count - len(df)
    logger.info(f"Exact duplicates removed: {exact_dupes_removed}")

    pre_fuzzy_count = len(df)
    df = fuzzy_deduplicate(df, "text", dedup_similarity_threshold)
    near_dupes_removed = pre_fuzzy_count - len(df)
    logger.info(f"Fuzzy near-duplicates removed: {near_dupes_removed}")

    final_row_count = len(df)
    logger.info(f"Final row count: {final_row_count}")

    metrics = {
        "raw_row_count": raw_row_count,
        "near_empty_flagged_count": near_empty_count,
        "exact_duplicates_removed": exact_dupes_removed,
        "near_duplicates_removed": near_dupes_removed,
        "final_row_count": final_row_count,
        "percentages_pre_norm": {
            key: round(value, 2) for key, value in percentages.items()
        },
    }
    return df, metrics
