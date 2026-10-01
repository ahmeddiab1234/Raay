"""The text primitives the cleaning pipeline is built from.

Pure functions on strings (plus the fuzzy dedupe over a frame), with no I/O, no
MLflow and no DVC knowledge. ``raay.data.preprocess`` re-exports all of these so
the existing import sites -- ``feedback_text``, ``feedback_review``, the tests --
keep working against the module the pipeline is named for.

Keeping them here rather than inline in ``preprocess`` is what lets
``normalize_pipeline`` (the ordered cleaning steps) sit in its own module without
an import cycle.
"""

from __future__ import annotations

import re

import pandas as pd
from loguru import logger
from rapidfuzz import fuzz, process

# Regex patterns
ARABIC_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670]")
LATIN_CHARS = re.compile(r"[a-zA-Z]")
EMOJI_PATTERN = re.compile(r"[\U00010000-\U0010ffff]", flags=re.UNICODE)
ELONGATION_PATTERN = re.compile(r"(.)\1{2,}")


def flag_near_empty(text: str, min_length: int) -> bool:
    """Flag texts that have fewer than min_length characters."""
    return len(str(text).strip()) < min_length


def remove_diacritics(text: str) -> str:
    """Remove Arabic diacritics (tashkeel)."""
    return ARABIC_DIACRITICS.sub("", str(text))


def collapse_elongation(text: str, max_repeat: int = 2) -> str:
    """Collapse repeated characters to max_repeat."""
    if pd.isna(text):
        return text
    # The regex replaces any character repeated 3 or more times with exactly 2 instances.
    # To support dynamic max_repeat, we construct the regex.
    pattern = r"(.)\1{" + str(max_repeat) + r",}"
    replacement = r"\1" * max_repeat
    return re.sub(pattern, replacement, str(text))


def normalize_casing(text: str) -> str:
    """Normalize casing (lowercase) for Latin characters."""
    return str(text).lower()


def has_latin(text: str) -> bool:
    """Check if text contains Latin characters."""
    return bool(LATIN_CHARS.search(str(text)))


def has_emoji(text: str) -> bool:
    """Check if text contains emojis."""
    return bool(EMOJI_PATTERN.search(str(text)))


def has_diacritics(text: str) -> bool:
    """Check if text contains Arabic diacritics."""
    return bool(ARABIC_DIACRITICS.search(str(text)))


def has_elongation(text: str) -> bool:
    """Check if text contains elongated characters."""
    return bool(ELONGATION_PATTERN.search(str(text)))


def fuzzy_deduplicate(
    df: pd.DataFrame, text_col: str, threshold: float
) -> pd.DataFrame:
    """Remove fuzzy near-duplicates using rapidfuzz."""
    texts = df[text_col].tolist()
    indices = df.index.tolist()

    kept_indices: list[int] = []
    kept_texts: list[str] = []

    threshold_score = threshold * 100 if threshold <= 1.0 else threshold

    logger.info(f"Starting fuzzy deduplication with threshold {threshold_score}...")

    for idx, text in zip(indices, texts):
        if not kept_texts:
            kept_texts.append(text)
            kept_indices.append(idx)
            continue

        match = process.extractOne(
            text, kept_texts, scorer=fuzz.ratio, score_cutoff=threshold_score
        )
        if not match:
            kept_texts.append(text)
            kept_indices.append(idx)

    return df.loc[kept_indices]
