"""Text normalization and label coercion for the feedback loop.

``normalize_text`` is load-bearing and applied *first*, not at merge time: the
training corpus went through ``raay.data.preprocess`` and raw CS text has not,
so without this the model would learn the normalization artifact instead of the
sentiment -- and corroboration, the leakage guard and the near-empty check would
all be comparing like with unlike.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from raay.data.preprocess import (
    collapse_elongation,
    normalize_casing,
    remove_diacritics,
)
from raay.enums.constants import LABELS


def normalize_text(text: Any, elongation_max_repeat: int = 2) -> str:
    """Apply the *same* normalization the training corpus got.

    Not optional, and applied *first* rather than at merge time. Two reasons:

    * ``data/processed/train.csv`` texts went through diacritic stripping,
      elongation collapsing and casing normalization in ``raay.data.preprocess``;
      raw text from a CS tool has not. Writing both into one training file puts a
      distribution artifact in front of the tokenizer and the model learns the
      artifact rather than the sentiment.
    * Corroboration, the leakage guard and the near-empty check then all compare
      like with like. Two agents posting the same review, one with tashkeel and
      one without, are one review -- and a test-split row only matches its
      feedback twin if both sides are normalized first.
    """
    return normalize_casing(
        collapse_elongation(remove_diacritics(text), elongation_max_repeat)
    )


def text_key(text: Any) -> str:
    """Whitespace-insensitive key for "the same review" comparisons."""
    return " ".join(normalize_text(text).split())


def coerce_label(value: Any) -> str:
    """A label in :data:`LABELS`, else ``""``."""
    text = str(value).strip()
    return text if text in LABELS else ""


def numeric_or_none(value: Any) -> float | None:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        text = str(value).strip()
        return float(text) if text else None
    except (TypeError, ValueError):
        return None


def _text_key_only(text: Any) -> str:
    """Whitespace-collapsed key for text already known to be normalized."""
    return " ".join(str(text).split())
