"""Heuristic Arabic dialect detection.

Converts ``text`` to a coarse dialect bucket. Markers are based on regionally
distinctive function words (negation particles, interrogatives, demonstratives,
common verbs). Scores are summed per dialect, ties broken by a fixed priority
order; text with no region markers (often formal/standard Arabic) resolves to
``msa``. Latin-script Arabizi (franco-arabe) is bucketed as ``arabizi``.

This is intentionally a lightweight, deterministic heuristic used to stratify
the held-out test set and report per-dialect evaluation breakdowns. It is *not*
a trained classifier and should not be used for production dialect tagging.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from raay.enums.constants import Dialects

# Region marker lexicons. Longer / more specific markers are weighted higher.
_MARKERS: dict[Dialects, dict[str, int]] = {
    Dialects.EGYPTIAN: {
        # negation
        "مش": 2,
        "موش": 3,
        "مش عارف": 3,
        # temporals / adverbs
        "دلوقتي": 3,
        "دلوقتى": 3,
        "بقالي": 3,
        "باقي": 1,
        "خالص": 2,
        "قوي": 2,
        "اوي": 2,
        "كده": 2,
        "كدا": 2,
        "النهارده": 3,
        "امبارح": 3,
        # interrogatives
        "فين": 3,
        "ايه": 2,
        "ليه": 2,
        "ازاي": 3,
        "ازاى": 3,
        "عايز": 3,
        "عايزين": 3,
        "عاوزه": 3,
        "عايزة": 2,
        "محصلش": 3,
        "معايا": 2,
        # modal / verb forms
        "ييجي": 2,
        "يجي": 1,
        "هييجي": 3,
        "مفيش": 3,
        "بتبقي": 3,
        "بيقول": 2,
        "بيقولوا": 3,
        "بتاع": 2,
        "بتاعة": 3,
        "اللي": 1,
        "دي": 1,
    },
    Dialects.GULF: {
        # negation
        "مو": 2,
        "مافي": 2,
        "موب": 3,
        # interrogatives
        "ليش": 3,
        "وش": 2,
        "شنو": 3,
        "ايش": 3,
        "كيفكم": 3,
        # demonstratives
        "هذي": 2,
        "هالحين": 3,
        "ذاك": 1,
        "ذي": 1,
        # modals / verbs
        "يبغى": 3,
        "يبغي": 3,
        "نبغى": 3,
        "ودي": 2,
        "ودها": 2,
        "احس": 2,
        "مره": 2,
        "صارلي": 1,
        "بالعكس": 2,
        "قهر": 2,
        "يبون": 2,
        "يبي": 2,
        "ابي": 2,
        "ابغى": 3,
        "ابغي": 3,
        # particles
        "وايد": 3,
        "احلى": 1,
        "عقب": 2,
        "بعدين": 1,
    },
    Dialects.LEVANTINE: {
        # negation
        "مش": 1,
        "مافي": 2,
        # interrogatives
        "شو": 3,
        "كيف": 2,
        "مين": 2,
        "وين": 3,
        "ليش": 2,
        "شلون": 3,
        # demonstratives
        "هيك": 3,
        "هكذا": 1,
        "هون": 2,
        # modals / verbs
        "بدي": 3,
        "بدنا": 3,
        "بدك": 3,
        "خليني": 3,
        "خلينا": 2,
        "عم": 1,
        "بروح": 2,
        "بشوف": 2,
        "عندي": 1,
        # temporals
        "هلق": 3,
        "هلقيت": 3,
        "بكرة": 1,
        # particles
        "زبط": 2,
        "كتير": 2,
        "مشان": 3,
        "عشان": 1,
    },
    Dialects.MAGHREBI: {
        "واش": 3,
        "علاش": 3,
        "كيفاش": 3,
        "دابا": 3,
        "دaba": 1,
        "مزيان": 3,
        "شحال": 3,
        "الو": 2,
        "بزاف": 3,
        "صارو": 1,
        "هاد": 2,
        "هاذ": 2,
        "شكون": 3,
        "اش": 2,
    },
}

# Public read-only alias for the region marker lexicons; the notebook EDA and
# other callers can inspect/aggregate the marker lists without poking privates.
DIALECT_MARKERS: dict[Dialects, dict[str, int]] = _MARKERS

# Fixed priority used to break ties and order the (rare) all-zero return.
_PRIORITY: list[Dialects] = [
    Dialects.LEVANTINE,
    Dialects.GULF,
    Dialects.EGYPTIAN,
    Dialects.MAGHREBI,
]

# Latin-script Arabizi heuristic: >= 3 Latin letters => likely franco-arabe.
_LATIN_MIN = 3


@dataclass(frozen=True)
class DialectResult:
    dialect: Dialects
    confidence: float


def detect_dialect(text: str) -> Dialects:
    """Detect the dominant dialect bucket for a single text string."""
    dialect = detect_dialect_scored(text).dialect
    return dialect


def detect_dialect_scored(text: str) -> DialectResult:
    """Return a dialect guess plus a normalized confidence score in [0, 1]."""
    if text is None:
        return DialectResult(Dialects.MSA, 0.0)

    norm = " ".join(str(text).split())

    # Arabizi heuristic
    latin = sum(1 for ch in norm if "a" <= ch.lower() <= "z")
    if latin >= _LATIN_MIN:
        return DialectResult(Dialects.ARABIZI, min(1.0, latin / 20.0))

    scores: dict[Dialects, int] = {
        d: 0 for d in Dialects if d not in (Dialects.MSA, Dialects.ARABIZI)
    }
    for dialect, markers in _MARKERS.items():
        for marker, weight in markers.items():
            if marker in norm:
                scores[dialect] += weight

    total = sum(scores.values())
    if total == 0:
        return DialectResult(Dialects.MSA, 0.0)

    best = max(_PRIORITY, key=lambda d: scores[d])
    confidence = scores[best] / total
    return DialectResult(best, confidence)


def add_dialect_column(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Attach a ``dialect`` column (and ``dialect_confidence``) to a DataFrame."""
    out = df.copy()
    results = out[text_col].map(detect_dialect_scored)
    out["dialect"] = [r.dialect for r in results]
    out["dialect_confidence"] = [r.confidence for r in results]
    return out
