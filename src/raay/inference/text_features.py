"""Per-text drift features: OOV rate/bucket, dialect, length, confidence.

These are the engineered columns that do *not* need the encoder, so they are
the cheap half of the input-drift picture and the half a test can pin exactly.
``oov_rates`` reuses ``serve.runtime._preprocess`` rather than keeping a second
arabert processor cache: the two paths must normalize identically or the OOV
rate would describe text the model never sees.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from raay.data.dialect import detect_dialect_scored
from raay.enums.constants import Models
from raay.inference.drift_columns import COL_CONFIDENCE
from raay.serving.runtime import _preprocess

# Per-class probability columns ``score_input`` writes; used to derive
# ``confidence_score`` when ``predicted_score`` is absent.
LABEL_PROB_COLUMNS = ("positive", "negative", "neutral")

# Upper edges of the OOV buckets, in increasing order, with ``>0.10`` last.
_OOV_EDGES = (0.0, 0.02, 0.05, 0.10)
_OOV_LABELS = ("none", "low", "moderate", "high")

# Tokenizer batch for the OOV pass. The embedding pass is far heavier, so this
# is only about not building a giant python list of ids at once.
OOV_BATCH = 256


def oov_rates(
    texts: Sequence[str],
    tokenizer: Any,
    model_name: str = Models.TEACHER.value,
    batch_size: int = OOV_BATCH,
    preprocess: Callable[[str], str] | None = None,
) -> np.ndarray:
    """Per-text fraction of AraBERT subword tokens that are ``[UNK]``.

    Measured on the *preprocessed* text, which is what the model actually
    tokenizes: measuring the raw string would count ``[UNK]``s that
    normalization then removes, so the rate would not describe what the
    encoder sees. ``preprocess`` is injectable so tests can bypass pyarabic,
    which rewrites text in ways that have nothing to do with OOV.

    Deliberately **not** truncated at ``max_length``: the signal is "how much of
    this review is unspellable to the tokenizer", and clipping to the first 128
    subwords would hide novelty that only appears late in a long review. The
    embedding pass does truncate, because the model has to.
    """
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if unk_id is None:
        raise ValueError(
            "tokenizer exposes no unk_token_id, so the OOV rate is undefined; "
            "refusing to report a constant 0.0 as if it were a measurement"
        )
    normalize = preprocess or (lambda t: _preprocess(t, model_name))
    rates = np.zeros(len(texts), dtype=np.float64)
    for i in range(0, len(texts), batch_size):
        chunk = [normalize(t) for t in texts[i : i + batch_size]]
        encoded = tokenizer(chunk, add_special_tokens=False, truncation=False)[
            "input_ids"
        ]
        for j, ids in enumerate(encoded):
            if len(ids) == 0:
                continue
            rates[i + j] = sum(1 for t in ids if t == unk_id) / len(ids)
    return rates


def oov_bucket(rate: float) -> str:
    """Bucket label for an ``oov_rate``: none / low / moderate / high.

    This exists because **PSI on the raw rate is not safe to gate**. Evidently
    picks its binning from the data: with more than 20 distinct values it hands
    the combined series to ``numpy.histogram_bin_edges(bins="sturges")``, and
    when the values span a tiny range numpy raises ``Too many bins for data
    range``. ``oov_rate`` is precisely the column that hits this -- most real
    reviews sit at 0.0, so the series is a near-spike -- and an exception there
    takes down the whole nightly report.

    The bucket has at most four distinct values, which keeps Evidently on its
    "few unique values" path (no histogram, no range to collapse), and it is
    the form a human actually wants anyway: "6% of reviews are now >10%
    unspellable" rather than a PSI over 30 quantile slices of a rate.
    """
    if rate > _OOV_EDGES[-1]:
        return _OOV_LABELS[-1]
    for edge, label in zip(_OOV_EDGES, _OOV_LABELS, strict=True):
        if rate <= edge:
            return label
    return _OOV_LABELS[-1]  # pragma: no cover - the zip above always returns


def dialect_labels(texts: Sequence[str]) -> list[str]:
    """Canonical (``Dialects.value``) dialect label per text.

    Recomputed rather than read from the stored ``dialect`` column:
    ``Dialects`` is a ``str, Enum``, so ``str(Dialects.ARABIZI) ==
    "Dialects.ARABIZI"`` and a frame round-tripped through CSV carries that
    prefix.
    """
    return [detect_dialect_scored(t).dialect.value for t in texts]


def text_lengths(texts: Sequence[str]) -> np.ndarray:
    """Review length in characters, the cheapest distribution there is."""
    return np.array([len(t) for t in texts], dtype=np.float64)


def confidence_scores(frame: pd.DataFrame) -> np.ndarray:
    """Row confidence: ``predicted_score`` if present, else max class prob.

    Raises rather than filling zeros when neither exists -- a constant 0.0
    column would pass PSI forever and read as "no drift" in every report.
    """
    if "predicted_score" in frame.columns:
        return frame["predicted_score"].to_numpy(dtype=np.float64)
    present = [c for c in LABEL_PROB_COLUMNS if c in frame.columns]
    if not present:
        raise ValueError(
            f"cannot derive {COL_CONFIDENCE}: frame has neither 'predicted_score' "
            f"nor any of {list(LABEL_PROB_COLUMNS)} (columns={list(frame.columns)})"
        )
    return frame[present].to_numpy(dtype=np.float64).max(axis=1)
