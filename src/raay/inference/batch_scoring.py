"""Scoring a day's panel of reviews in large batches against the INT8 graph.

The nightly job's first two modes live here:

- ``make_input``: there is no live scraper in this repo, so "a day's worth of
  reviews" is a *deterministic* sample of ``data/processed/test.csv``, seeded by
  the date string. Same day, same rows -- which is what makes a re-run
  idempotent and a diff readable.
- ``score_input``: run the whole panel through ``serve.predict_probs`` in
  ``--batch-size`` chunks and persist every class probability, the argmax label
  and its score.

Both are deliberately free of any MLflow or drift concerns; see
:mod:`raay.inference.batch_tracking` and :mod:`raay.inference.drift_check`.
"""

from __future__ import annotations

import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import pandas as pd
from loguru import logger
from transformers import AutoTokenizer

from raay.enums.constants import LABELS, DefaultPaths, Models
from raay.serving.runtime import load_id2label, predict_probs


def _date_seed(date: str) -> int:
    """Deterministic uint32 seed for a date string (same day -> same sample)."""
    return zlib.crc32(date.encode("utf-8"))


@dataclass
class Scorer:
    """Lazy INT8 ONNX scorer sharing ``serve.predict_probs``.

    ``score`` returns the full row-softmax probability matrix so the nightly
    job can persist every class probability, not just the argmax. Nothing is
    loaded until the first call: ``make-input`` and the drift reference builder
    construct a ``Scorer`` they may never score anything with.
    """

    onnx_path: str = DefaultPaths.ONNX_INT8_MODEL.value
    tokenizer_dir: str = DefaultPaths.BASELINE_MODEL.value
    model_name: str = Models.TEACHER.value
    max_length: int = 128
    batch_size: int = 64
    _session: Any = field(init=False, repr=False, default=None)
    _tokenizer: Any = field(init=False, repr=False, default=None)
    _id2label: dict[int, str] = field(init=False, repr=False, default_factory=dict)
    _loaded: bool = field(init=False, repr=False, default=False)

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._session = ort.InferenceSession(
            self.onnx_path, providers=["CPUExecutionProvider"]
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_dir)
        self._id2label = load_id2label(self.tokenizer_dir)
        self._loaded = True
        logger.info(
            f"Batch scorer loaded {self.onnx_path} labels={self._id2label} "
            f"providers={self._session.get_providers()}"
        )

    @property
    def label_columns(self) -> list[str]:
        return [self._id2label[i] for i in sorted(self._id2label)]

    def score(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, len(self.label_columns)))
        self._ensure_loaded()
        assert self._session is not None and self._tokenizer is not None
        return predict_probs(
            self._session,
            self._tokenizer,
            texts,
            self.model_name,
            self.max_length,
            batch_size=self.batch_size,
        )


def make_input(
    pool_csv: str,
    date: str,
    samples: int,
    out_csv: str,
    seed: int | None = None,
) -> pd.DataFrame:
    """Sample ``samples`` reviews from the pool (seeded by date) into ``out_csv``."""
    df = pd.read_csv(pool_csv)
    if len(df) < samples:
        raise ValueError(
            f"pool has {len(df)} rows, need at least {samples}; pick a smaller --samples"
        )
    rng = np.random.default_rng(seed if seed is not None else _date_seed(date))
    sample = df.sample(n=samples, random_state=rng).copy()
    sample.insert(0, "day", date)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out_csv, index=False)
    logger.info(f"Wrote {len(sample)} reviews to {out_csv}")
    return sample


def score_input(
    input_csv: str,
    out_csv: str,
    scorer: Scorer,
    min_samples: int,
    target_label: str = "label",
) -> dict[str, Any]:
    """Score every text in ``input_csv`` in batches and persist predictions."""
    df = pd.read_csv(input_csv)
    if "text" not in df.columns:
        raise ValueError(
            f"input CSV must have a 'text' column (got {list(df.columns)})"
        )
    n = len(df)
    if n < min_samples:
        raise ValueError(
            f"input has {n} rows; nightly batch should re-score at least "
            f"{min_samples} (got --min-samples {min_samples})"
        )
    t0 = time.perf_counter()
    probs = scorer.score(df["text"].astype(str).tolist())
    elapsed = time.perf_counter() - t0
    cols = scorer.label_columns
    for i, col in enumerate(cols):
        df[col] = probs[:, i]
    arg = np.argmax(probs, axis=1)
    df["predicted_label"] = [cols[int(i)] for i in arg]
    df["predicted_score"] = probs[np.arange(n), arg]
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    logger.info(f"Scored {n} reviews in {elapsed:.2f}s -> {out_csv}")
    return _scoring_stats(df, cols, elapsed, scorer.batch_size, target_label, n)


def _scoring_stats(
    df: pd.DataFrame,
    cols: list[str],
    elapsed: float,
    batch_size: int,
    target_label: str,
    n: int,
) -> dict[str, Any]:
    """Summarise a scored panel, zero-filling every class in ``LABELS``.

    ``value_counts()`` omits a class no row happened to get, which would
    silently drop Neutral from the record on a day the model predicted it never
    occurs. Zero-filled so every day has the same shape.
    """
    counts = df["predicted_label"].value_counts()
    return {
        "n_reviews": n,
        "elapsed_sec": elapsed,
        "batch_size": batch_size,
        "label_columns": cols,
        # Named for what they are: these were counts under a "proportions"
        # key, which is the kind of thing that gets charted as if it were a
        # percentage. Phase 6 step 2 owns the proper share/PSI tracking.
        "predicted_class_counts": {
            label: int(counts.get(label, 0)) for label in LABELS
        },
        "predicted_class_proportions": {
            label: round(float(counts.get(label, 0)) / n, 6) for label in LABELS
        },
        "target_label": target_label,
    }
