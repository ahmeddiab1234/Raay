"""Turns a scored reviews frame into the engineered drift feature frame.

The one object that knows how the columns fit together: embeddings (encoder) ->
frozen PCs (projection) -> the cheap per-text features. It takes its embedder
and tokenizer by injection rather than building them, so the unit tests drive
the whole pipeline with a tiny fake encoder and never touch a real model graph.
Use :meth:`DriftFeatureBuilder.from_config` for the real thing.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from raay.enums.constants import DataColumns, DefaultPaths, Models
from raay.inference.drift_columns import (
    COL_CONFIDENCE,
    COL_DIALECT_LABEL,
    COL_OOV_BUCKET,
    COL_OOV_RATE,
    COL_TEXT_LENGTH,
    EMBEDDING_PREFIX,
    PCS,
)
from raay.inference.encoder import AraBertEmbedder
from raay.inference.projection import ProjectionBasis, fit_basis
from raay.inference.text_features import (
    confidence_scores,
    dialect_labels,
    oov_bucket,
    oov_rates,
    text_lengths,
)


class DriftFeatureBuilder:
    """Turns a scored reviews frame into the engineered drift feature frame."""

    def __init__(
        self,
        embedder: Any,
        tokenizer: Any,
        *,
        n_components: int = PCS,
        model_dir: str = DefaultPaths.BASELINE_MODEL.value,
        model_name: str = Models.TEACHER.value,
        max_length: int = 128,
        preprocess: Callable[[str], str] | None = None,
    ) -> None:
        self.embedder = embedder
        self.tokenizer = tokenizer
        self.n_components = n_components
        self.model_dir = model_dir
        self.model_name = model_name
        self.max_length = max_length
        self.preprocess = preprocess

    @classmethod
    def from_config(
        cls,
        *,
        tokenizer_dir: str = DefaultPaths.BASELINE_MODEL.value,
        model_name: str = Models.TEACHER.value,
        n_components: int = PCS,
        max_length: int = 128,
        pooling: str = "mean",
        batch_size: int = 32,
    ) -> DriftFeatureBuilder:
        embedder = AraBertEmbedder(
            model_dir=tokenizer_dir,
            model_name=model_name,
            max_length=max_length,
            batch_size=batch_size,
            pooling=pooling,
        )
        return cls(
            embedder,
            embedder.tokenizer,
            n_components=n_components,
            model_dir=tokenizer_dir,
            model_name=model_name,
            max_length=max_length,
        )

    @property
    def pooling(self) -> str:
        return getattr(self.embedder, "pooling", "mean")

    def embed(self, frame: pd.DataFrame) -> np.ndarray:
        """Embed a frame's ``text`` column (validated against the row count)."""
        if DataColumns.TEXT.value not in frame.columns:
            raise ValueError(
                f"frame must have a {DataColumns.TEXT.value!r} column to embed "
                f"(got {list(frame.columns)})"
            )
        return self.embedder.embed(frame[DataColumns.TEXT.value].astype(str).tolist())

    def fit_basis(self, embeddings: np.ndarray) -> ProjectionBasis:
        return fit_basis(
            embeddings,
            requested_components=self.n_components,
            model_dir=self.model_dir,
            model_name=self.model_name,
            max_length=self.max_length,
            pooling=self.pooling,
        )

    def engineer(
        self,
        frame: pd.DataFrame,
        basis: ProjectionBasis,
        embeddings: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Return a copy of ``frame`` plus every engineered drift column.

        ``embeddings`` may be passed when the caller already has them (fitting
        the basis needs the same matrix), which halves the encoder passes.
        """
        out = frame.copy()
        texts = out[DataColumns.TEXT.value].astype(str).tolist()
        if embeddings is None:
            embeddings = self.embedder.embed(texts)
        if embeddings.shape[0] != len(out):
            raise ValueError(
                f"got {embeddings.shape[0]} embeddings for {len(out)} rows; the "
                f"embedding and feature frames must be row-aligned"
            )
        pcs = basis.pca.transform(embeddings)
        if pcs.shape[1] != basis.n_components:
            raise ValueError(
                f"basis projects to {basis.n_components} components but the "
                f"transform returned {pcs.shape[1]}"
            )
        for i in range(pcs.shape[1]):
            out[f"{EMBEDDING_PREFIX}{i + 1}"] = pcs[:, i]
        out[COL_TEXT_LENGTH] = text_lengths(texts)
        out[COL_CONFIDENCE] = confidence_scores(out)
        out[COL_OOV_RATE] = oov_rates(
            texts, self.tokenizer, self.model_name, preprocess=self.preprocess
        )
        out[COL_OOV_BUCKET] = [oov_bucket(v) for v in out[COL_OOV_RATE]]
        out[COL_DIALECT_LABEL] = dialect_labels(texts)
        return out
