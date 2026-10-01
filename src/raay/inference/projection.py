"""The frozen PCA projection the embedding drift columns are compared in.

**The projection is frozen, never refit.** A PCA refit per day would rotate the
axes under the comparison and every column would drift for arithmetic reasons.
``fit_basis`` therefore runs once (at ``init-reference`` time) and
``ProjectionBasis`` is persisted with joblib; the nightly job only
``transform``s.

The metadata stored alongside the components is not decoration:
``check_compatible`` refuses to compare a day against a basis that was fitted
with a different ``max_length`` or pooling, because that is a numerically valid
but meaningless comparison, and nothing downstream would flag it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from loguru import logger
from sklearn.decomposition import PCA

from raay.enums.constants import DefaultPaths, Models
from raay.inference.drift_columns import PCS


@dataclass(frozen=True)
class ProjectionBasis:
    """A frozen PCA projection fitted on the reference panel's embeddings."""

    pca: PCA
    n_components: int
    requested_components: int
    explained_variance_ratio: list[float]
    encoder_dir: str
    model_name: str
    max_length: int
    pooling: str
    n_reference: int

    def check_compatible(
        self, *, model_dir: str, max_length: int, pooling: str
    ) -> None:
        mismatches = []
        if self.max_length != max_length:
            mismatches.append(f"max_length {self.max_length} != {max_length}")
        if self.pooling != pooling:
            mismatches.append(f"pooling {self.pooling!r} != {pooling!r}")
        if mismatches:
            raise ValueError(
                "the frozen PCA basis at this path was fitted with different "
                f"preprocessing ({'; '.join(mismatches)}), so the projection would "
                "not match the reference. Re-run `--mode init-reference` to refit it."
            )
        if Path(self.encoder_dir) != Path(model_dir):
            # Warned, not raised: the same encoder can legitimately live at two
            # paths (absolute vs repo-relative, a copy on the staging host).
            logger.warning(
                f"PCA basis was fitted on encoder {self.encoder_dir} but this run "
                f"is using {model_dir}; the two must be the same fine-tuned model"
            )

    def summary(self) -> dict[str, Any]:
        return {
            "n_components": self.n_components,
            "requested_components": self.requested_components,
            "explained_variance_ratio": [
                round(v, 4) for v in self.explained_variance_ratio
            ],
            "explained_variance_total": round(
                float(sum(self.explained_variance_ratio)), 4
            ),
            "encoder_dir": self.encoder_dir,
            "model_name": self.model_name,
            "max_length": self.max_length,
            "pooling": self.pooling,
            "n_reference": self.n_reference,
        }


def fit_basis(
    embeddings: np.ndarray,
    *,
    requested_components: int = PCS,
    model_dir: str = DefaultPaths.BASELINE_MODEL.value,
    model_name: str = Models.TEACHER.value,
    max_length: int = 128,
    pooling: str = "mean",
) -> ProjectionBasis:
    """Fit the PCA basis on reference embeddings, clamped to what they support.

    ``sklearn`` raises if ``n_components`` exceeds ``min(n_samples,
    n_features)``. A short reference panel (the smoke-test sizes) would
    therefore crash the nightly job on a config problem, so the count is
    clamped and the reduction is recorded in ``requested_components`` -- the
    drift column list is then intersected with the columns that exist rather
    than being asked for values that were never computed.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2-D, got shape {embeddings.shape}")
    n_rows, n_dims = embeddings.shape
    n_components = max(1, min(requested_components, n_rows, n_dims))
    if n_components != requested_components:
        logger.warning(
            f"reference has {n_rows} rows x {n_dims} dims; clamping PCA to "
            f"{n_components} components (asked for {requested_components})"
        )
    pca = PCA(n_components=n_components, random_state=0)
    pca.fit(embeddings)
    logger.info(
        f"Fitted PCA basis {n_components} components on {n_rows} reference "
        f"embeddings, explaining {float(pca.explained_variance_ratio_.sum()):.3f} "
        f"of the variance"
    )
    return ProjectionBasis(
        pca=pca,
        n_components=n_components,
        requested_components=requested_components,
        explained_variance_ratio=[float(v) for v in pca.explained_variance_ratio_],
        encoder_dir=model_dir,
        model_name=model_name,
        max_length=max_length,
        pooling=pooling,
        n_reference=n_rows,
    )


def save_basis(basis: ProjectionBasis, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(basis, path)
    logger.info(f"Saved PCA basis to {path}")


def load_basis(path: str) -> ProjectionBasis:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"no frozen PCA basis at {path}. The engineered drift columns cannot be "
            f"compared against a projection that is refit per day (it would rotate "
            f"the axes under the comparison). Run `--mode init-reference` to build it."
        )
    basis = joblib.load(path)
    if not isinstance(basis, ProjectionBasis):
        raise TypeError(f"{path} holds {type(basis).__name__}, not a ProjectionBasis")
    return basis
