"""Attaching the engineered input features to the reference + daily panels.

``engineered_frames`` is where the encoder actually runs, so it carries the two
decisions worth stating out loud:

* **The reference side is cached.** ``reference_engineered.csv`` is the frozen
  comparison; recomputing it nightly would be wasted encoder work *and* would
  silently re-derive the PCA basis if the panel changed, making day-over-day
  scores incomparable. A missing cache file is a warning, not an error: it falls
  back to engineering the reference on the fly (one extra pass) so the check stays
  runnable against a reference built by an older version.
* **The current side is always recomputed and written out**, because that CSV is
  what a human opens when the gate fires.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from raay.inference.builder import DriftFeatureBuilder
from raay.inference.drift_columns import (
    COL_DIALECT_LABEL,
    COL_OOV_BUCKET,
    default_drift_columns,
)
from raay.inference.projection import ProjectionBasis, load_basis

#: The two output-only columns, used when the engineered features are off.
OUTPUT_ONLY_COLUMNS = ("predicted_label", "positive")


@dataclass
class EngineeredDriftSpec:
    """Everything the drift check needs to add the engineered input features.

    Bundled into one object (and injected as ``None`` for the legacy path)
    rather than threaded through six parameters: ``drift_check`` has to keep
    working for a runner that only has the DVC-tracked files.
    """

    builder: DriftFeatureBuilder
    pca_path: str
    reference_engineered: str
    current_engineered: str

    @property
    def n_components(self) -> int:
        return self.builder.n_components


def engineered_frames(
    reference_csv: str,
    current_csv: str,
    spec: EngineeredDriftSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, ProjectionBasis, bool]:
    """Reference/current frames with the engineered columns attached.

    Returns ``(ref, cur, basis, from_cache)``.
    """
    basis = load_basis(spec.pca_path)
    basis.check_compatible(
        model_dir=spec.builder.model_dir,
        max_length=spec.builder.max_length,
        pooling=spec.builder.pooling,
    )
    if Path(spec.reference_engineered).exists():
        ref = pd.read_csv(spec.reference_engineered)
        from_cache = True
    else:
        logger.warning(
            f"{spec.reference_engineered} is missing; engineering the reference "
            f"from {reference_csv} on the fly (one extra encoder pass)"
        )
        ref = spec.builder.engineer(pd.read_csv(reference_csv), basis)
        from_cache = False
    cur = spec.builder.engineer(pd.read_csv(current_csv), basis)
    Path(spec.current_engineered).parent.mkdir(parents=True, exist_ok=True)
    cur.to_csv(spec.current_engineered, index=False)
    logger.info(f"Wrote engineered current panel to {spec.current_engineered}")
    return ref, cur, basis, from_cache


def resolve_columns(
    ref: pd.DataFrame,
    cur: pd.DataFrame,
    labels: tuple[str, str],
    drift_columns: tuple[str, ...] | None,
    engineered: EngineeredDriftSpec | None,
    features_applied: bool,
) -> tuple[list[str], list[str]]:
    """Split the requested columns into the present ones and the absent ones.

    Requested columns that are not in the frames are skipped and named in
    ``skipped_columns`` rather than being silently dropped or crashing the job
    -- a clamped PCA basis legitimately has fewer components than requested.
    """
    if drift_columns is None:
        drift_columns = (
            default_drift_columns(engineered.n_components)
            if features_applied and engineered is not None
            else OUTPUT_ONLY_COLUMNS
        )
    present = [c for c in drift_columns if c in ref.columns and c in cur.columns]
    skipped = [c for c in drift_columns if c not in present]
    if skipped:
        logger.warning(
            f"Skipping {len(skipped)} drift columns absent from the frames: {skipped}"
        )
    if not present:
        reference_csv, current_csv = labels
        raise ValueError(
            f"none of the requested drift columns {list(drift_columns)} are present "
            f"in both {reference_csv} {list(ref.columns)} and {current_csv} "
            f"{list(cur.columns)}"
        )
    return present, skipped


def engineered_block(
    ref: pd.DataFrame,
    cur: pd.DataFrame,
    basis: ProjectionBasis | None,
    reference_from_cache: bool,
    reported_means: tuple[str, ...],
) -> dict[str, Any]:
    """The extra report fields that only exist when the features were applied."""
    from raay.inference.drift_columns import dialect_mix, feature_means

    block: dict[str, Any] = {
        "engineered": True,
        "reference_from_cache": reference_from_cache,
        "feature_means": feature_means(cur, reported_means),
        "oov_bucket_mix": dialect_mix(cur, COL_OOV_BUCKET),
    }
    if basis is not None:
        block["projection"] = basis.summary()
    if COL_DIALECT_LABEL in cur.columns:
        from raay.inference.drift_columns import dialect_total_variation

        block["dialect_mix"] = dialect_total_variation(ref, cur)
    return block
