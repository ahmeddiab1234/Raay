"""Build the frozen drift reference (and the PCA basis fitted on it).

``--mode init-reference`` is a setup step, not a nightly one: the reference
panel and the projection it implies are frozen on purpose, because refitting
either per day would rotate the axes (or the baseline) under the comparison.
The Airflow DAG runs the other modes.

With the engineered features on, this is also the only place ``fit_basis`` is
ever called, and the only place the reference embeddings are written.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from raay.inference.batch_scoring import Scorer, make_input, score_input
from raay.inference.drift_columns import COL_OOV_RATE, COL_TEXT_LENGTH, feature_means
from raay.inference.drift_engine import EngineeredDriftSpec
from raay.inference.projection import save_basis

#: Means logged for the reference so a day can be compared against it directly.
REFERENCE_MEANS = (COL_TEXT_LENGTH, COL_OOV_RATE)


def init_reference(
    pool_csv: str,
    samples: int,
    out_csv: str,
    scorer: Scorer,
    engineered: EngineeredDriftSpec | None = None,
) -> pd.DataFrame:
    """Score a fixed slice of the pool once; the drift reference for all days.

    With ``engineered``, also embeds the reference, fits the PCA basis on those
    embeddings and writes the engineered reference panel. The basis is fitted
    here and only here -- it is the reference distribution's projection, and
    refitting it per day would rotate the axes under the comparison.
    """
    make_input(pool_csv, "reference", samples, out_csv, seed=0)
    score_input(out_csv, out_csv, scorer, min_samples=1)
    frame = pd.read_csv(out_csv)
    if engineered is None:
        return frame
    embeddings = engineered.builder.embed(frame)
    basis = engineered.builder.fit_basis(embeddings)
    save_basis(basis, engineered.pca_path)
    featured = engineered.builder.engineer(frame, basis, embeddings)
    Path(engineered.reference_engineered).parent.mkdir(parents=True, exist_ok=True)
    featured.to_csv(engineered.reference_engineered, index=False)
    logger.info(
        f"Wrote engineered reference ({len(featured)} rows, "
        f"{basis.n_components} PCs) to {engineered.reference_engineered}"
    )
    return featured


def reference_artifacts(
    frame: pd.DataFrame, out_csv: str, engineered: EngineeredDriftSpec | None
) -> tuple[dict[str, float], list[tuple[str, str]]]:
    """MLflow metrics and artifact pairs for an ``init-reference`` run."""
    metrics: dict[str, float] = {"n_reference": float(len(frame))}
    artifacts: list[tuple[str, str]] = [("reference", out_csv)]
    if engineered is not None:
        artifacts += [
            ("reference_engineered", engineered.reference_engineered),
            ("pca_basis", engineered.pca_path),
        ]
        metrics |= {
            f"reference_mean_{col}": value
            for col, value in feature_means(frame, REFERENCE_MEANS).items()
        }
    return metrics, artifacts
