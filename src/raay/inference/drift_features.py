"""Engineered *input* drift features for the nightly monitoring job.

Phase 6 step 1. ``batch_score.drift_check`` already compares the model's own
outputs (``predicted_label``, per-class probabilities) against a frozen
reference. That is **output** drift: it can only tell you the model changed
*after* its inputs changed, and on this corpus the reference and the day are
slices of the same pool, so it reads PASS by construction.

This package adds the input side, which is what actually moves first:

- ``embedding_pc1..embedding_pcN`` -- the review encoded by the **fine-tuned**
  AraBERT encoder, projected onto a PCA basis **frozen on the reference
  panel**. Raw 768-dim PSI is not interpretable: Evidently bins numerics on 30
  reference quantiles, so 768 columns means 768 mostly-empty bin comparisons
  and a report nobody reads. 10-20 dims is the standard reduction and 10 is
  the default here.
- ``oov_rate`` -- fraction of AraBERT subword tokens that fall back to
  ``[UNK]``. This is the *earliest* signal available: ``[UNK]`` is a hard
  boundary, so new slang or transliterated foreign words show up here before
  the encoder has smeared them into their nearest neighbours.
- ``dialect_label`` -- the heuristic MSA/Egyptian/Gulf/Levantine/Maghrebi/
  Arabizi mix, so a seasonal vocabulary shift (Ramadan, a campaign that only
  runs in one region) is legible as a proportion move.
- ``text_length`` and ``confidence_score`` -- the cheap shape signals.

Two deliberate choices worth knowing before changing anything here:

**The projection is frozen, never refit.** See :mod:`raay.inference.projection`.

**Dialect labels are recomputed from ``text``, never read from the stored
``dialect`` column.** ``Dialects`` is a ``str, Enum``, so
``str(Dialects.ARABIZI) == "Dialects.ARABIZI"``: a frame round-tripped
through CSV carries that prefix (the same quirk that puts
``"Dialects.ARABIZI"`` keys in ``reports/split_metrics.json``). Recomputing
gives clean, canonical labels on both sides of the comparison.

Layout, cheapest-first:

* :mod:`raay.inference.drift_columns` -- column names, the gated column list
  and the aggregate summaries (mixes, means).
* :mod:`raay.inference.text_features` -- per-text OOV/dialect/length/confidence.
* :mod:`raay.inference.encoder` -- the AraBERT pooled embedder (the expensive
  half; only ``init-reference`` and ``drift`` should ever build one).
* :mod:`raay.inference.projection` -- the frozen PCA basis.
* :mod:`raay.inference.builder` -- ``DriftFeatureBuilder``, which composes them.

This module stays the public surface: everything the drift job and the tests
import is re-exported here.
"""

from __future__ import annotations

from raay.inference.builder import DriftFeatureBuilder
from raay.inference.drift_columns import (
    COL_CONFIDENCE,
    COL_DIALECT_LABEL,
    COL_OOV_BUCKET,
    COL_OOV_RATE,
    COL_TEXT_LENGTH,
    EMBEDDING_PREFIX,
    PCS,
    default_drift_columns,
    dialect_mix,
    dialect_total_variation,
    embedding_pc_columns,
    feature_means,
)
from raay.inference.encoder import AraBertEmbedder, encoder_weights_present
from raay.inference.projection import (
    ProjectionBasis,
    fit_basis,
    load_basis,
    save_basis,
)
from raay.inference.text_features import (
    confidence_scores,
    dialect_labels,
    oov_bucket,
    oov_rates,
    text_lengths,
)

__all__ = [
    "COL_CONFIDENCE",
    "COL_DIALECT_LABEL",
    "COL_OOV_BUCKET",
    "COL_OOV_RATE",
    "COL_TEXT_LENGTH",
    "EMBEDDING_PREFIX",
    "PCS",
    "AraBertEmbedder",
    "DriftFeatureBuilder",
    "ProjectionBasis",
    "confidence_scores",
    "default_drift_columns",
    "dialect_labels",
    "dialect_mix",
    "dialect_total_variation",
    "embedding_pc_columns",
    "encoder_weights_present",
    "feature_means",
    "fit_basis",
    "load_basis",
    "oov_bucket",
    "oov_rates",
    "save_basis",
    "text_lengths",
]
