"""Schemas, statuses, routes and config for the customer-service feedback loop.

Every column list here is part of a file contract: ``REVIEWED_COLUMNS`` must
match the capture sink's schema so a raw file and a reviewed file diff cleanly,
and ``MERGED_COLUMNS`` starts from ``train.csv``'s schema so the sidecar can be
concatenated without touching ``train.py``'s column assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

# Mirrors split.py: the CI drift gate (+/-0.005) compares stable values, not full
# float reprs.
_PROPORTION_DP = 5

STATUS_CONFIRMATION = "confirmation"
STATUS_CORROBORATED = "corroborated"
STATUS_DISPUTED = "disputed"
STATUS_ADJUDICATED = "adjudicated"
STATUS_SINGLE_AGENT = "single_agent"
STATUS_LEAK = "leak"
STATUS_DUPLICATE = "duplicate"
STATUS_NEAR_EMPTY = "near_empty"

#: Statuses whose rows may enter the training merge. Confirmations are excluded
#: unconditionally -- they are measurements, not labels to learn from.
TRAINABLE_STATUSES: tuple[str, ...] = (STATUS_CORROBORATED, STATUS_ADJUDICATED)

ALL_STATUSES: tuple[str, ...] = (
    STATUS_CORROBORATED,
    STATUS_ADJUDICATED,
    STATUS_DISPUTED,
    STATUS_SINGLE_AGENT,
    STATUS_CONFIRMATION,
    STATUS_LEAK,
    STATUS_DUPLICATE,
    STATUS_NEAR_EMPTY,
)

#: Columns of ``data/feedback/reviewed/overrides.csv``. The first eleven are the
#: capture sink's schema, so a raw file and a reviewed file diff cleanly.
REVIEWED_COLUMNS: tuple[str, ...] = (
    "override_id",
    "captured_at",
    "text",
    "model_label",
    "corrected_label",
    "agent_id",
    "model_score",
    "model_version",
    "company",
    "note",
    "guideline_version",
    "second_agent_id",
    "corroborating_agents",
    "second_label",
    "adjudicator_id",
    "adjudicated_label",
    "status",
    "route",
    "is_correction",
)

#: The merged output reuses train.csv's schema first -- so the file can be
#: concatenated without touching train.py's column assumptions -- then adds
#: provenance. ``model_label`` is the load-bearing one: it is what makes these
#: rows hard negatives rather than merely new examples.
TRAIN_BASE_COLUMNS: tuple[str, ...] = (
    "text",
    "label",
    "company",
    "is_near_empty",
    "dialect",
    "dialect_confidence",
)
MERGED_COLUMNS: tuple[str, ...] = TRAIN_BASE_COLUMNS + (
    "source",
    "guideline_version",
    "model_label",
    "model_version",
    "captured_at",
    "override_id",
    "qa_status",
    "corroborated_by",
)

RAW_REQUIRED: tuple[str, ...] = ("text", "model_label", "corrected_label", "agent_id")

#: Routes for statuses that cannot train, straight from :func:`classify_row`. They
#: name what the row still *needs*; an operator triaging the reviewed file reads
#: these to know whether to chase a second agent or an adjudicator.
ROUTES_NEEDS_SECOND_AGENT = "needs_second_agent"
ROUTES_NEEDS_ADJUDICATOR = "needs_adjudicator"
ROUTE_METRICS_ONLY = "metrics_only"
ROUTE_ADJUDICATE_FIRST = "adjudicate_first"
ROUTE_TRAIN = "train"


@dataclass
class FeedbackConfig:
    """The ``feedback`` block of ``params.yaml``, with defaults."""

    min_corroborating_agents: int = 2
    suspicious_model_score: float = 0.9
    max_neutral_per_batch: int = 50
    min_char_length: int = 10

    @classmethod
    def from_params(cls, params: dict[str, Any] | None = None) -> FeedbackConfig:
        params = params or {}
        block = params.get("feedback", {}) or {}
        # min_char_length falls back to the preprocessing value so the emptiness
        # rule cannot drift between the two pipelines.
        fallback = params.get("preprocessing", {}).get("min_char_length", 10)
        return cls(
            min_corroborating_agents=int(
                block.get("min_corroborating_agents", cls.min_corroborating_agents)
            ),
            suspicious_model_score=float(
                block.get("suspicious_model_score", cls.suspicious_model_score)
            ),
            max_neutral_per_batch=int(
                block.get("max_neutral_per_batch", cls.max_neutral_per_batch)
            ),
            min_char_length=int(block.get("min_char_length", fallback)),
        )


@dataclass
class ReviewResult:
    """Outcome of :func:`review_rows`: the reviewed frame plus status counts."""

    frame: pd.DataFrame
    counts: dict[str, int] = field(default_factory=dict)
