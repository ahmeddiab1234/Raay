"""Request schema and on-disk column list for the feedback capture sink.

The endpoint takes **string labels only** and validates them against
:data:`raay.enums.constants.LABELS` -- the id2label order every graph in the repo
carries. An integer label is rejected rather than coerced: accepting one would
mean guessing which convention the caller meant, and the two conventions in this
repo's history differ.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime

from pydantic import BaseModel, field_validator

from raay.enums.constants import LABELS

#: Restricted to characters that cannot break the append-only CSV: an
#: ``agent_id`` is written into a row, and a comma or newline in it would forge a
#: second record. It also keys the corroboration groups in
#: ``raay.data.feedback``, so it doubles as the roster identity.
AGENT_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

#: The on-disk schema of ``data/feedback/raw/{date}.csv``. A module constant
#: rather than something derived from the pydantic model, because the sink and
#: the review stage must agree on it, and a field-ordering change derived from
#: the model would silently rewrite historical files instead of appending to
#: them.
RAW_COLUMNS: tuple[str, ...] = (
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
)


class FeedbackOverride(BaseModel):
    """One agent's assertion about one review.

    ``model_label`` is what the model said when the agent looked at it. It is
    recorded rather than recomputed, because the graph that produced it need not
    be the one running when the row is reviewed months later.
    """

    text: str
    model_label: str
    corrected_label: str
    agent_id: str
    captured_at: datetime | None = None
    model_score: float | None = None
    model_version: str = ""
    company: str = ""
    note: str = ""
    guideline_version: str = "v1.0"

    @field_validator("model_label", "corrected_label")
    @classmethod
    def _known_label(cls, value: str) -> str:
        # Deliberately `LABELS` -- the id2label order every ONNX graph carries
        # (configs/train.yaml: labels [positive, negative, neutral]) -- and not
        # the encoding in docs/labeling_guidelines.md section 1, which states
        # positive=2/negative=0/neutral=1 and is inverted relative to every
        # graph in the repo. Following the doc here flips every label silently.
        if value not in LABELS:
            raise ValueError(
                f"label must be one of {list(LABELS)} (the id2label order the graphs "
                f"use), got {value!r}"
            )
        return value

    @field_validator("agent_id")
    @classmethod
    def _safe_agent_id(cls, value: str) -> str:
        if not AGENT_ID_RE.match(value):
            raise ValueError(
                "agent_id must match [A-Za-z0-9._-]{1,64}: it is written into an "
                "append-only CSV and keys the corroboration groups"
            )
        return value

    @field_validator("text")
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be empty or whitespace")
        return value

    @property
    def is_correction(self) -> bool:
        """False when the agent confirmed the model instead of disputing it."""
        return self.model_label != self.corrected_label

    def override_id(self) -> str:
        """Content hash identifying this assertion.

        ``captured_at`` is excluded so a retried POST is idempotent: a CS tool
        that times out and retries must not produce two rows for one assertion.
        The tradeoff is deliberate and real -- a genuine second correction by the
        *same* agent on the same text with the same labels collapses into one
        row. Distinct agents still get distinct ids, which is what the
        two-annotator corroboration rule depends on.
        """
        parts = (
            self.text.strip(),
            self.model_label,
            self.corrected_label,
            self.agent_id,
        )
        return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]
