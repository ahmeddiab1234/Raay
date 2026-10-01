"""Customer-service feedback: QA review and train-set merge (Phase 6 step 4).

The capture endpoint (``raay.serving.feedback_service``) writes raw assertions to
``data/feedback/raw/{date}.csv``. This module turns them into training data in
two steps, deliberately separated:

``--mode review``
    Raw rows -> ``data/feedback/reviewed/overrides.csv`` with a ``status`` per
    row. **Not a DVC stage**: it reads a directory a running service appends to,
    which is not a dependency DVC can hash. An operator runs it, checks the
    disputed rows, then ``dvc add`` + ``dvc push`` the result -- the same ritual
    as ``data/raw/Final_Data.csv``.

``--mode merge``
    Reviewed rows -> ``data/processed/train_feedback.csv``. This one *is* a DVC
    stage, and it is the only piece that touches training data.

Why the merge is a train-only sidecar
-------------------------------------

Feedback must not re-enter ``data/interim/normalized.csv``. If it did, ``split``
would re-run, ``data/processed/test.csv`` would change, and
``scripts/promote_model.py:check_frozen_split`` would exit 2 with no report --
taking the whole 14-gate promotion story down with it. So merged rows go to their
own file that ``train.py`` concatenates onto **train only**. The frozen test
split never moves, and ``promote_model.py`` never reads ``train.csv`` anyway.

Why an override is not ground truth
-----------------------------------

A support agent's correction is a human opinion formed under time pressure during
a dispute, and disputes are frequently about shipping, refunds or seller conduct
rather than sentiment -- exactly the scope violation
``docs/labeling_guidelines.md`` section 2 warns about. The brief is right that a
mislabeled override is worse than no label, so nothing reaches the training set on
a single assertion:

1. two distinct agents must agree on the same text, or
2. an adjudicator must have ruled (``adjudicated_label`` wins over both).

That mirrors the 2-annotator + adjudicator process in section 4 of the
guidelines, applied to a population where the second annotator is cheap because
the text already exists.

**Operational prerequisite, not a bug:** corroboration needs an agent roster with
at least two real identities. Until one exists, ``--mode review`` legitimately
returns 100% ``single_agent`` and the merge accepts nothing.

What this measures that nothing else can
----------------------------------------

``model_label`` on a production review is a *free* labelled error record. Every
other Phase 6 signal is a seeded draw from ``data/processed/test.csv`` and says
so in its own ``caveat`` field; this is actual traffic. The
``production_error_rate`` block of ``reports/feedback_metrics.json`` is therefore
the only production-measured error rate in the project -- and it is only
interpretable if confirmations are posted too, since they are the denominator.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from rapidfuzz import fuzz, process

from raay.config.env import load_environment
from raay.data.dialect import add_dialect_column
from raay.data.preprocess import (
    collapse_elongation,
    flag_near_empty,
    normalize_casing,
    remove_diacritics,
)
from raay.enums.constants import LABELS, DefaultPaths, Experiments

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


# --------------------------------------------------------------------------
# text handling
# --------------------------------------------------------------------------


def normalize_text(text: Any, elongation_max_repeat: int = 2) -> str:
    """Apply the *same* normalization the training corpus got.

    Not optional, and applied *first* rather than at merge time. Two reasons:

    * ``data/processed/train.csv`` texts went through diacritic stripping,
      elongation collapsing and casing normalization in ``raay.data.preprocess``;
      raw text from a CS tool has not. Writing both into one training file puts a
      distribution artifact in front of the tokenizer and the model learns the
      artifact rather than the sentiment.
    * Corroboration, the leakage guard and the near-empty check then all compare
      like with like. Two agents posting the same review, one with tashkeel and
      one without, are one review -- and a test-split row only matches its
      feedback twin if both sides are normalized first.
    """
    return normalize_casing(
        collapse_elongation(remove_diacritics(text), elongation_max_repeat)
    )


def text_key(text: Any) -> str:
    """Whitespace-insensitive key for "the same review" comparisons."""
    return " ".join(normalize_text(text).split())


def coerce_label(value: Any) -> str:
    """A label in :data:`LABELS`, else ``""``."""
    text = str(value).strip()
    return text if text in LABELS else ""


def numeric_or_none(value: Any) -> float | None:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        text = str(value).strip()
        return float(text) if text else None
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------
# the QA ladder
# --------------------------------------------------------------------------


#: Routes for statuses that cannot train, straight from :func:`classify_row`. They
#: name what the row still *needs*; an operator triaging the reviewed file reads
#: these to know whether to chase a second agent or an adjudicator.
ROUTES_NEEDS_SECOND_AGENT = "needs_second_agent"
ROUTES_NEEDS_ADJUDICATOR = "needs_adjudicator"
ROUTE_METRICS_ONLY = "metrics_only"
ROUTE_ADJUDICATE_FIRST = "adjudicate_first"
ROUTE_TRAIN = "train"


def corroboration_groups(
    frame: pd.DataFrame, min_agents: int = 2
) -> dict[str, dict[str, Any]]:
    """Per-text summary: the agents that spoke, the label they agreed on, and any
    adjudication.

    ``agreed_label`` is ``""`` when no label reaches ``min_agents`` distinct
    agents. That emptiness is the signal a row needs an adjudicator -- not
    something to break with a plurality rule over *three-way* splits, which would
    silently invent a label nobody defended.

    A text is corroborated when at least ``min_agents`` **distinct** agents assert
    the same label. Counting rows would be wrong: an agent who accidentally
    double-posts one override must not corroborate themselves.

    ``adjudicated_label`` is group-scoped. A human adjudicates a *review*, not one
    assertion about it, so a ruling on any row of the group resolves the whole
    group and is propagated onto every row -- otherwise a two-row group where only
    the second row carries the ruling would merge the ruling once and the bare
    corroboration once, i.e. two training rows for one decision. Conflicting
    rulings are treated as no ruling at all, which leaves the rows in dispute where
    a human still has to look.
    """
    groups: dict[str, dict[str, Any]] = {}
    for key, group in frame.groupby("_key", sort=False):
        # agent -> set of labels that agent asserted for this text. An agent that
        # asserts two different labels is self-contradictory and contributes
        # neither: counting it as a vote either way would manufacture agreement.
        by_agent: dict[str, set[str]] = {}
        for agent, label in zip(group["agent_id"], group["_label"], strict=True):
            agent = str(agent).strip()
            label = coerce_label(label)
            if not agent or not label:
                continue
            by_agent.setdefault(agent, set()).add(label)
        consistent = {
            agent: next(iter(labels))
            for agent, labels in by_agent.items()
            if len(labels) == 1
        }

        label_votes: dict[str, set[str]] = {}
        for label in set(consistent.values()):
            label_votes[label] = {
                agent for agent, value in consistent.items() if value == label
            }
        agreed = next(
            (
                label
                for label, voters in label_votes.items()
                if len(voters) >= min_agents
            ),
            "",
        )
        dissent = (
            {label for label in label_votes if label != agreed}
            if agreed
            else set(label_votes)
        )

        rulings = {
            coerce_label(r)
            for r in group.get("adjudicated_label", pd.Series(dtype=str))
            if coerce_label(r)
        }
        adjudicated = rulings.pop() if len(rulings) == 1 else ""

        groups[str(key)] = {
            "agents": sorted(by_agent),
            "agreed_label": agreed,
            "adjudicated_label": adjudicated,
            "n_rows": len(group),
            "n_consistent_agents": len(consistent),
            "voters": label_votes.get(agreed, set()) if agreed else set(),
            "dissenting_labels": sorted(dissent),
        }
    return groups


def classify_row(
    row: pd.Series,
    group: dict[str, Any],
    config: FeedbackConfig,
) -> tuple[str, str]:
    """``(status, route)`` for one assertion.

    Order is load-bearing. A confirmation is settled before corroboration is
    considered, because a confirmation's ``corrected_label`` *is* the model's own
    prediction: two agents agreeing that the model was right is evidence about
    nothing, and treating it as a corroborated label would put the model's own
    output into the training set.
    """
    is_correction = bool(row.get("is_correction", True))

    if not is_correction:
        return STATUS_CONFIRMATION, ROUTE_METRICS_ONLY
    if group["adjudicated_label"]:
        return STATUS_ADJUDICATED, ROUTE_TRAIN
    if group["agreed_label"]:
        return STATUS_CORROBORATED, ROUTE_TRAIN
    if group["n_consistent_agents"] >= config.min_corroborating_agents:
        return STATUS_DISPUTED, ROUTES_NEEDS_ADJUDICATOR
    return STATUS_SINGLE_AGENT, ROUTES_NEEDS_SECOND_AGENT


def route_for_confidence(
    status: str,
    model_score: Any,
    config: FeedbackConfig,
    base_route: str = ROUTE_TRAIN,
) -> str:
    """Escalate a trainable row's route to ``adjudicate_first`` if the model was confident.

    Routing, never a gate. A model that was confident and wrong is exactly the
    hard case this loop exists to collect, so rejecting on confidence would
    discard the valuable signal -- and a wrong-but-confident correction is still
    a genuine QA candidate, just one worth a human eye before it trains.

    ``base_route`` is the row's own route from :func:`classify_row` and is returned
    untouched for anything that cannot train. Relabelling a ``disputed`` row
    ``train`` (which this did) erased the one instruction that tells an operator
    the row needs an adjudicator, and the reviewed file is the human's only view.
    """
    if status not in TRAINABLE_STATUSES:
        return base_route
    score = numeric_or_none(model_score)
    if score is not None and score >= config.suspicious_model_score:
        return ROUTE_ADJUDICATE_FIRST
    return ROUTE_TRAIN


def find_test_overlap(frame: pd.DataFrame, test_csv: str, threshold: float) -> set[str]:
    """Keys in ``frame`` that fuzzy-match the frozen test split.

    A feedback row that duplicates a test-split review would be train-on-test.
    ``scripts/promote_model.py`` hashes ``data/processed/test.csv`` but cannot
    know the *labels* were also seen, so this is the only thing standing between
    a repeated production review and an inflated eval number. Reuses
    ``preprocessing.dedup_similarity_threshold`` and the same scorer
    ``preprocess.fuzzy_deduplicate`` uses.
    """
    if frame.empty or not Path(test_csv).exists():
        return set()
    test_texts = pd.read_csv(test_csv)["text"].astype(str).tolist()
    cutoff = threshold * 100 if threshold <= 1.0 else threshold
    return {
        str(key)
        for key in frame["_key"].unique()
        if process.extractOne(
            str(key), test_texts, scorer=fuzz.ratio, score_cutoff=cutoff
        )
        is not None
    }


# --------------------------------------------------------------------------
# --mode review
# --------------------------------------------------------------------------


def read_raw(raw_dir: str) -> pd.DataFrame:
    """Concatenate every ``data/feedback/raw/*.csv``.

    An empty directory yields an empty frame rather than an error: before any
    customer service exists there is nothing to review, and that is a normal
    state for the nightly job to be in.
    """
    paths = sorted(Path(raw_dir).glob("*.csv"))
    if not paths:
        return pd.DataFrame(columns=list(REVIEWED_COLUMNS))
    frames = [pd.read_csv(path, dtype=str, keep_default_na=False) for path in paths]
    raw = pd.concat(frames, ignore_index=True)
    missing = [column for column in RAW_REQUIRED if column not in raw.columns]
    if missing:
        raise ValueError(
            f"{raw_dir} is missing required column(s) {missing}; the capture sink "
            f"writes {list(REVIEWED_COLUMNS[:11])}"
        )
    return raw


def review_rows(
    raw: pd.DataFrame,
    config: FeedbackConfig,
    test_csv: str = DefaultPaths.TEST_SPLIT.value,
    dedup_threshold: float = 0.9,
    already_merged: pd.DataFrame | None = None,
    elongation_max_repeat: int = 2,
) -> ReviewResult:
    """Apply the QA ladder; return the reviewed frame and the status counts."""
    if raw.empty:
        return ReviewResult(frame=raw.copy(), counts={})

    frame = raw.copy()
    # Filled by hand downstream, and a CSV round trip must not turn an absent
    # corroborating agent into the string "nan".
    for column in (
        "model_score",
        "second_agent_id",
        "second_label",
        "adjudicator_id",
        "adjudicated_label",
        "override_id",
        "captured_at",
        "model_version",
        "company",
        "note",
        "guideline_version",
    ):
        if column not in frame.columns:
            frame[column] = ""

    # Normalize first: every comparison below then runs on the same footing as
    # the training corpus and the test split (see normalize_text).
    frame["text"] = frame["text"].map(
        lambda t: normalize_text(t, elongation_max_repeat)
    )
    frame["model_label"] = frame["model_label"].map(coerce_label)
    frame["corrected_label"] = frame["corrected_label"].map(coerce_label)
    frame["_key"] = frame["text"].map(lambda t: " ".join(str(t).split()))
    frame["is_correction"] = frame["corrected_label"] != frame["model_label"]
    # Resolved once, on the coerced values, so corroboration cannot be fooled by
    # a label the ladder has already rejected as out-of-vocabulary.
    frame["_label"] = frame["corrected_label"]

    groups = corroboration_groups(frame, config.min_corroborating_agents)
    statuses: list[str] = []
    routes: list[str] = []
    second_agents: list[str] = []
    corroborating: list[str] = []
    adjudicated_labels: list[str] = []
    adjudicators: list[str] = []
    for _, row in frame.iterrows():
        group = groups[str(row["_key"])]
        status, route = classify_row(row, group, config)
        statuses.append(status)
        routes.append(
            route_for_confidence(status, row.get("model_score"), config, route)
        )
        voters = sorted(
            str(a).strip()
            for a in group["voters"]
            if str(a).strip() != str(row["agent_id"]).strip()
        )
        second_agents.append(voters[0] if voters else "")
        # Every agent that defended the agreed label, so the merge can record the
        # full set on the single row it keeps.
        corroborating.append(";".join(sorted(group["voters"])))
        # Propagate the group's ruling onto every row, so the reviewed CSV records
        # the resolution once per assertion and `build_merged` can read it without
        # re-running the ladder.
        adjudicated_labels.append(group["adjudicated_label"])
        adjudicators.append(
            str(row.get("adjudicator_id", "")).strip()
            if group["adjudicated_label"]
            else ""
        )
    frame["status"] = statuses
    frame["route"] = routes
    frame["second_agent_id"] = second_agents
    frame["corroborating_agents"] = corroborating
    frame["adjudicated_label"] = adjudicated_labels
    frame["adjudicator_id"] = adjudicators
    frame["second_label"] = frame["_key"].map(
        lambda key: groups[str(key)]["agreed_label"]
    )

    # Exclusions run *after* the ladder so the counters still show how many
    # assertions were disputed, not merely how many rows survived: an override
    # rejected for overlapping the test split is still a production error.
    #
    # Precedence matters. Each exclusion overwrites the previous one, so the order
    # is the priority ladder: a row that is both a leak and a duplicate must report
    # as a leak (train-on-test is the serious one, and a duplicate is the milder
    # explanation), while a near-empty row still reports as near-empty because
    # that is why it can never be trusted.
    leaked = find_test_overlap(frame, test_csv, dedup_threshold)
    if leaked:
        frame.loc[frame["_key"].isin(leaked), "status"] = STATUS_LEAK

    # Rows merged by an earlier run are duplicates of work already done -- this is
    # what makes merge_reviewed idempotent after a crash.
    duplicate = pd.Series(False, index=frame.index)
    if (
        already_merged is not None
        and not already_merged.empty
        and "text" in already_merged
    ):
        prior = {_text_key_only(t) for t in already_merged["text"]}
        duplicate = frame["_key"].isin(prior)
    frame.loc[duplicate, "status"] = STATUS_DUPLICATE

    frame.loc[
        frame["text"].map(lambda t: flag_near_empty(t, config.min_char_length)),
        "status",
    ] = STATUS_NEAR_EMPTY

    reviewed = frame.drop(columns=["_key", "_label"])
    counts = {
        status: int((reviewed["status"] == status).sum()) for status in ALL_STATUSES
    }
    return ReviewResult(frame=reviewed, counts=counts)


def _text_key_only(text: Any) -> str:
    """Whitespace-collapsed key for text already known to be normalized."""
    return " ".join(str(text).split())


def write_reviewed(result: ReviewResult, out_csv: str) -> pd.DataFrame:
    frame = result.frame.copy()
    for column in REVIEWED_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    ordered = frame[list(REVIEWED_COLUMNS)].copy()
    ordered["is_correction"] = ordered["is_correction"].map(
        lambda v: str(bool(v)).lower() if v not in ("", None) else ""
    )
    # Sorted by a content hash, not by capture time: a row's position must not
    # depend on the order files happened to be read in, because this file is
    # git-tracked and DVC-hashed.
    ordered = ordered.sort_values("override_id", kind="stable").reset_index(drop=True)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    ordered.to_csv(out_csv, index=False)
    return ordered


# --------------------------------------------------------------------------
# --mode merge
# --------------------------------------------------------------------------


def build_merged(
    reviewed: pd.DataFrame,
    config: FeedbackConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Turn reviewed rows into train-shaped rows.

    Returns the frame and a summary recording anything suppressed, so a cap that
    ate half the batch shows up in the report instead of being inferred from a
    mysteriously smaller file.
    """
    summary: dict[str, Any] = {
        "n_candidates": len(reviewed),
        "n_eligible": 0,
        "n_accepted": 0,
        "n_collapsed_to_one_row_per_review": 0,
        "n_suppressed_by_cap": 0,
        "neutral_cap": config.max_neutral_per_batch,
        "accepted_label_proportions": {label: 0.0 for label in LABELS},
    }
    if reviewed.empty:
        return pd.DataFrame(columns=list(MERGED_COLUMNS)), summary

    eligible = reviewed[reviewed["status"].isin(TRAINABLE_STATUSES)].copy()
    summary["n_eligible"] = len(eligible)
    if eligible.empty:
        return pd.DataFrame(columns=list(MERGED_COLUMNS)), summary

    # Resolve the adjudicated label *before* the cap, so the cap is applied to the
    # label the row will actually train with. Capping on `corrected_label` would let
    # adjudicated-to-neutral rows through uncapped and defeat the whole point.
    adjudicated = eligible["adjudicated_label"].map(coerce_label)
    eligible["_final_label"] = np.where(
        adjudicated != "", adjudicated, eligible["corrected_label"]
    )

    # One review, one training row.
    #
    # Two agents corroborating a single review produce two assertion rows, and both
    # carry the same text and the same label. Emitting both would put an identical
    # (text, label) pair into train.csv twice: the reviewer would weight that one
    # hard negative 2x for no informational reason, and a review five agents flagged
    # would weight 5x. The corroboration is *evidence for the label*, recorded in
    # `corroborating_agents` -- it is not extra training data. Deduping here also
    # keeps the Neutral cap counting reviews rather than votes.
    before = len(eligible)
    eligible = eligible.drop_duplicates(subset=["text"], keep="first")
    summary["n_collapsed_to_one_row_per_review"] = before - len(eligible)

    # Deterministic order so the cap takes the same rows on every re-run. A cap
    # that picked arbitrarily would make the output non-reproducible, which is
    # exactly what `git diff --exit-code dvc.lock` exists to catch.
    eligible = eligible.sort_values(["captured_at", "override_id"], kind="stable")

    if config.max_neutral_per_batch >= 0:
        neutral_idx = eligible.index[eligible["_final_label"] == "neutral"]
        overflow = neutral_idx[config.max_neutral_per_batch :]
        if len(overflow):
            summary["n_suppressed_by_cap"] = len(overflow)
            eligible = eligible.drop(index=overflow)

    if eligible.empty:
        return pd.DataFrame(columns=list(MERGED_COLUMNS)), summary

    merged = pd.DataFrame(
        {
            # The adjudicated label wins: it is the resolution of a dispute the
            # two agents could not settle. falls back to the agreed label.
            "label": eligible["_final_label"],
            "text": eligible["text"],
            "company": eligible["company"],
            "model_label": eligible["model_label"],
            "model_version": eligible["model_version"],
            "captured_at": eligible["captured_at"],
            "override_id": eligible["override_id"],
            "qa_status": eligible["status"],
            "corroborated_by": eligible["corroborating_agents"],
        }
    )
    merged["source"] = "customer_service_feedback"
    merged["guideline_version"] = eligible["guideline_version"]
    # Never true: review_rows excluded these rows. Carrying the column (rather
    # than dropping it) keeps the schema identical to train.csv.
    merged["is_near_empty"] = False

    merged = add_dialect_column(merged, text_col="text")
    ordered = merged[list(MERGED_COLUMNS)]
    summary["n_accepted"] = len(ordered)
    summary["accepted_label_proportions"] = label_proportions(ordered["label"])
    return ordered, summary


def label_proportions(series: pd.Series) -> dict[str, float]:
    """Value counts as fractions of the total, zero-filled over :data:`LABELS`.

    Zero-filled so the report's shape is stable across days: an absent class is a
    fact about this batch, not a missing key.
    """
    counts = series.value_counts()
    total = float(counts.sum())
    if total == 0:
        return {label: 0.0 for label in LABELS}
    return {
        label: round(float(counts.get(label, 0)) / total, _PROPORTION_DP)
        for label in LABELS
    }


def merge_reviewed(
    reviewed_csv: str,
    out_csv: str,
    config: FeedbackConfig,
) -> dict[str, Any]:
    """Idempotent: the same reviewed file always yields the same merged output."""
    if Path(reviewed_csv).exists():
        reviewed = pd.read_csv(reviewed_csv, dtype=str, keep_default_na=False)
    else:
        logger.warning(
            f"No reviewed overrides at {reviewed_csv}; writing an empty merge."
        )
        reviewed = pd.DataFrame(columns=list(REVIEWED_COLUMNS))

    existing = read_merged(out_csv)
    # Drop rows already present, so re-running after a crash never double-counts
    # a correction that landed before the failure.
    if existing is not None and not existing.empty:
        seen = {_text_key_only(t) for t in existing["text"]}
        reviewed = reviewed[~reviewed["text"].map(_text_key_only).isin(seen)]

    fresh, summary = build_merged(reviewed, config)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)

    # Append to what is already merged, never replace it. The training sidecar is
    # cumulative on purpose: `train.py` concatenates this whole file onto train.csv,
    # so overwriting it with only the newest batch would silently *drop* every
    # previously-validated hard negative. Rows are ordered by a content hash so the
    # accumulated file stays byte-stable across re-runs.
    combined = fresh
    if existing is not None and not existing.empty:
        combined = pd.concat([existing, fresh], ignore_index=True)
    if not combined.empty:
        combined = combined.sort_values("override_id", kind="stable").reset_index(
            drop=True
        )
    combined = combined.reindex(columns=list(MERGED_COLUMNS))
    combined.to_csv(out_csv, index=False)

    summary["cumulative_rows"] = len(combined)
    summary["merged_path"] = out_csv
    return summary


def read_merged(path: str) -> pd.DataFrame | None:
    """The current merged file, or ``None`` when it does not exist yet."""
    if not Path(path).exists():
        return None
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    return frame if "text" in frame.columns else None


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

_ERROR_RATE_NOTE = (
    "Measured on captured production traffic, not on a draw from "
    "data/processed/test.csv. Only interpretable if the CS tool posts "
    "confirmations as well as disputes: a confirmation (model_label == "
    "corrected_label) is the denominator, and a tool that posts only disputes "
    "makes this number uninterpretable."
)


def production_error_rate(raw: pd.DataFrame) -> dict[str, Any]:
    """Per-class share of captured reviews the model got wrong.

    ``corrections_where_model_said_c / total_where_model_said_c``.

    Returns ``None`` for a class never observed rather than 0.0, because "the
    model was never wrong about a positive" and "we never saw a positive" are
    different claims and a report that conflates them is worse than one with a
    hole in it.
    """
    empty: dict[str, Any] = {
        "note": _ERROR_RATE_NOTE,
        "by_class": {},
        "overall": None,
        "n_observed": 0,
    }
    if raw.empty or "model_label" not in raw.columns:
        return empty

    predicted = raw["model_label"].map(coerce_label)
    corrected = raw["corrected_label"].map(coerce_label)
    by_class: dict[str, Any] = {}
    total_seen = 0
    total_wrong = 0
    for label in LABELS:
        seen = predicted == label
        n_seen = int(seen.sum())
        n_wrong = int((corrected[seen] != label).sum()) if n_seen else 0
        total_seen += n_seen
        total_wrong += n_wrong
        by_class[label] = {
            "n_observed": n_seen,
            "n_wrong": n_wrong,
            "rate": round(n_wrong / n_seen, _PROPORTION_DP) if n_seen else None,
        }
    return {
        "note": _ERROR_RATE_NOTE,
        "by_class": by_class,
        "overall": round(total_wrong / total_seen, _PROPORTION_DP)
        if total_seen
        else None,
        "n_observed": total_seen,
    }


def build_report(
    raw: pd.DataFrame,
    review: ReviewResult,
    merge_summary: dict[str, Any],
    config: FeedbackConfig,
) -> dict[str, Any]:
    """Assemble ``reports/feedback_metrics.json``."""
    counts = {status: 0 for status in ALL_STATUSES}
    counts.update(review.counts)
    error_rate = production_error_rate(raw)
    n_corrections = sum(entry["n_wrong"] for entry in error_rate["by_class"].values())
    n_raw = len(raw)
    return {
        "config": {
            "min_corroborating_agents": config.min_corroborating_agents,
            "suspicious_model_score": config.suspicious_model_score,
            "max_neutral_per_batch": config.max_neutral_per_batch,
            "min_char_length": config.min_char_length,
        },
        "captured": {
            "n_raw": n_raw,
            "n_corrections": n_corrections,
            "n_confirmations": n_raw - n_corrections,
            "model_label_mix": (
                label_proportions(raw["model_label"].map(coerce_label)) if n_raw else {}
            ),
        },
        "qa": {
            "counts": counts,
            "n_trainable": int(sum(counts[status] for status in TRAINABLE_STATUSES)),
            "n_route_adjudicate_first": (
                int((review.frame["route"] == ROUTE_ADJUDICATE_FIRST).sum())
                if not review.frame.empty and "route" in review.frame.columns
                else 0
            ),
        },
        "production_error_rate": error_rate,
        "merge": merge_summary,
        "caveat": (
            "There is no deployed service or CS tool yet, so every value here is a "
            "rehearsal until real traffic arrives. Two-agent corroboration also "
            "needs a roster with at least two real agent ids: until one exists "
            "every row is `single_agent` and nothing is trainable."
        ),
    }


def write_report(report: dict[str, Any], out_json: str) -> None:
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)


def log_run(mode: str, report: dict[str, Any], out_json: str) -> None:
    """Log to the ``raay_batch`` experiment, matching ``batch_score._log_run``."""
    import mlflow

    mlflow.set_experiment(Experiments.BATCH.value)
    with mlflow.start_run(run_name=f"feedback-{mode}"):
        mlflow.set_tag("run_type", f"feedback_{mode}")
        error_rate = report["production_error_rate"]
        if error_rate["overall"] is not None:
            mlflow.log_metric("feedback_production_error_rate", error_rate["overall"])
        for label, entry in error_rate["by_class"].items():
            if entry["rate"] is not None:
                mlflow.log_metric(f"feedback_error_rate_{label}", entry["rate"])
        mlflow.log_metric("feedback_n_raw", float(report["captured"]["n_raw"]))
        mlflow.log_metric(
            "feedback_n_corrections", float(report["captured"]["n_corrections"])
        )
        mlflow.log_metric(
            "feedback_n_confirmations", float(report["captured"]["n_confirmations"])
        )
        mlflow.log_metric("feedback_n_trainable", float(report["qa"]["n_trainable"]))
        mlflow.log_artifact(out_json, artifact_path="feedback")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def load_params(path: str) -> dict[str, Any]:
    import yaml

    if not Path(path).exists():
        logger.warning(f"No {path}; using FeedbackConfig defaults.")
        return {}
    with open(path) as handle:
        return yaml.safe_load(handle) or {}


def has_new_feedback(reviewed_csv: str, merged_csv: str) -> bool:
    """Any reviewed row that has not been merged yet.

    The Airflow predicate for skipping the nightly merge. Reading the merged file
    rather than trusting a counter is what makes a re-run after a crash correct:
    a row that landed in ``train_feedback.csv`` before the task failed is not
    re-merged, so the retry is a no-op instead of a duplicate.
    """
    if not Path(reviewed_csv).exists():
        return False
    reviewed = pd.read_csv(reviewed_csv, dtype=str, keep_default_na=False)
    if reviewed.empty or "status" not in reviewed.columns:
        return False
    trainable = reviewed[reviewed["status"].isin(TRAINABLE_STATUSES)]
    if trainable.empty:
        return False
    merged = read_merged(merged_csv)
    if merged is None or merged.empty:
        return True
    seen = {_text_key_only(t) for t in merged["text"]}
    return not bool(trainable["text"].map(_text_key_only).isin(seen).all())


def _read_reviewed_for_report(reviewed_csv: str) -> pd.DataFrame:
    """The reviewed file, for the error-rate block in ``--mode merge``."""
    if not Path(reviewed_csv).exists():
        return pd.DataFrame()
    return pd.read_csv(reviewed_csv, dtype=str, keep_default_na=False)


def _review_for_report(raw: pd.DataFrame, config: FeedbackConfig) -> ReviewResult:
    """Re-derive statuses from already-reviewed rows.

    ``--mode merge`` must not re-review (that would re-run the ladder, and the
    reviewed file's statuses are the operator-approved truth). This only rebuilds
    the ``ReviewResult`` envelope so ``build_report`` has status counts to report,
    reading the statuses straight off the rows rather than recomputing them -- a
    recompute here could disagree with the file DVC just hashed.
    """
    if raw.empty or "status" not in raw.columns:
        return ReviewResult(frame=pd.DataFrame(), counts={})
    counts = {status: int((raw["status"] == status).sum()) for status in ALL_STATUSES}
    return ReviewResult(frame=raw, counts=counts)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Customer-service feedback QA and merge"
    )
    parser.add_argument("--mode", choices=["review", "merge"], default="merge")
    parser.add_argument("--raw", default=DefaultPaths.FEEDBACK_RAW.value)
    parser.add_argument("--reviewed", default=DefaultPaths.FEEDBACK_REVIEWED.value)
    parser.add_argument("--merged", default=DefaultPaths.FEEDBACK_MERGED.value)
    parser.add_argument("--test-split", default=DefaultPaths.TEST_SPLIT.value)
    parser.add_argument("--metrics-out", default=DefaultPaths.FEEDBACK_METRICS.value)
    parser.add_argument("--params", default=DefaultPaths.PARAMS.value)
    parser.add_argument("--dedup-threshold", type=float, default=None)
    parser.add_argument("--max-neutral", type=int, default=None)
    parser.add_argument("--no-mlflow", action="store_true")
    return parser.parse_args(argv)


def _run_review(
    args: argparse.Namespace, config: FeedbackConfig, threshold: float, elongation: int
) -> ReviewResult:
    """Raw assertions -> the reviewed file. The operator-facing half.

    Reads the append-only capture directory and rewrites ``reviewed/overrides.csv``
    with a status per row. It also marks rows an earlier run already merged, so a
    re-review cannot resurrect them as fresh candidates.
    """
    raw = read_raw(args.raw)
    existing = read_merged(args.merged)
    review = review_rows(raw, config, args.test_split, threshold, existing, elongation)
    write_reviewed(review, args.reviewed)
    return review


def main(argv: list[str] | None = None) -> int:
    load_environment()
    args = parse_args(argv)
    params = load_params(args.params)
    config = FeedbackConfig.from_params(params)
    if args.max_neutral is not None:
        config.max_neutral_per_batch = args.max_neutral
    preprocessing = params.get("preprocessing", {})
    threshold = (
        args.dedup_threshold
        if args.dedup_threshold is not None
        else float(preprocessing.get("dedup_similarity_threshold", 0.9))
    )
    elongation = int(preprocessing.get("elongation_max_repeat", 2))

    # The two modes read *different* things and must not share a path.
    #
    # `--mode merge` is a DVC stage whose only declared dependency is
    # `reviewed/overrides.csv`. If it also read `data/feedback/raw/` it would be
    # unreproducible from its own deps (a runner has no raw files, since they are
    # git-ignored), and worse, it would *rewrite* the reviewed file -- destroying
    # every `adjudicator_id` / `adjudicated_label` the operator typed in by hand
    # after `--mode review`, silently demoting every adjudication back to
    # `single_agent`. So merge consumes the reviewed artifact as-is.
    #
    # The reviewed file is the human-in-the-loop boundary: review writes it, a
    # person edits it, DVC hashes it, merge consumes it.
    if args.mode == "review":
        review = _run_review(args, config, threshold, elongation)
        # Report what *would* be trainable without touching data/processed/. That
        # is the mode an operator runs before `dvc add`.
        _, summary = build_merged(review.frame, config)
        existing = read_merged(args.merged)
        summary["merged_path"] = None
        summary["cumulative_rows"] = 0 if existing is None else len(existing)
        raw = read_raw(args.raw)
        logger.info(
            f"Review only: {review.counts}; {summary['n_accepted']} would be trainable"
        )
    else:
        summary = merge_reviewed(args.reviewed, args.merged, config)
        logger.info(
            f"Merged {summary['n_accepted']} rows into {args.merged} "
            f"(collapsed {summary['n_collapsed_to_one_row_per_review']} "
            f"per-review duplicates, cap suppressed "
            f"{summary['n_suppressed_by_cap']} neutral)"
        )
        # The merge has no business reading raw captures, so the error-rate block
        # is built from the reviewed file -- the same assertions the merge saw.
        raw = _read_reviewed_for_report(args.reviewed)

    report = build_report(
        raw,
        review if args.mode == "review" else _review_for_report(raw, config),
        summary,
        config,
    )
    write_report(report, args.metrics_out)
    if not args.no_mlflow:
        log_run(args.mode, report, args.metrics_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
