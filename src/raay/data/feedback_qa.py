"""The QA ladder: corroboration groups, per-row status, route and leak guard.

An override is a proposal, not ground truth: nothing reaches training on a single
assertion. See ``raay.data.feedback`` for the full argument; the three rules that
are easy to get wrong (and each cost a test) are

1. corroboration counts **distinct agents**, not rows;
2. ``adjudicated_label`` is read per *group*, not per row;
3. ``suspicious_model_score`` **routes** a row, it never rejects one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from rapidfuzz import fuzz, process

from raay.data.feedback_schema import (
    ROUTE_ADJUDICATE_FIRST,
    ROUTE_METRICS_ONLY,
    ROUTE_TRAIN,
    ROUTES_NEEDS_ADJUDICATOR,
    ROUTES_NEEDS_SECOND_AGENT,
    STATUS_ADJUDICATED,
    STATUS_CONFIRMATION,
    STATUS_CORROBORATED,
    STATUS_DISPUTED,
    STATUS_SINGLE_AGENT,
    TRAINABLE_STATUSES,
    FeedbackConfig,
)
from raay.data.feedback_text import coerce_label, numeric_or_none


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
