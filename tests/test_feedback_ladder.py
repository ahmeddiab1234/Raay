"""Hermetic tests for the feedback QA ladder and the train-set merge.

Every test here runs against small frames and a tmp_path. Nothing touches
``data/``, ``models/`` or a real graph (the AGENTS.md rule), and nothing needs a
tokenizer -- the merge reuses ``preprocess``'s pure text helpers and
``dialect``'s heuristics, both of which run on strings alone.

The QA ladder is where the real risk lives: a mislabeled override is worse than
no label. So the tests below are mostly about what is *refused*.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from feedback_helpers import CONFIG, _corroborated, _no_test_split, _row

from raay.data.feedback import (
    STATUS_ADJUDICATED,
    STATUS_CONFIRMATION,
    STATUS_CORROBORATED,
    STATUS_DISPUTED,
    STATUS_SINGLE_AGENT,
    FeedbackConfig,
    build_merged,
    review_rows,
)

# --------------------------------------------------------------- primitives


# ------------------------------------------------------------- the ladder


def test_two_agents_agreeing_is_corroborated(tmp_path: Path):
    result = review_rows(
        pd.DataFrame(_corroborated()), CONFIG, _no_test_split(tmp_path)
    )
    assert set(result.frame["status"]) == {STATUS_CORROBORATED}
    assert result.counts[STATUS_CORROBORATED] == 2


def test_one_agent_alone_is_never_trainable(tmp_path: Path):
    """The whole point: a single assertion cannot reach the training set."""
    result = review_rows(
        pd.DataFrame([_corroborated()[0]]), CONFIG, _no_test_split(tmp_path)
    )
    assert result.frame["status"].tolist() == [STATUS_SINGLE_AGENT]
    merged, summary = build_merged(result.frame, CONFIG)
    assert len(merged) == 0
    assert summary["n_eligible"] == 0


def test_two_agents_disagreeing_is_disputed_not_broken_by_a_plurality(tmp_path: Path):
    rows = [
        _row(
            "review about the product and the delivery",
            "agent.01",
            model_label="neutral",
            corrected_label="negative",
        ),
        _row(
            "review about the product and the delivery",
            "agent.02",
            model_label="neutral",
            corrected_label="positive",
        ),
    ]
    result = review_rows(pd.DataFrame(rows), CONFIG, _no_test_split(tmp_path))
    assert set(result.frame["status"]) == {STATUS_DISPUTED}
    assert set(result.frame["route"]) == {"needs_adjudicator"}
    merged, _ = build_merged(result.frame, CONFIG)
    assert len(merged) == 0


def test_a_third_agent_joining_a_two_to_one_split_corroborates(tmp_path: Path):
    """Two agents agreeing is the approved bar, whatever a third says.

    The dissent stays visible in the reviewed file (`dissenting_labels` feeds the
    report), so an adjudicator can still look -- but the two-agent process in
    docs/labeling_guidelines.md section 4 has already happened.
    """
    rows = [
        _row(
            "review about the product and the delivery",
            "agent.01",
            model_label="neutral",
            corrected_label="negative",
        ),
        _row(
            "review about the product and the delivery",
            "agent.02",
            model_label="neutral",
            corrected_label="negative",
        ),
        _row(
            "review about the product and the delivery",
            "agent.03",
            model_label="positive",
            corrected_label="positive",
        ),
    ]
    result = review_rows(pd.DataFrame(rows), CONFIG, _no_test_split(tmp_path))
    # agent.03 agreed with the model, so it is a confirmation and never a voter.
    assert list(result.frame["status"]) == [
        STATUS_CORROBORATED,
        STATUS_CORROBORATED,
        STATUS_CONFIRMATION,
    ]
    merged, _ = build_merged(result.frame, CONFIG)
    assert merged["label"].tolist() == ["negative"]


def test_a_three_way_split_does_not_corroborate(tmp_path: Path):
    """No two agents defend the same label, so there is nothing to train on."""
    rows = [
        _row(
            "review about the product and the delivery",
            "agent.01",
            model_label="neutral",
            corrected_label="negative",
        ),
        _row(
            "review about the product and the delivery",
            "agent.02",
            model_label="neutral",
            corrected_label="positive",
        ),
        _row(
            "review about the product and the delivery",
            "agent.03",
            model_label="neutral",
            corrected_label="neutral",
        ),
    ]
    result = review_rows(pd.DataFrame(rows), CONFIG, _no_test_split(tmp_path))
    assert list(result.frame["status"]) == [STATUS_DISPUTED] * 2 + [STATUS_CONFIRMATION]
    merged, _ = build_merged(result.frame, CONFIG)
    assert len(merged) == 0


def test_one_agent_cannot_corroborate_themselves(tmp_path: Path):
    """Two rows from the same agent are one assertion posted twice."""
    text = "review about the product and the delivery"
    rows = [
        _row(
            text,
            "agent.01",
            model_label="neutral",
            corrected_label="negative",
            override_id="a",
        ),
        _row(
            text,
            "agent.01",
            model_label="neutral",
            corrected_label="negative",
            override_id="b",
        ),
    ]
    result = review_rows(pd.DataFrame(rows), CONFIG, _no_test_split(tmp_path))
    assert set(result.frame["status"]) == {STATUS_SINGLE_AGENT}
    merged, _ = build_merged(result.frame, CONFIG)
    assert len(merged) == 0


def test_a_self_contradicting_agent_does_not_vote(tmp_path: Path):
    """One agent asserting two labels cannot manufacture agreement.

    agent.01 is dropped as self-contradictory, which leaves agent.02 alone -- a
    single voter, so nothing corroborates.
    """
    text = "review about the product and the delivery"
    rows = [
        _row(
            text,
            "agent.01",
            model_label="neutral",
            corrected_label="negative",
            override_id="a",
        ),
        _row(
            text,
            "agent.01",
            model_label="neutral",
            corrected_label="positive",
            override_id="b",
        ),
        _row(
            text,
            "agent.02",
            model_label="neutral",
            corrected_label="negative",
            override_id="c",
        ),
    ]
    result = review_rows(pd.DataFrame(rows), CONFIG, _no_test_split(tmp_path))
    assert set(result.frame["status"]) == {STATUS_SINGLE_AGENT}
    merged, _ = build_merged(result.frame, CONFIG)
    assert len(merged) == 0


def test_an_adjudicated_label_is_final(tmp_path: Path):
    rows = [
        _row(
            "a genuinely disputed review",
            "agent.01",
            adjudicator_id="senior.1",
            adjudicated_label="neutral",
        )
    ]
    result = review_rows(pd.DataFrame(rows), CONFIG, _no_test_split(tmp_path))
    assert result.frame["status"].tolist() == [STATUS_ADJUDICATED]
    merged, _ = build_merged(result.frame, CONFIG)
    assert merged["label"].tolist() == ["neutral"]


def test_adjudication_beats_corroboration(tmp_path: Path):
    """If both are present the adjudicator's ruling wins -- it is the resolution."""
    rows = _corroborated()[:1]
    rows.append(
        _row(
            rows[0]["text"],
            "agent.02",
            corrected_label="negative",
            adjudicator_id="senior.1",
            adjudicated_label="neutral",
        )
    )
    result = review_rows(pd.DataFrame(rows), CONFIG, _no_test_split(tmp_path))
    # The ruling is group-scoped: a human adjudicates a review, not one assertion
    # about it, so it resolves both rows.
    assert set(result.frame["status"]) == {STATUS_ADJUDICATED}
    merged, _ = build_merged(result.frame, CONFIG)
    assert merged["label"].tolist() == ["neutral"]
    assert merged["qa_status"].tolist() == [STATUS_ADJUDICATED]


def test_conflicting_adjudications_are_not_a_ruling(tmp_path: Path):
    """Two adjudicators disagreeing must not pick one -- a human still has to look."""
    text = "a genuinely contested review"
    rows = [
        _row(text, "agent.01", adjudicator_id="s.1", adjudicated_label="negative"),
        _row(text, "agent.02", adjudicator_id="s.2", adjudicated_label="positive"),
    ]
    result = review_rows(pd.DataFrame(rows), CONFIG, _no_test_split(tmp_path))
    assert set(result.frame["status"]) == {STATUS_CORROBORATED}
    assert set(result.frame["adjudicated_label"]) == {""}
    # The agents agreed on `negative`, so the *corroboration* is used and the
    # conflicting rulings are discarded -- not silently resolved in favour of one
    # adjudicator, and not left to train on the adjudication.
    merged, _ = build_merged(result.frame, CONFIG)
    assert merged["label"].tolist() == ["negative"]
    assert merged["qa_status"].tolist() == [STATUS_CORROBORATED]


def test_a_confirmation_is_never_trainable_even_when_corroborated(tmp_path: Path):
    """A confirmation's 'corrected' label is the model's own prediction.

    Two agents agreeing that the model was right is evidence about nothing, and
    treating it as a label would feed the model's own output back into training.
    """
    rows = [
        _row("the model got this one right", "agent.01", corrected_label="positive"),
        _row("the model got this one right", "agent.02", corrected_label="positive"),
    ]
    result = review_rows(pd.DataFrame(rows), CONFIG, _no_test_split(tmp_path))
    assert set(result.frame["status"]) == {STATUS_CONFIRMATION}
    merged, _ = build_merged(result.frame, CONFIG)
    assert len(merged) == 0


def test_a_mixed_group_does_not_inherit_corroboration(tmp_path: Path):
    """One agent confirming and one disputing is not a corroborated correction."""
    rows = [
        _row(
            "a review one agent disputes",
            "agent.01",
            model_label="neutral",
            corrected_label="negative",
        ),
        _row(
            "a review one agent disputes",
            "agent.02",
            model_label="neutral",
            corrected_label="positive",
        ),
    ]
    result = review_rows(pd.DataFrame(rows), CONFIG, _no_test_split(tmp_path))
    assert set(result.frame["status"]) == {STATUS_DISPUTED}


def test_min_corroborating_agents_is_configurable(tmp_path: Path):
    loose = FeedbackConfig(min_corroborating_agents=1)
    result = review_rows(
        pd.DataFrame(_corroborated()[:1]), loose, _no_test_split(tmp_path)
    )
    assert result.frame["status"].tolist() == [STATUS_CORROBORATED]


def test_corroboration_groups_agents_with_and_without_diacritics(tmp_path: Path):
    """Normalization runs first, so the same review groups together."""
    rows = [
        _row("المنتجJourney ممتاز", "agent.01", corrected_label="negative"),
        _row("المنتجJourney  ممتاز", "agent.02", corrected_label="negative"),
    ]
    result = review_rows(pd.DataFrame(rows), CONFIG, _no_test_split(tmp_path))
    assert set(result.frame["status"]) == {STATUS_CORROBORATED}
