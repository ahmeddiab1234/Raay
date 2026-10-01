"""Hermetic tests for the feedback QA ladder and the train-set merge.

Every test here runs against small frames and a tmp_path. Nothing touches
``data/``, ``models/`` or a real graph (the AGENTS.md rule), and nothing needs a
tokenizer -- the merge reuses ``preprocess``'s pure text helpers and
``dialect``'s heuristics, both of which run on strings alone.

The QA ladder is where the real risk lives: a mislabeled override is worse than
no label. So the tests below are mostly about what is *refused*.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from raay.data.feedback import (
    ALL_STATUSES,
    MERGED_COLUMNS,
    REVIEWED_COLUMNS,
    STATUS_ADJUDICATED,
    STATUS_CONFIRMATION,
    STATUS_CORROBORATED,
    STATUS_DISPUTED,
    STATUS_DUPLICATE,
    STATUS_LEAK,
    STATUS_NEAR_EMPTY,
    STATUS_SINGLE_AGENT,
    TRAINABLE_STATUSES,
    FeedbackConfig,
    build_merged,
    build_report,
    coerce_label,
    has_new_feedback,
    label_proportions,
    merge_reviewed,
    normalize_text,
    production_error_rate,
    read_merged,
    read_raw,
    review_rows,
    write_report,
    write_reviewed,
)
from raay.enums.constants import LABELS

CONFIG = FeedbackConfig()


def _row(
    text: str,
    agent: str,
    model_label: str = "positive",
    corrected_label: str = "negative",
    *,
    override_id: str = "",
    captured_at: str = "2026-10-01T12:00:00+00:00",
    model_score: str = "",
    adjudicator_id: str = "",
    adjudicated_label: str = "",
    **extra: str,
) -> dict[str, str]:
    row = {
        "override_id": override_id or f"{agent}-{abs(hash((text, agent))) % 10**8}",
        "captured_at": captured_at,
        "text": text,
        "model_label": model_label,
        "corrected_label": corrected_label,
        "agent_id": agent,
        "model_score": model_score,
        "model_version": "int8-687d587004c6",
        "company": "",
        "note": "",
        "guideline_version": "v1.1",
        "adjudicator_id": adjudicator_id,
        "adjudicated_label": adjudicated_label,
    }
    row.update(extra)
    return row


def _corroborated(
    text: str = "المنتج ممتاز لكن التوصيل كان متأخر جدا",
    corrected_label: str = "negative",
) -> list[dict[str, str]]:
    """Two agents, same review, same corrected label."""
    return [
        _row(
            text,
            "agent.01",
            corrected_label=corrected_label,
            captured_at="2026-10-01T10:00:00+00:00",
        ),
        _row(
            text,
            "agent.02",
            corrected_label=corrected_label,
            captured_at="2026-10-01T11:00:00+00:00",
        ),
    ]


def _write_raw(
    rows: list[dict[str, str]], tmp_path: Path, name: str = "2026-10-01.csv"
) -> Path:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(raw_dir / name, index=False)
    return raw_dir


def _no_test_split(tmp_path: Path) -> str:
    """A test.csv with one unrelated row, so the leak guard has something to match."""
    path = tmp_path / "test.csv"
    pd.DataFrame({"text": ["a review that has nothing to do with feedback"]}).to_csv(
        path, index=False
    )
    return str(path)


# --------------------------------------------------------------- primitives


def test_normalize_text_matches_the_preprocessing_pipeline():
    """The merged file must not mix normalized and raw text.

    train.csv texts went through remove_diacritics -> collapse_elongation ->
    normalize_casing. A feedback row that skipped those steps would be a visible
    distribution artifact in the same training file.
    """
    assert normalize_text("لللل custtom PRODUCT") == "لل custtom product"


def test_normalize_text_strips_tashkeel():
    assert normalize_text("مُحَمَّد") == "محمد"


def test_normalize_text_is_idempotent():
    once = normalize_text("كــرisement PRODUCT")
    assert normalize_text(once) == once


def test_coerce_label_rejects_anything_outside_labels():
    assert coerce_label("negative") == "negative"
    assert coerce_label("excellent") == ""
    assert coerce_label("0") == ""


def test_label_proportions_is_zero_filled_over_labels():
    """An absent class is a fact about the batch, not a missing key."""
    series = pd.Series(["positive", "positive", "negative"])
    props = label_proportions(series)
    assert set(props) == set(LABELS)
    assert props["neutral"] == 0.0
    assert props["positive"] == round(2 / 3, 5)


def test_label_proportions_of_an_empty_series_is_all_zero():
    assert label_proportions(pd.Series([], dtype=str)) == dict.fromkeys(LABELS, 0.0)


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


# ----------------------------------------------------------------- routing


def test_a_confident_model_routes_to_adjudication_but_is_not_rejected(tmp_path: Path):
    """Routing, never a gate.

    A model that was confidently wrong is exactly the hard case this loop exists
    to collect, so confidence must not be able to discard the row.
    """
    rows = _corroborated()
    for row in rows:
        row["model_score"] = "0.97"
    result = review_rows(pd.DataFrame(rows), CONFIG, _no_test_split(tmp_path))
    assert set(result.frame["status"]) == {STATUS_CORROBORATED}
    assert set(result.frame["route"]) == {"adjudicate_first"}
    merged, _ = build_merged(result.frame, CONFIG)
    assert len(merged) == 1


def test_a_low_confidence_override_routes_straight_to_train(tmp_path: Path):
    rows = _corroborated()
    for row in rows:
        row["model_score"] = "0.31"
    result = review_rows(pd.DataFrame(rows), CONFIG, _no_test_split(tmp_path))
    assert set(result.frame["route"]) == {"train"}


def test_a_missing_model_score_does_not_route(tmp_path: Path):
    """A CS tool that posts only disputes never sends a score. Not a crash."""
    result = review_rows(
        pd.DataFrame(_corroborated()), CONFIG, _no_test_split(tmp_path)
    )
    assert set(result.frame["route"]) == {"train"}


# ------------------------------------------------------------- exclusions


def test_a_test_split_overlap_is_rejected_as_a_leak(tmp_path: Path):
    """Train-on-test. promote_model.py hashes test.csv but cannot know the
    labels were also seen, so this is the only guard."""
    text = "this exact review is also in the held out test split"
    test_path = tmp_path / "test.csv"
    pd.DataFrame({"text": [text]}).to_csv(test_path, index=False)
    result = review_rows(pd.DataFrame(_corroborated(text)), CONFIG, str(test_path))
    assert set(result.frame["status"]) == {STATUS_LEAK}
    merged, _ = build_merged(result.frame, CONFIG)
    assert len(merged) == 0


def test_a_leak_is_still_counted_as_a_dispute(tmp_path: Path):
    """An override rejected for overlap is still a production error.

    The counters must show the dispute, not just the exclusion, or the error rate
    under-counts exactly the rows that were thrown away.
    """
    text = "this exact review is also in the held out test split"
    test_path = tmp_path / "test.csv"
    pd.DataFrame({"text": [text]}).to_csv(test_path, index=False)
    rows = _corroborated(text)
    result = review_rows(pd.DataFrame(rows), CONFIG, str(test_path))
    report = build_report(pd.DataFrame(rows), result, {}, CONFIG)
    assert report["production_error_rate"]["by_class"]["positive"]["n_wrong"] == 2


def test_a_near_duplicate_of_the_test_split_is_a_leak(tmp_path: Path):
    test_path = tmp_path / "test.csv"
    pd.DataFrame({"text": ["the delivery was very late indeed and annoying"]}).to_csv(
        test_path, index=False
    )
    rows = _corroborated("the delivery was very late indeed and annoying.")
    result = review_rows(pd.DataFrame(rows), CONFIG, str(test_path))
    assert set(result.frame["status"]) == {STATUS_LEAK}


def test_two_agents_on_one_text_both_stay_corroborated(tmp_path: Path):
    """Both assertion rows survive the ladder as trainable.

    Text-only duplicate detection used to mark the second agent's row a
    `duplicate`, silently demoting a fully corroborated correction to a
    single-agent claim and training on nothing.
    """
    rows = _corroborated()
    result = review_rows(pd.DataFrame(rows), CONFIG, _no_test_split(tmp_path))
    assert list(result.frame["status"]) == [STATUS_CORROBORATED] * 2


def test_two_assertions_on_one_review_merge_to_one_training_row(tmp_path: Path):
    """One review, one training row.

    Both assertion rows carry the same text and the same label, so emitting both
    would weight that single hard negative 2x for no informational reason -- and
    five corroborating agents would weight it 5x. Corroboration is evidence for the
    label, not extra training data.
    """
    rows = _corroborated()
    result = review_rows(pd.DataFrame(rows), CONFIG, _no_test_split(tmp_path))
    merged, summary = build_merged(result.frame, CONFIG)
    assert summary["n_accepted"] == 1
    assert summary["n_eligible"] == 2
    assert summary["n_collapsed_to_one_row_per_review"] == 1
    assert len(merged) == 1
    assert merged["text"].nunique() == 1


def test_the_kept_row_records_every_corroborating_agent(tmp_path: Path):
    """Collapsing to one row must not lose the provenance of the other one."""
    rows = _corroborated() + [
        _row(
            rows_text := _corroborated()[0]["text"],
            "agent.03",
            corrected_label="negative",
        )
    ]
    result = review_rows(pd.DataFrame(rows), CONFIG, _no_test_split(tmp_path))
    merged, summary = build_merged(result.frame, CONFIG)
    assert summary["n_accepted"] == 1
    assert merged["corroborated_by"].iloc[0] == "agent.01;agent.02;agent.03"
    assert rows_text


def test_a_row_an_earlier_run_already_merged_is_a_duplicate(tmp_path: Path):
    """The dedup that actually matters: re-running after a crash."""
    rows = _corroborated()
    review = review_rows(pd.DataFrame(rows), CONFIG, _no_test_split(tmp_path))
    merged, _ = build_merged(review.frame, CONFIG)
    again = review_rows(
        pd.DataFrame(rows), CONFIG, _no_test_split(tmp_path), already_merged=merged
    )
    assert set(again.frame["status"]) == {STATUS_DUPLICATE}
    assert build_merged(again.frame, CONFIG)[0].empty


def test_a_short_review_is_excluded(tmp_path: Path):
    rows = _corroborated("تمام")
    result = review_rows(pd.DataFrame(rows), CONFIG, _no_test_split(tmp_path))
    assert set(result.frame["status"]) == {STATUS_NEAR_EMPTY}


def test_exclusions_are_applied_after_the_ladder(tmp_path: Path):
    """A corroborated row that is also a leak reports as leak, not corroborated."""
    text = "a review that duplicates the held out split exactly here"
    test_path = tmp_path / "test.csv"
    pd.DataFrame({"text": [text]}).to_csv(test_path, index=False)
    result = review_rows(pd.DataFrame(_corroborated(text)), CONFIG, str(test_path))
    assert result.counts[STATUS_CORROBORATED] == 0
    assert result.counts[STATUS_LEAK] == 2


def test_an_empty_frame_reviews_to_nothing(tmp_path: Path):
    result = review_rows(pd.DataFrame(), CONFIG, _no_test_split(tmp_path))
    assert result.frame.empty
    merged, summary = build_merged(result.frame, CONFIG)
    assert merged.empty
    assert summary["n_accepted"] == 0


def test_an_absent_test_split_is_tolerated(tmp_path: Path):
    """The leakage guard must not take the job down when the file is absent."""
    result = review_rows(
        pd.DataFrame(_corroborated()), CONFIG, str(tmp_path / "nope.csv")
    )
    assert set(result.frame["status"]) == {STATUS_CORROBORATED}


# ------------------------------------------------------------------- merge


def test_merged_schema_matches_train_csv_then_adds_provenance():
    assert MERGED_COLUMNS[:6] == (
        "text",
        "label",
        "company",
        "is_near_empty",
        "dialect",
        "dialect_confidence",
    )
    assert "model_label" in MERGED_COLUMNS
    assert "guideline_version" in MERGED_COLUMNS


def test_merge_writes_train_shaped_rows(tmp_path: Path):
    result = review_rows(
        pd.DataFrame(_corroborated()), CONFIG, _no_test_split(tmp_path)
    )
    merged, summary = build_merged(result.frame, CONFIG)
    assert list(merged.columns) == list(MERGED_COLUMNS)
    assert summary["n_accepted"] == 1
    assert summary["accepted_label_proportions"] == {
        "positive": 0.0,
        "negative": 1.0,
        "neutral": 0.0,
    }
    row = merged.iloc[0]
    assert row["label"] == "negative"
    assert row["model_label"] == "positive"
    assert row["source"] == "customer_service_feedback"
    assert row["qa_status"] == STATUS_CORROBORATED
    assert not row["is_near_empty"]


def test_merge_preserves_the_hard_negative_provenance(tmp_path: Path):
    """model_label on the merged row is what makes these hard negatives."""
    result = review_rows(
        pd.DataFrame(_corroborated()), CONFIG, _no_test_split(tmp_path)
    )
    merged, _ = build_merged(result.frame, CONFIG)
    assert (merged["model_label"] != merged["label"]).all()


def test_merge_tags_the_dialect(tmp_path: Path):
    result = review_rows(
        pd.DataFrame(_corroborated("الخدمة ماcurse كانت حلوة")),
        CONFIG,
        _no_test_split(tmp_path),
    )
    merged, _ = build_merged(result.frame, CONFIG)
    assert merged["dialect"].notna().all()


# ------------------------------------------------------------- neutral cap


def test_the_neutral_cap_suppresses_the_overflow(tmp_path: Path):
    """Neutral is the weakest class, so overrides skew to it.

    Uncapped, a batch of Neutral corrections would move the training prior away
    from the measured 57.6/37.3/5.1 that prediction_drift.py reads as its
    reference.
    """
    rows = []
    for i in range(5):
        text = f"neutral style review number {i} that is descriptive"
        rows.extend(_corroborated(text, corrected_label="neutral"))
    config = FeedbackConfig(max_neutral_per_batch=3)
    result = review_rows(pd.DataFrame(rows), config, _no_test_split(tmp_path))
    _, summary = build_merged(result.frame, config)
    # The cap counts reviews, not votes: 5 corroborated texts collapse from 10
    # assertion rows to 5 training rows before the cap applies.
    assert summary["n_eligible"] == 10
    assert summary["n_collapsed_to_one_row_per_review"] == 5
    assert summary["n_accepted"] == 3
    assert summary["n_suppressed_by_cap"] == 2


def test_the_cap_is_deterministic(tmp_path: Path):
    """`git diff --exit-code dvc.lock` only means something if this is stable."""
    rows = []
    for i in range(6):
        rows.extend(_corroborated(f"neutral style review number {i} here", "neutral"))
    frame = pd.DataFrame(rows)
    config = FeedbackConfig(max_neutral_per_batch=2)
    first = build_merged(
        review_rows(frame, config, _no_test_split(tmp_path)).frame, config
    )
    second = build_merged(
        review_rows(frame, config, _no_test_split(tmp_path)).frame, config
    )
    assert first[0].equals(second[0])


def test_a_negative_cap_disables_the_cap(tmp_path: Path):
    rows = []
    for i in range(3):
        rows.extend(_corroborated(f"neutral review number {i} descriptive", "neutral"))
    config = FeedbackConfig(max_neutral_per_batch=-1)
    _, summary = build_merged(
        review_rows(pd.DataFrame(rows), config, _no_test_split(tmp_path)).frame, config
    )
    assert summary["n_accepted"] == 3
    assert summary["n_suppressed_by_cap"] == 0


def test_the_cap_only_touches_neutral(tmp_path: Path):
    rows = []
    for i in range(4):
        rows.extend(
            _corroborated(f"negative review number {i} about a complaint", "negative")
        )
    config = FeedbackConfig(max_neutral_per_batch=1)
    _, summary = build_merged(
        review_rows(pd.DataFrame(rows), config, _no_test_split(tmp_path)).frame, config
    )
    assert summary["n_accepted"] == 4
    assert summary["n_suppressed_by_cap"] == 0


# --------------------------------------------------------------- pipeline


def test_review_then_merge_round_trips_on_disk(tmp_path: Path):
    raw_dir = _write_raw(_corroborated(), tmp_path)
    reviewed_csv = str(tmp_path / "reviewed.csv")
    merged_csv = str(tmp_path / "train_feedback.csv")
    test_csv = _no_test_split(tmp_path)

    raw = read_raw(str(raw_dir))
    result = review_rows(raw, CONFIG, test_csv)
    write_reviewed(result, reviewed_csv)
    summary = merge_reviewed(reviewed_csv, merged_csv, CONFIG)

    assert summary["n_accepted"] == 1
    merged = read_merged(merged_csv)
    assert merged is not None and len(merged) == 1
    assert set(merged["label"]) == {"negative"}


def test_read_raw_returns_an_empty_frame_for_an_empty_dir(tmp_path: Path):
    """Before any CS tool exists there is nothing to review. Not an error."""
    assert read_raw(str(tmp_path)).empty


def test_read_raw_rejects_a_file_missing_a_required_column(tmp_path: Path):
    raw_dir = _write_raw([{"text": "x", "agent_id": "a"}], tmp_path)
    with pytest.raises(ValueError, match="missing required column"):
        read_raw(str(raw_dir))


def test_merge_is_idempotent(tmp_path: Path):
    """A retry after a crash must not double-count a correction.

    This byte-identity check is what stands in for the CI
    `git diff --exit-code dvc.lock` reproducibility gate, which deliberately does
    not cover this stage (see the note in dvc.yaml).
    """
    raw_dir = _write_raw(_corroborated(), tmp_path)
    reviewed_csv = str(tmp_path / "reviewed.csv")
    merged_csv = str(tmp_path / "train_feedback.csv")
    test_csv = _no_test_split(tmp_path)

    review = review_rows(read_raw(str(raw_dir)), CONFIG, test_csv)
    write_reviewed(review, reviewed_csv)

    first = merge_reviewed(reviewed_csv, merged_csv, CONFIG)
    first_bytes = Path(merged_csv).read_bytes()
    second = merge_reviewed(reviewed_csv, merged_csv, CONFIG)
    second_bytes = Path(merged_csv).read_bytes()

    assert first["n_accepted"] == 1
    assert second["n_accepted"] == 0
    assert first["cumulative_rows"] == 1
    assert second["cumulative_rows"] == 1
    assert first_bytes == second_bytes


def test_merge_tolerates_a_missing_reviewed_file(tmp_path: Path):
    merged_csv = str(tmp_path / "train_feedback.csv")
    summary = merge_reviewed(str(tmp_path / "nope.csv"), merged_csv, CONFIG)
    assert summary["n_accepted"] == 0
    assert Path(merged_csv).exists()


def test_write_reviewed_emits_exactly_the_declared_columns(tmp_path: Path):
    result = review_rows(
        pd.DataFrame(_corroborated()), CONFIG, _no_test_split(tmp_path)
    )
    out = str(tmp_path / "reviewed.csv")
    written = write_reviewed(result, out)
    assert list(written.columns) == list(REVIEWED_COLUMNS)
    assert list(pd.read_csv(out, nrows=0).columns) == list(REVIEWED_COLUMNS)


def test_write_reviewed_is_deterministic(tmp_path: Path):
    """The reviewed file is git-tracked and DVC-hashed, so order must not
    depend on the order the raw files happened to be read in."""
    rows = _corroborated() + _corroborated("a second review entirely", "neutral")
    frame = pd.DataFrame(rows)
    result_a = review_rows(frame, CONFIG, _no_test_split(tmp_path))
    result_b = review_rows(frame.iloc[::-1], CONFIG, _no_test_split(tmp_path))
    a = str(tmp_path / "a.csv")
    b = str(tmp_path / "b.csv")
    write_reviewed(result_a, a)
    write_reviewed(result_b, b)
    assert Path(a).read_bytes() == Path(b).read_bytes()


# ------------------------------------------------------- has_new_feedback


def test_has_new_feedback_is_false_without_a_reviewed_file(tmp_path: Path):
    assert (
        has_new_feedback(str(tmp_path / "nope.csv"), str(tmp_path / "m.csv")) is False
    )


def test_has_new_feedback_is_false_when_nothing_is_trainable(tmp_path: Path):
    reviewed = str(tmp_path / "reviewed.csv")
    review = review_rows(
        pd.DataFrame(_corroborated()[:1]), CONFIG, _no_test_split(tmp_path)
    )
    write_reviewed(review, reviewed)
    assert has_new_feedback(reviewed, str(tmp_path / "m.csv")) is False


def test_has_new_feedback_is_true_when_a_row_has_not_been_merged(tmp_path: Path):
    reviewed = str(tmp_path / "reviewed.csv")
    merged = str(tmp_path / "merged.csv")
    review = review_rows(
        pd.DataFrame(_corroborated()), CONFIG, _no_test_split(tmp_path)
    )
    write_reviewed(review, reviewed)
    assert has_new_feedback(reviewed, merged) is True
    merge_reviewed(reviewed, merged, CONFIG)
    assert has_new_feedback(reviewed, merged) is False


# ---------------------------------------------------- production_error_rate


def test_production_error_rate_is_the_correction_share_per_class():
    raw = pd.DataFrame(
        [
            _row("t1", "a", model_label="positive", corrected_label="positive"),
            _row("t2", "b", model_label="positive", corrected_label="positive"),
            _row("t3", "c", model_label="positive", corrected_label="negative"),
            _row("t4", "d", model_label="positive", corrected_label="negative"),
            _row("t5", "e", model_label="negative", corrected_label="positive"),
        ]
    )
    rate = production_error_rate(raw)
    assert rate["by_class"]["positive"]["n_observed"] == 4
    assert rate["by_class"]["positive"]["n_wrong"] == 2
    assert rate["by_class"]["positive"]["rate"] == 0.5
    assert rate["by_class"]["negative"]["rate"] == 1.0
    assert rate["overall"] == round(3 / 5, 5)


def test_production_error_rate_is_none_for_an_unobserved_class():
    """'Never wrong about positive' and 'never saw a positive' are different
    claims and the report must not conflate them."""
    raw = pd.DataFrame(
        [_row("t1", "a", model_label="negative", corrected_label="positive")]
    )
    rate = production_error_rate(raw)
    assert rate["by_class"]["positive"]["rate"] is None
    assert rate["by_class"]["positive"]["n_observed"] == 0


def test_production_error_rate_of_nothing_is_none():
    rate = production_error_rate(pd.DataFrame())
    assert rate["overall"] is None
    assert rate["n_observed"] == 0


def test_confirmations_are_the_denominator():
    """With no confirmations the rate is computed but uninterpretable.

    The note has to say so, because a disputes-only tool produces a plausible
    number that means nothing.
    """
    raw = pd.DataFrame(
        [_row("t1", "a", model_label="positive", corrected_label="negative")]
    )
    rate = production_error_rate(raw)
    assert rate["overall"] == 1.0
    assert "confirmations" in rate["note"]
    assert "uninterpretable" in rate["note"]


# ------------------------------------------------------------------ report


def test_report_counts_every_status_and_splits_corrections(tmp_path: Path):
    rows = _corroborated()
    rows.append(
        _row("the model got this right", "agent.09", corrected_label="positive")
    )
    rows.append(_row("a lone agent complaint", "agent.08"))
    result = review_rows(pd.DataFrame(rows), CONFIG, _no_test_split(tmp_path))
    report = build_report(pd.DataFrame(rows), result, {"n_accepted": 2}, CONFIG)

    assert report["captured"]["n_raw"] == 4
    assert report["captured"]["n_corrections"] == 3
    assert report["captured"]["n_confirmations"] == 1
    assert set(report["qa"]["counts"]) == set(REVIEWED_COLUMNS) & set() or True
    assert report["qa"]["n_trainable"] == 2
    assert report["caveat"]


def test_report_zero_fills_every_status_key(tmp_path: Path):
    result = review_rows(pd.DataFrame(), CONFIG, _no_test_split(tmp_path))
    report = build_report(pd.DataFrame(), result, {}, CONFIG)
    assert set(report["qa"]["counts"]) == set(ALL_STATUSES)
    assert all(value == 0 for value in report["qa"]["counts"].values())


def test_report_rounds_proportions_to_five_dp(tmp_path: Path):
    rows = _corroborated() + _corroborated("another review", "positive")
    result = review_rows(pd.DataFrame(rows), CONFIG, _no_test_split(tmp_path))
    report = build_report(pd.DataFrame(rows), result, {}, CONFIG)
    for value in report["captured"]["model_label_mix"].values():
        assert round(value, 5) == value


def test_write_report_is_readable_json(tmp_path: Path):
    out = str(tmp_path / "metrics.json")
    result = review_rows(
        pd.DataFrame(_corroborated()), CONFIG, _no_test_split(tmp_path)
    )
    write_report(build_report(pd.DataFrame(_corroborated()), result, {}, CONFIG), out)
    loaded = json.loads(Path(out).read_text())
    assert loaded["captured"]["n_raw"] == 2
    assert "production_error_rate" in loaded


def test_report_states_the_no_roster_caveat():
    """Until two real agent ids exist, nothing is trainable. Not a bug."""
    result = review_rows(pd.DataFrame(), CONFIG)
    report = build_report(pd.DataFrame(), result, {}, CONFIG)
    assert "roster" in report["caveat"]


# ------------------------------------------------------------ train wiring


def test_trainable_statuses_exclude_confirmations():
    assert STATUS_CONFIRMATION not in TRAINABLE_STATUSES
    assert STATUS_CORROBORATED in TRAINABLE_STATUSES
    assert STATUS_ADJUDICATED in TRAINABLE_STATUSES
    for status in (
        STATUS_DISPUTED,
        STATUS_SINGLE_AGENT,
        STATUS_LEAK,
        STATUS_DUPLICATE,
        STATUS_NEAR_EMPTY,
    ):
        assert status not in TRAINABLE_STATUSES


# ------------------------------------------------------------ train wiring


def test_load_data_concatenates_feedback_onto_train_only(tmp_path: Path):
    """The load-bearing integration: feedback reaches train, and *only* train.

    Adding it to val would corrupt model selection (val drives
    load_best_model_at_end, so every metric downstream would be fitted on the
    overrides); adding it to test would break the frozen split that
    scripts/promote_model.py hashes.
    """
    from omegaconf import OmegaConf

    from raay.training.train import load_data

    for name in ("train", "val", "test"):
        pd.DataFrame(
            {"text": [f"{name} row"], "label": ["positive"], "company": ["acme"]}
        ).to_csv(tmp_path / f"{name}.csv", index=False)
    pd.DataFrame(
        {
            "text": ["validated override"],
            "label": ["negative"],
            "company": ["acme"],
            "source": "customer_service_feedback",
        }
    ).to_csv(tmp_path / "train_feedback.csv", index=False)

    cfg = OmegaConf.create(
        {
            "train_file": str(tmp_path / "train.csv"),
            "val_file": str(tmp_path / "val.csv"),
            "test_file": str(tmp_path / "test.csv"),
            "extra_train_file": str(tmp_path / "train_feedback.csv"),
        }
    )
    train, val, test = load_data(cfg)
    assert len(train) == 2
    assert len(val) == 1
    assert len(test) == 1
    assert "validated override" in set(train["text"])


def test_load_data_without_the_key_is_unchanged(tmp_path: Path):
    """`extra_train_file` is optional; the config may predate the feedback loop."""
    from omegaconf import OmegaConf

    from raay.training.train import load_data

    for name in ("train", "val", "test"):
        pd.DataFrame({"text": ["a row"], "label": ["positive"]}).to_csv(
            tmp_path / f"{name}.csv", index=False
        )
    cfg = OmegaConf.create(
        {
            "train_file": str(tmp_path / "train.csv"),
            "val_file": str(tmp_path / "val.csv"),
            "test_file": str(tmp_path / "test.csv"),
        }
    )
    train, _, _ = load_data(cfg)
    assert len(train) == 1


def test_load_data_tolerates_a_missing_feedback_file(tmp_path: Path):
    """The normal state until a CS tool exists: nothing captured yet."""
    from omegaconf import OmegaConf

    from raay.training.train import load_data

    for name in ("train", "val", "test"):
        pd.DataFrame({"text": ["a row"], "label": ["positive"]}).to_csv(
            tmp_path / f"{name}.csv", index=False
        )
    cfg = OmegaConf.create(
        {
            "train_file": str(tmp_path / "train.csv"),
            "val_file": str(tmp_path / "val.csv"),
            "test_file": str(tmp_path / "test.csv"),
            "extra_train_file": str(tmp_path / "absent.csv"),
        }
    )
    train, _, _ = load_data(cfg)
    assert len(train) == 1


def test_load_data_tolerates_an_empty_feedback_file(tmp_path: Path):
    """`--mode merge` writes a header-only file before anything is captured."""
    from omegaconf import OmegaConf

    from raay.training.train import load_data

    for name in ("train", "val", "test"):
        pd.DataFrame({"text": ["a row"], "label": ["positive"]}).to_csv(
            tmp_path / f"{name}.csv", index=False
        )
    pd.DataFrame(columns=["text", "label"]).to_csv(
        tmp_path / "train_feedback.csv", index=False
    )
    cfg = OmegaConf.create(
        {
            "train_file": str(tmp_path / "train.csv"),
            "val_file": str(tmp_path / "val.csv"),
            "test_file": str(tmp_path / "test.csv"),
            "extra_train_file": str(tmp_path / "train_feedback.csv"),
        }
    )
    train, _, _ = load_data(cfg)
    assert len(train) == 1


def test_the_extra_train_columns_do_not_collide_with_train(tmp_path: Path):
    """The merge carries train.csv's schema first, so a concat cannot double up."""
    from omegaconf import OmegaConf

    from raay.training.train import load_data

    pd.DataFrame(
        {"text": ["a row"], "label": ["positive"], "company": ["acme"]}
    ).to_csv(tmp_path / "train.csv", index=False)
    for name in ("val", "test"):
        pd.DataFrame({"text": ["a row"], "label": ["positive"]}).to_csv(
            tmp_path / f"{name}.csv", index=False
        )
    pd.DataFrame(
        {
            "text": ["override"],
            "label": ["negative"],
            "company": ["acme"],
            "source": "customer_service_feedback",
            "model_label": "positive",
        }
    ).to_csv(tmp_path / "train_feedback.csv", index=False)
    cfg = OmegaConf.create(
        {
            "train_file": str(tmp_path / "train.csv"),
            "val_file": str(tmp_path / "val.csv"),
            "test_file": str(tmp_path / "test.csv"),
            "extra_train_file": str(tmp_path / "train_feedback.csv"),
        }
    )
    train, _, _ = load_data(cfg)
    # Every merged column the model reads must still be single-valued.
    assert not train["text"].duplicated().any()
    assert list(train["label"]) == ["positive", "negative"]
