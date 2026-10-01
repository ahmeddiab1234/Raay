"""Hermetic tests for feedback review routing and leak checks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from feedback_helpers import CONFIG, _corroborated, _no_test_split, _row

from raay.data.feedback import (
    STATUS_CORROBORATED,
    STATUS_DUPLICATE,
    STATUS_LEAK,
    STATUS_NEAR_EMPTY,
    build_merged,
    build_report,
    review_rows,
)


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
