"""Hermetic tests for feedback merge and persistence."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from feedback_helpers import CONFIG, _corroborated, _no_test_split, _write_raw

from raay.data.feedback import (
    MERGED_COLUMNS,
    REVIEWED_COLUMNS,
    STATUS_CORROBORATED,
    FeedbackConfig,
    build_merged,
    has_new_feedback,
    merge_reviewed,
    read_merged,
    read_raw,
    review_rows,
    write_reviewed,
)


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
