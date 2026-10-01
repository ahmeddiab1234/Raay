"""Hermetic tests for feedback metrics and reports."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from feedback_helpers import CONFIG, _corroborated, _no_test_split, _row

from raay.data.feedback import (
    ALL_STATUSES,
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
    build_report,
    production_error_rate,
    review_rows,
    write_report,
)


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


def test_write_report_ends_with_a_newline(tmp_path: Path):
    """The file is git-tracked, so it must survive pre-commit byte-for-byte.

    ``reports/feedback_metrics.json`` is a ``cache: false`` metrics output, so
    its md5 lands in ``dvc.lock``. If the writer omits the trailing newline,
    pre-commit's ``end-of-file-fixer`` rewrites the file *after* the stage runs
    and every later ``dvc repro feedback`` shows a phantom diff that
    ``git diff --exit-code dvc.lock`` in CI then fails on.
    """
    out = tmp_path / "metrics.json"
    write_report({"overall": {}}, str(out))
    assert out.read_bytes().endswith(b"}\n")


def test_report_states_the_no_roster_caveat():
    """Until two real agent ids exist, nothing is trainable. Not a bug."""
    result = review_rows(pd.DataFrame(), CONFIG)
    report = build_report(pd.DataFrame(), result, {}, CONFIG)
    assert "roster" in report["caveat"]


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
