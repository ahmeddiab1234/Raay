"""Unit tests for the CI metrics drift gate (``scripts/ci_metrics_gate.py``).

The diff payload shape is fixed by ``dvc/utils/diff.py``: flattened dotted
keys, ``{"old", "new", "diff"}`` entries, and ``None`` on the missing side of
a key that exists in only one revision. These tests pin the gate's verdicts
for each of those cases, plus the failure mode that matters most -- an
unreadable baseline, which DVC's own CLI reports as an empty (passing) diff.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from ci_metrics_gate import (
    FAIL,
    NEW,
    PASS,
    REASON_ADDED,
    REASON_DRIFT,
    REASON_MALFORMED,
    REASON_NEW_FILE,
    REASON_REMOVED,
    REASON_UNREADABLE,
    GateResult,
    classify,
    evaluate,
    format_errors,
    render_markdown,
)

METRICS_FILE = "reports/preprocess_metrics.json"
SPLIT_METRICS_FILE = "reports/split_metrics.json"


def _change(old, new):
    """Mirror ``dvc.utils.diff._diff_vals``: numeric sides get a ``diff`` key."""
    entry = {"old": old, "new": new}
    if isinstance(new, (int, float)) and isinstance(old, (int, float)):
        entry["diff"] = new - old
    return entry


def test_no_changes_passes():
    result = evaluate({})
    assert result.ok
    assert result.rows == []
    assert "No metric changed." in render_markdown(result)


def test_change_within_threshold_passes():
    diff = {METRICS_FILE: {"percentages_pre_norm.latin_chars": _change(4.09, 4.094)}}
    result = evaluate(diff, threshold=0.005)
    assert result.ok
    assert result.rows[0].status == PASS


def test_change_exactly_at_threshold_passes():
    diff = {METRICS_FILE: {"percentages_pre_norm.latin_chars": _change(4.09, 4.095)}}
    result = evaluate(diff, threshold=0.005)
    assert result.ok


def test_change_beyond_threshold_fails():
    diff = {METRICS_FILE: {"percentages_pre_norm.latin_chars": _change(4.09, 4.12)}}
    result = evaluate(diff, threshold=0.005)
    assert not result.ok
    assert result.violations[0].reason == REASON_DRIFT
    assert result.violations[0].change == pytest.approx(0.03)


def test_integer_count_drift_of_one_fails():
    # Counts are integers, so at +/-0.005 "within tolerance" means "identical".
    diff = {METRICS_FILE: {"final_row_count": _change(36045, 36044)}}
    result = evaluate(diff, threshold=0.005)
    assert not result.ok
    assert result.violations[0].change == pytest.approx(-1.0)


def test_negative_drift_beyond_threshold_fails():
    diff = {SPLIT_METRICS_FILE: {"label_proportions.test.positive": _change(0.6, 0.55)}}
    result = evaluate(diff, threshold=0.005)
    assert not result.ok
    assert result.violations[0].change == pytest.approx(-0.05)


def test_new_metric_key_fails_without_diff_entry():
    # dvc emits no "diff" when a key exists on only one side. The file also has
    # a key with a real baseline, so this is an added key, not a new file.
    diff = {
        SPLIT_METRICS_FILE: {
            "dialect_proportions.val.maghrebi": {"old": None, "new": 0.09},
            "test_size": _change(7209, 7209),
        }
    }
    result = evaluate(diff)
    assert not result.ok
    row = result.violations[0]
    assert row.reason == REASON_ADDED
    assert row.change is None


def test_file_with_no_baseline_is_reported_as_new_not_drift():
    # Bootstrap case: the PR introduces reports/split_metrics.json, so every key
    # arrives with old=None and nothing can be compared. That is a missing
    # baseline to record, not drift to block on.
    diff = {
        SPLIT_METRICS_FILE: {
            "test_size": {"old": None, "new": 7209},
            "train_size": {"old": None, "new": 25231},
            "label_proportions.test.positive": {"old": None, "new": 0.57595},
        }
    }
    result = evaluate(diff)
    assert result.ok
    assert result.violations == []
    assert {row.status for row in result.rows} == {NEW}
    assert {row.reason for row in result.rows} == {REASON_NEW_FILE}


def test_fail_on_new_option_blocks_a_missing_baseline():
    diff = {SPLIT_METRICS_FILE: {"test_size": {"old": None, "new": 7209}}}
    result = evaluate(diff, fail_on_new=True)
    assert not result.ok
    assert result.violations[0].status == FAIL
    assert result.violations[0].reason == REASON_NEW_FILE


def test_new_file_and_drift_in_one_diff_reports_both():
    diff = {
        METRICS_FILE: {"final_row_count": _change(36045, 36040)},
        SPLIT_METRICS_FILE: {"test_size": {"old": None, "new": 7209}},
    }
    result = evaluate(diff)
    assert not result.ok
    # The new file is tolerated, the real drift is still reported.
    statuses = {(row.path, row.status) for row in result.rows}
    assert (SPLIT_METRICS_FILE, NEW) in statuses
    assert (METRICS_FILE, FAIL) in statuses


def test_markdown_reports_new_metrics_in_the_verdict():
    diff = {SPLIT_METRICS_FILE: {"test_size": {"old": None, "new": 7209}}}
    markdown = render_markdown(evaluate(diff))
    assert "1 of them new with no baseline yet" in markdown
    assert "**PASSED**" in markdown


def test_removed_metric_key_fails():
    diff = {
        METRICS_FILE: {
            "near_duplicates_removed": {"old": 2062, "new": None},
            "raw_row_count": _change(40046, 40046),
        }
    }
    result = evaluate(diff)
    assert not result.ok
    assert result.violations[0].reason == REASON_REMOVED


def test_non_numeric_metric_change_fails():
    diff = {METRICS_FILE: {"mode": _change("fuzzy", "exact")}}
    result = evaluate(diff)
    assert not result.ok
    assert result.violations[0].reason == REASON_MALFORMED


def test_malformed_entry_fails():
    diff = {METRICS_FILE: {"final_row_count": "not-a-dict"}}
    result = evaluate(diff)
    assert not result.ok
    assert result.violations[0].reason == REASON_MALFORMED


def test_malformed_metrics_file_fails():
    diff = {METRICS_FILE: ["not", "a", "dict"]}
    result = evaluate(diff)
    assert not result.ok
    assert result.violations[0].reason == REASON_MALFORMED


def test_unreadable_baseline_fails_even_with_no_rows():
    # The whole point: dvc's CLI reports this as an empty diff and exits 0.
    result = evaluate({}, errors={"origin/old": {METRICS_FILE: "config file error"}})
    assert not result.ok
    assert result.errors


def test_errors_mention_revision_and_path():
    errors = {"origin/old": {METRICS_FILE: "expected 'url' for dictionary value"}}
    formatted = format_errors(errors)
    assert formatted == [
        f"origin/old: {METRICS_FILE}: expected 'url' for dictionary value"
    ]


def test_exception_errors_are_rendered_by_type():
    errors = {"origin/old": ValueError("boom")}
    assert format_errors(errors) == ["origin/old: ValueError: boom"]


def test_non_dict_errors_are_stringified():
    assert format_errors("plain") == ["plain"]


def test_no_errors_when_mapping_is_empty():
    assert format_errors({}) == []
    assert format_errors(None) == []


def test_rows_are_sorted_by_path_then_metric():
    diff = {
        SPLIT_METRICS_FILE: {"val_size": _change(3670, 3671)},
        METRICS_FILE: {"final_row_count": _change(1, 2)},
    }
    result = evaluate(diff)
    assert [(row.path, row.metric) for row in result.rows] == [
        (METRICS_FILE, "final_row_count"),
        (SPLIT_METRICS_FILE, "val_size"),
    ]


def test_classify_defaults_to_documented_threshold():
    row = classify(METRICS_FILE, "final_row_count", _change(10, 11))
    assert row.status == FAIL
    assert row.reason == REASON_DRIFT


def test_ignore_excludes_matching_keys_even_with_huge_delta():
    diff = {
        SPLIT_METRICS_FILE: {
            "train_size": _change(25231, 80000),
            "test_size": _change(7209, 22000),
            "label_proportions.test.positive": _change(0.57595, 0.57605),
        }
    }
    result = evaluate(diff, threshold=0.005, ignore=".*_size$")
    assert result.ok
    assert result.ignored == 2
    assert [row.metric for row in result.rows] == ["label_proportions.test.positive"]


def test_ignore_matches_against_the_metrics_path_too():
    # `--ignore` must be able to drop an entire metrics file by its path.
    diff = {METRICS_FILE: {"final_row_count": _change(36045, 1)}}
    result = evaluate(diff, ignore=r"preprocess_metrics\.json")
    assert result.ok
    assert result.ignored == 1
    assert result.rows == []


def test_ignore_does_not_mask_unignored_violations():
    diff = {
        METRICS_FILE: {"final_row_count": _change(36045, 36040)},
        SPLIT_METRICS_FILE: {
            "train_size": _change(25231, 90000),
            "label_proportions.test.positive": _change(0.57595, 0.55),
        },
    }
    result = evaluate(diff, threshold=0.005, ignore=".*_size$")
    assert not result.ok
    assert result.ignored == 1
    assert {row.metric for row in result.violations} == {
        "final_row_count",
        "label_proportions.test.positive",
    }


def test_ignore_accepts_a_compiled_pattern():
    import re as _re

    diff = {SPLIT_METRICS_FILE: {"val_size": _change(3605, 10000)}}
    result = evaluate(diff, ignore=_re.compile(r".*_size$"))
    assert result.ok
    assert result.ignored == 1


def test_ignore_still_counts_new_file_metrics_as_ignored():
    # An added file whose keys are all ignored is not reported at all -- the
    # refresh gate asked for those keys specifically, so a missing baseline is
    # expected rather than a signal.
    diff = {SPLIT_METRICS_FILE: {"test_size": {"old": None, "new": 7209}}}
    result = evaluate(diff, ignore=".*_size$")
    assert result.ok
    assert result.ignored == 1
    assert result.rows == []


def test_markdown_notes_ignored_metrics():
    diff = {SPLIT_METRICS_FILE: {"test_size": _change(7209, 8000)}}
    markdown = render_markdown(
        evaluate(diff, ignore=".*_size$"), ignore_desc=".*_size$"
    )
    assert "Ignored 1 metric(s) matching `.*_size$`" in markdown
    assert "excluded from this comparison by design" in markdown


def test_markdown_omits_ignore_note_when_nothing_ignored():
    markdown = render_markdown(evaluate({}))
    assert "Ignored 0" not in markdown
    assert "Ignored" not in markdown


def test_markdown_reports_status_and_reason_columns():
    diff = {METRICS_FILE: {"percentages_pre_norm.emoji": _change(10.5, 12.0)}}
    markdown = render_markdown(evaluate(diff), base="origin/dev")
    assert "| Path | Metric | origin/dev | workspace | Change | Status |" in markdown
    assert "10.5" in markdown
    assert "12" in markdown
    assert f"FAIL ({REASON_DRIFT})" in markdown
    assert "**FAILED**" in markdown


def test_markdown_renders_missing_side_as_dash():
    diff = {METRICS_FILE: {"near_duplicates_removed": {"old": 2062, "new": None}}}
    markdown = render_markdown(evaluate(diff))
    assert "| 2062 | - |" in markdown
    assert f"FAIL ({REASON_REMOVED})" in markdown


def test_markdown_passes_when_within_tolerance():
    diff = {METRICS_FILE: {"percentages_pre_norm.emoji": _change(10.5, 10.502)}}
    markdown = render_markdown(evaluate(diff))
    assert "**PASSED**" in markdown
    assert FAIL not in markdown


def test_markdown_lists_unreadable_baseline_errors():
    result = evaluate({}, errors={"origin/dev": {METRICS_FILE: "config file error"}})
    markdown = render_markdown(result)
    assert "could not read the baseline" in markdown
    assert "config file error" in markdown


def test_gate_result_defaults_to_empty_errors():
    assert GateResult(rows=[]).errors == []


def test_unreadable_reason_constant_is_referenced_by_markdown():
    # Guards against the constant drifting away from the rendered copy.
    assert REASON_UNREADABLE == "unreadable baseline"
