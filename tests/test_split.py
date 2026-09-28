import pandas as pd
import pytest

from raay.data.split import build_split_metrics, perform_stratified_split
from raay.enums.constants import DefaultPaths


def _frame(n, label, dialect="msa"):
    return pd.DataFrame(
        {
            "text": [f"مراجعة رقم {i}" for i in range(n)],
            "label": [label] * n,
            "dialect": [dialect] * n,
        }
    )


def test_perform_stratified_split_sizes_and_stratification():
    df = pd.concat(
        [
            _frame(60, "positive"),
            _frame(30, "negative"),
            _frame(10, "neutral"),
        ],
        ignore_index=True,
    )
    train, val, test = perform_stratified_split(
        df, test_size=0.2, val_size=0.1, random_state=42
    )
    assert len(train) + len(val) + len(test) == len(df)
    # test_size and val_size are fractions of the *whole* dataset
    assert len(test) == pytest.approx(20, abs=1)
    assert len(val) == pytest.approx(10, abs=1)
    for part in (train, val, test):
        assert set(part["label"]) <= {"positive", "negative", "neutral"}


def test_build_split_metrics_reports_sizes_and_proportions():
    splits = {
        "train": _frame(70, "positive"),
        "val": _frame(20, "negative"),
        "test": _frame(10, "neutral", dialect="gulf"),
    }
    metrics = build_split_metrics(splits)

    assert metrics["train_size"] == 70
    assert metrics["val_size"] == 20
    assert metrics["test_size"] == 10

    labels = metrics["label_proportions"]
    assert labels["train"] == {"positive": 1.0}
    assert labels["val"] == {"negative": 1.0}
    assert labels["test"] == {"neutral": 1.0}

    dialects = metrics["dialect_proportions"]
    assert dialects["test"] == {"gulf": 1.0}
    assert dialects["train"] == {"msa": 1.0}


def test_build_split_metrics_proportions_sum_to_one():
    splits = {
        "train": pd.concat(
            [_frame(3, "positive"), _frame(1, "negative")], ignore_index=True
        ),
        "val": _frame(2, "positive"),
        "test": _frame(5, "neutral"),
    }
    metrics = build_split_metrics(splits)
    for part, counts in metrics["label_proportions"].items():
        assert sum(counts.values()) == pytest.approx(1.0), part


def test_build_split_metrics_rounds_to_five_dp():
    splits = {
        "train": pd.concat(
            [_frame(1, "positive"), _frame(2, "negative")], ignore_index=True
        )
    }
    proportions = build_split_metrics(splits)["label_proportions"]["train"]
    assert proportions == {
        "negative": pytest.approx(0.66667),
        "positive": pytest.approx(0.33333),
    }
    for value in proportions.values():
        # 5 decimal places -> stable strings for the +/-0.005 CI drift gate
        assert round(value, 5) == value


def test_build_split_metrics_handles_empty_split():
    metrics = build_split_metrics(
        {"train": _frame(3, "positive"), "test": _frame(0, "")}
    )
    assert metrics["test_size"] == 0
    assert metrics["label_proportions"]["test"] == {}
    assert metrics["dialect_proportions"]["train"] == {"msa": 1.0}


def test_split_metrics_path_is_git_tracked_not_cached():
    # dvc.yaml declares this file with `cache: false`, so it must be a
    # repo-relative path that git can see.
    assert DefaultPaths.SPLIT_METRICS.value == "reports/split_metrics.json"
    assert not DefaultPaths.SPLIT_METRICS.value.startswith("/")
