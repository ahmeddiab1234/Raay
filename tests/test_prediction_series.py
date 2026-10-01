"""Prediction-drift series helpers: priors, class mix, confidence history."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from prediction_drift_helpers import REAL_PRIOR, train_csv

from raay.enums.constants import DefaultPaths
from raay.inference.prediction_drift import (
    LABELS,
    class_distribution,
    collect_confidence_history,
    confidence_z_score,
    mean_confidence,
    rolling_baseline,
    share_delta_pp,
    training_label_reference,
)


def test_training_label_reference_uses_the_train_split(tmp_path: Path) -> None:
    frame = training_label_reference(train_csv(tmp_path))
    assert set(frame["predicted_label"]) == set(LABELS)
    assert len(frame) == 2000


def test_training_label_reference_names_the_missing_class(tmp_path: Path) -> None:
    """A prior without a class cannot be compared; it must say which one."""
    partial = pd.DataFrame({"label": ["positive"] * 50 + ["negative"] * 50})
    path = tmp_path / "partial.csv"
    partial.to_csv(path, index=False)
    with pytest.raises(ValueError, match="neutral"):
        training_label_reference(path)


def test_training_label_reference_defaults_to_the_repo_train_split() -> None:
    """The default is the real DVC-tracked split, not a constant."""
    assert DefaultPaths.TRAIN_SPLIT.value == "data/processed/train.csv"


def test_class_distribution_zero_fills_an_absent_class() -> None:
    """Neutral vanishing must read 0.0, not vanish from the record."""
    frame = pd.DataFrame({"predicted_label": ["positive"] * 10 + ["negative"] * 5})
    dist = class_distribution(frame)
    assert dist["neutral"] == 0.0
    assert sum(dist.values()) == pytest.approx(1.0)


def test_class_distribution_requires_the_column() -> None:
    with pytest.raises(KeyError, match="predicted_label"):
        class_distribution(pd.DataFrame({"label": ["positive"]}))


def test_share_delta_pp_is_signed_and_in_points() -> None:
    delta = share_delta_pp(
        {"positive": 0.5, "negative": 0.3, "neutral": 0.2}, REAL_PRIOR
    )
    assert delta["neutral"] == pytest.approx(round((0.2 - 0.051) * 100, 3))
    assert delta["positive"] == pytest.approx(round((0.5 - 0.576) * 100, 3))


def test_rolling_baseline_withholds_z_score_until_min_days() -> None:
    base = rolling_baseline([0.90, 0.91], window=14, min_days=7)
    assert base["sufficient_history"] is False
    assert base["z_score"] is None
    assert "need >= 7 days" in base["reason"]


def test_rolling_baseline_handles_identical_history() -> None:
    """Zero spread means the z-score is undefined, not zero."""
    base = rolling_baseline([0.9] * 10, window=14, min_days=3)
    assert base["sufficient_history"] is True
    assert base["z_score"] is None
    assert "std is 0" in base["reason"]


def test_confidence_z_score_math() -> None:
    base = {"sufficient_history": True, "mean": 0.9, "std": 0.02}
    assert confidence_z_score(0.94, base) == pytest.approx(2.0)
    assert confidence_z_score(0.9, base) == pytest.approx(0.0)
    assert confidence_z_score(0.94, {"sufficient_history": False}) is None


def test_collect_confidence_history_skips_reports_without_the_key(
    tmp_path: Path,
) -> None:
    """The 2026-09-23/24 reports predate ``feature_means``.

    A reader that indexed the key instead of getting it would raise on the
    first real history it met.
    """
    directory = tmp_path / "reports"
    directory.mkdir()
    (directory / "2026-09-23.json").write_text(json.dumps({"overall": "PASS"}))
    (directory / "2026-09-24.json").write_text(
        json.dumps({"confidence": {"mean": 0.91}})
    )
    (directory / "broken.json").write_text("{not json")
    assert collect_confidence_history(directory) == [0.91]


def test_collect_confidence_history_excludes_the_day_being_scored(
    tmp_path: Path,
) -> None:
    """Today's own report must not become its own baseline."""
    directory = tmp_path / "reports"
    directory.mkdir()
    for day, mean in [("2026-01-01", 0.5), ("2026-01-02", 0.6)]:
        (directory / f"{day}.json").write_text(
            json.dumps({"confidence": {"mean": mean}})
        )
    assert collect_confidence_history(directory, exclude="2026-01-02") == [0.5]


def test_mean_confidence_requires_the_column() -> None:
    with pytest.raises(KeyError, match="predicted_score"):
        mean_confidence(pd.DataFrame({"predicted_label": ["positive"]}))
