"""Prediction drift: the output side of Phase 6 monitoring.

Hermetic by AGENTS.md rule -- panels are built here, never read from
``data/``. The two tests that matter most are
``test_clean_panel_passes_against_the_real_prior`` and
``test_brief_prior_45_35_20_would_fail_a_clean_panel``: together they pin the
one decision in this step that a well-meaning edit could silently undo.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from raay.enums.constants import DefaultPaths
from raay.inference.prediction_drift import (
    LABELS,
    PredictionDriftSpec,
    class_distribution,
    classify_triage,
    collect_confidence_history,
    confidence_z_score,
    mean_confidence,
    mlflow_metrics,
    predict_drift_check,
    rolling_baseline,
    share_delta_pp,
    training_label_reference,
)

# The measured real proportions of data/processed/train.csv.
REAL_PRIOR = {"positive": 0.576, "negative": 0.373, "neutral": 0.051}
# What the brief specified. Does not describe this dataset.
BRIEF_PRIOR = {"positive": 0.45, "negative": 0.35, "neutral": 0.20}


def panel(mix: dict[str, float], n: int = 3000, score: float = 0.9) -> pd.DataFrame:
    """A scored-output-shaped frame with an exact class mix.

    ``n`` large enough that PSI on a 3-class mix is not dominated by the
    counting noise of a tiny sample.
    """
    rows: list[str] = []
    for label, share in mix.items():
        rows += [label] * round(share * n)
    return pd.DataFrame(
        {
            "text": [f"review {i}" for i in range(len(rows))],
            "predicted_label": rows,
            "predicted_score": np.full(len(rows), score),
        }
    )


def train_csv(
    tmp_path: Path, mix: dict[str, float] = REAL_PRIOR, name: str = "train"
) -> str:
    """A stand-in for data/processed/train.csv with the given label mix.

    ``name`` keeps two priors in one test from overwriting each other -- the
    brief-prior test needs both, and a shared filename silently reads whichever
    was written last.
    """
    rows: list[str] = []
    for label, share in mix.items():
        rows += [label] * round(share * 2000)
    path = tmp_path / f"{name}.csv"
    pd.DataFrame({"text": [f"t{i}" for i in range(len(rows))], "label": rows}).to_csv(
        path, index=False
    )
    return str(path)


def write_csv(df: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return str(path)


def spec(tmp_path: Path, current: pd.DataFrame, **kw) -> PredictionDriftSpec:
    """A spec wired to temporary files; no repository paths involved."""
    defaults = {
        "current_csv": write_csv(current, tmp_path / "current.csv"),
        "reference_csv": write_csv(panel(REAL_PRIOR), tmp_path / "reference.csv"),
        "train_prior_csv": train_csv(tmp_path),
        "day": "2026-01-01",
        "min_days": 3,
    }
    defaults.update(kw)
    return PredictionDriftSpec(**defaults)  # type: ignore[arg-type]


# --- training_label_reference ------------------------------------------


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


# --- class distribution --------------------------------------------------


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


# --- the reference-prior decision (the load-bearing tests) --------------


def test_clean_panel_passes_against_the_real_prior(tmp_path: Path) -> None:
    """A panel shaped like the model actually predicts must PASS.

    Guards the measured ~0.02 baseline: the model under-predicts Neutral, so
    a little PSI is expected, and the threshold must stay above it.
    """
    # What the model really emits on a clean panel (59.5/37.6/2.9), not the
    # ideal prior -- using the ideal here would not reproduce the floor.
    clean = panel({"positive": 0.595, "negative": 0.376, "neutral": 0.029})
    report = predict_drift_check(spec(tmp_path, clean))
    cd = report["class_distribution"]
    assert cd["psi_vs_training_prior"]["decision"] == "PASS"
    assert cd["psi_vs_training_prior"]["drift_score"] < 0.1
    assert report["output_drift"]["overall"] == "PASS"


def test_brief_prior_45_35_20_would_fail_a_clean_panel(tmp_path: Path) -> None:
    """Documents *why* the brief's prior is not used.

    If a future edit reintroduces 45/35/20 as the reference, this pins the
    consequence: the same clean panel that PASSes above reads 0.41 and fails,
    every night, with nothing actually wrong.
    """
    clean = panel({"positive": 0.595, "negative": 0.376, "neutral": 0.029})
    result = predict_drift_check(
        spec(
            tmp_path,
            clean,
            train_prior_csv=train_csv(tmp_path, BRIEF_PRIOR, name="train_brief"),
        )
    )
    score = result["class_distribution"]["psi_vs_training_prior"]["drift_score"]
    assert score >= 0.2
    assert result["output_drift"]["overall"] == "FAIL"


def test_neutral_heavy_panel_fails_against_the_real_prior(tmp_path: Path) -> None:
    """The gate must still have teeth with the corrected prior."""
    shifted = panel({"positive": 0.35, "negative": 0.30, "neutral": 0.35})
    report = predict_drift_check(spec(tmp_path, shifted))
    assert report["class_distribution"]["psi_vs_training_prior"]["decision"] == "FAIL"


def test_reference_panel_comparison_is_reported_alongside(tmp_path: Path) -> None:
    clean = panel({"positive": 0.595, "negative": 0.376, "neutral": 0.029})
    report = predict_drift_check(spec(tmp_path, clean))
    cd = report["class_distribution"]
    assert cd["psi_vs_reference_panel"]["decision"] == "PASS"
    assert cd["training_prior_source"].endswith("train.csv")


# --- confidence series ---------------------------------------------------


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


def test_confidence_falling_is_flagged_and_escalation_needs_both(
    tmp_path: Path,
) -> None:
    """`escalate` is a conjunction: a falling mean alone is not escalation."""
    lower = panel(REAL_PRIOR, score=0.70)
    report = predict_drift_check(spec(tmp_path, lower))
    assert report["confidence"]["falling"] is True
    assert report["escalate"] is False  # class PSI is still PASS

    shifted = panel({"positive": 0.35, "negative": 0.30, "neutral": 0.35}, score=0.70)
    report = predict_drift_check(spec(tmp_path, shifted))
    assert report["confidence"]["falling"] is True
    assert report["escalate"] is True  # both halves present


def test_mean_confidence_requires_the_column() -> None:
    with pytest.raises(KeyError, match="predicted_score"):
        mean_confidence(pd.DataFrame({"predicted_label": ["positive"]}))


# --- triage --------------------------------------------------------------


@pytest.mark.parametrize(
    ("input_verdict", "output_verdict", "expected"),
    [
        ("PASS", "PASS", "stable"),
        ("FAIL", "PASS", "world_changed"),
        ("PASS", "FAIL", "model_degraded"),
        ("FAIL", "FAIL", "ambiguous"),
        ("WARN", "PASS", "world_changed"),
        ("PASS", "WARN", "model_degraded"),
        (None, "PASS", "indeterminate"),
        (None, "FAIL", "indeterminate"),
    ],
)
def test_classify_triage_matrix(
    input_verdict: str | None, output_verdict: str, expected: str
) -> None:
    assert classify_triage(input_verdict, output_verdict) == expected


def test_triage_pairs_with_the_same_days_input_drift_report(
    tmp_path: Path,
) -> None:
    drift_report = tmp_path / "reports" / "drift" / "2026-01-01.json"
    drift_report.parent.mkdir(parents=True)
    drift_report.write_text(json.dumps({"overall": "FAIL"}))
    clean = panel({"positive": 0.595, "negative": 0.376, "neutral": 0.029})
    report = predict_drift_check(
        spec(tmp_path, clean, input_drift_report=str(drift_report))
    )
    assert report["input_drift"]["overall"] == "FAIL"
    assert report["triage"] == "world_changed"


def test_missing_input_report_is_indeterminate_not_a_silent_pass(
    tmp_path: Path,
) -> None:
    """Attribution needs both halves; a missing half must not read as PASS."""
    clean = panel({"positive": 0.595, "negative": 0.376, "neutral": 0.029})
    report = predict_drift_check(
        spec(tmp_path, clean, input_drift_report=str(tmp_path / "nope.json"))
    )
    assert report["input_drift"]["overall"] is None
    assert report["triage"] == "indeterminate"


def test_unreadable_input_report_is_indeterminate(tmp_path: Path) -> None:
    bad = tmp_path / "drift.json"
    bad.write_text("{oops")
    clean = panel({"positive": 0.595, "negative": 0.376, "neutral": 0.029})
    report = predict_drift_check(spec(tmp_path, clean, input_drift_report=str(bad)))
    assert report["triage"] == "indeterminate"


def test_report_carries_the_caveat(tmp_path: Path) -> None:
    clean = panel(REAL_PRIOR)
    report = predict_drift_check(spec(tmp_path, clean))
    assert "unverified against production traffic" in report["caveat"]


# --- MLflow --------------------------------------------------------------


def test_mlflow_metrics_skip_none_rather_than_logging_zero(
    tmp_path: Path,
) -> None:
    """A cold-start z-score must not chart as a healthy 0.0."""
    clean = panel(REAL_PRIOR)
    report = predict_drift_check(spec(tmp_path, clean))
    metrics = mlflow_metrics(report)
    assert "pred_confidence_z_score" not in metrics
    assert metrics["pred_mean_confidence"] == pytest.approx(0.9)


def test_mlflow_metrics_include_class_shares(tmp_path: Path) -> None:
    clean = panel({"positive": 0.595, "negative": 0.376, "neutral": 0.029})
    metrics = mlflow_metrics(predict_drift_check(spec(tmp_path, clean)))
    assert metrics["pred_class_share_neutral"] == pytest.approx(0.029, abs=1e-3)
    assert metrics["pred_psi_vs_train_prior"] > 0
