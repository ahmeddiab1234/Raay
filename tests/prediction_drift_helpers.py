"""Shared builders for the prediction-drift tests (hermetic, no repo data)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from raay.inference.prediction_drift import PredictionDriftSpec

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
