"""The ``preprocess`` DVC stage: raw CSV -> ``data/interim/normalized.csv``.

    python -m raay.data.preprocess      # invoked by ``dvc repro preprocess``

Reads ``data/raw/Final_Data.csv``, renames ``review_description -> text`` and
``rating -> label``, runs the cleaning pipeline in
``raay.data.normalize_pipeline``, then writes the interim frame plus
``reports/preprocess_metrics.json`` (``cache: false``, so it is git-tracked and
its md5 lands in ``dvc.lock``). Every run is logged to the ``raay_preprocessing``
MLflow experiment.

The text primitives live in ``raay.data.text_clean`` and are re-exported here so
``from raay.data.preprocess import remove_diacritics`` keeps resolving.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import yaml
from loguru import logger

from raay.config.env import load_environment
from raay.data.normalize_pipeline import normalize_frame
from raay.data.text_clean import (
    ARABIC_DIACRITICS,
    ELONGATION_PATTERN,
    EMOJI_PATTERN,
    LATIN_CHARS,
    collapse_elongation,
    flag_near_empty,
    fuzzy_deduplicate,
    has_diacritics,
    has_elongation,
    has_emoji,
    has_latin,
    normalize_casing,
    remove_diacritics,
)
from raay.enums.constants import DefaultPaths, Experiments

__all__ = [
    "ARABIC_DIACRITICS",
    "ELONGATION_PATTERN",
    "EMOJI_PATTERN",
    "LATIN_CHARS",
    "collapse_elongation",
    "flag_near_empty",
    "fuzzy_deduplicate",
    "has_diacritics",
    "has_elongation",
    "has_emoji",
    "has_latin",
    "load_config",
    "main",
    "normalize_casing",
    "remove_diacritics",
]


def load_config(config_path: str = DefaultPaths.PARAMS.value) -> dict[str, Any]:
    """Load preprocessing configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    """The ``preprocess`` DVC stage: read raw, clean, write interim + metrics."""
    load_environment()

    # Set up logging
    log_dir = Path(DefaultPaths.LOGS.value)
    log_dir.mkdir(exist_ok=True)
    logger.add(log_dir / f"preprocess_{int(time.time())}.log")

    config = load_config()
    prep_config = config.get("preprocessing", {})

    raw_data_path = Path(DefaultPaths.RAW_DATA.value)
    interim_data_path = Path(DefaultPaths.INTERIM_DATA.value)
    metrics_path = Path(DefaultPaths.PREPROCESS_METRICS.value)

    interim_data_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading raw data from {raw_data_path}")
    df = pd.read_csv(raw_data_path)

    # Rename actual columns to expected 'text' and 'label'
    df = df.rename(columns={"review_description": "text", "rating": "label"})

    # Ensure expected columns
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"Dataset missing expected columns. Found: {df.columns.tolist()}"
        )

    mlflow.set_experiment(Experiments.PREPROCESSING.value)
    with mlflow.start_run(run_name="preprocessing_pipeline"):
        mlflow.log_params(prep_config)

        df, metrics = normalize_frame(df, prep_config)

        logger.info(f"Saving normalized data to {interim_data_path}")
        df.to_csv(interim_data_path, index=False)

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

        logger.info(f"Metrics saved to {metrics_path}")
        mlflow.log_metrics(
            {
                "raw_rows": metrics["raw_row_count"],
                "final_rows": metrics["final_row_count"],
                "exact_dupes": metrics["exact_duplicates_removed"],
                "fuzzy_dupes": metrics["near_duplicates_removed"],
            }
        )


if __name__ == "__main__":
    main()
