import json
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from loguru import logger
from rapidfuzz import fuzz, process

import mlflow
from raay.config.env import load_environment
from raay.enums.constants import DefaultPaths, Experiments

# Regex patterns
ARABIC_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670]")
LATIN_CHARS = re.compile(r"[a-zA-Z]")
EMOJI_PATTERN = re.compile(r"[\U00010000-\U0010ffff]", flags=re.UNICODE)
ELONGATION_PATTERN = re.compile(r"(.)\1{2,}")


def load_config(config_path: str = DefaultPaths.PARAMS.value) -> dict[str, Any]:
    """Load preprocessing configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def flag_near_empty(text: str, min_length: int) -> bool:
    """Flag texts that have fewer than min_length characters."""
    return len(str(text).strip()) < min_length


def remove_diacritics(text: str) -> str:
    """Remove Arabic diacritics (tashkeel)."""
    return ARABIC_DIACRITICS.sub("", str(text))


def collapse_elongation(text: str, max_repeat: int = 2) -> str:
    """Collapse repeated characters to max_repeat."""
    if pd.isna(text):
        return text
    # The regex replaces any character repeated 3 or more times with exactly 2 instances.
    # To support dynamic max_repeat, we construct the regex.
    pattern = r"(.)\1{" + str(max_repeat) + r",}"
    replacement = r"\1" * max_repeat
    return re.sub(pattern, replacement, str(text))


def normalize_casing(text: str) -> str:
    """Normalize casing (lowercase) for Latin characters."""
    return str(text).lower()


def has_latin(text: str) -> bool:
    """Check if text contains Latin characters."""
    return bool(LATIN_CHARS.search(str(text)))


def has_emoji(text: str) -> bool:
    """Check if text contains emojis."""
    return bool(EMOJI_PATTERN.search(str(text)))


def has_diacritics(text: str) -> bool:
    """Check if text contains Arabic diacritics."""
    return bool(ARABIC_DIACRITICS.search(str(text)))


def has_elongation(text: str) -> bool:
    """Check if text contains elongated characters."""
    return bool(ELONGATION_PATTERN.search(str(text)))


def fuzzy_deduplicate(
    df: pd.DataFrame, text_col: str, threshold: float
) -> pd.DataFrame:
    """Remove fuzzy near-duplicates using rapidfuzz."""
    texts = df[text_col].tolist()
    indices = df.index.tolist()

    kept_indices: list[int] = []
    kept_texts: list[str] = []

    threshold_score = threshold * 100 if threshold <= 1.0 else threshold

    logger.info(f"Starting fuzzy deduplication with threshold {threshold_score}...")

    for idx, text in zip(indices, texts):
        if not kept_texts:
            kept_texts.append(text)
            kept_indices.append(idx)
            continue

        match = process.extractOne(
            text, kept_texts, scorer=fuzz.ratio, score_cutoff=threshold_score
        )
        if not match:
            kept_texts.append(text)
            kept_indices.append(idx)

    return df.loc[kept_indices]


def main():
    load_environment()

    # Set up logging
    log_dir = Path(DefaultPaths.LOGS.value)
    log_dir.mkdir(exist_ok=True)
    logger.add(log_dir / f"preprocess_{int(time.time())}.log")

    config = load_config()
    prep_config = config.get("preprocessing", {})
    min_char_length = prep_config.get("min_char_length", 10)
    elongation_max_repeat = prep_config.get("elongation_max_repeat", 2)
    dedup_similarity_threshold = prep_config.get("dedup_similarity_threshold", 0.9)

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

    raw_row_count = len(df)
    logger.info(f"Raw row count: {raw_row_count}")

    # MLflow tracking
    mlflow.set_experiment(Experiments.PREPROCESSING.value)
    with mlflow.start_run(run_name="preprocessing_pipeline"):
        mlflow.log_params(prep_config)

        # 1. Near-empty flag
        df["is_near_empty"] = df["text"].apply(
            lambda x: flag_near_empty(x, min_char_length)
        )
        near_empty_count = int(df["is_near_empty"].sum())
        logger.info(f"Near-empty flagged count: {near_empty_count}")

        # Calculate pre-normalization stats for metrics
        pct_latin = df["text"].apply(has_latin).mean() * 100
        pct_diacritics = df["text"].apply(has_diacritics).mean() * 100
        pct_elongation = df["text"].apply(has_elongation).mean() * 100
        pct_emoji = df["text"].apply(has_emoji).mean() * 100

        logger.info(
            f"Pre-normalization stats: {pct_latin:.2f}% Latin, {pct_diacritics:.2f}% Diacritics, {pct_elongation:.2f}% Elongated, {pct_emoji:.2f}% Emoji"
        )

        # 2. Diacritics stripping
        logger.info("Stripping Arabic diacritics...")
        df["text"] = df["text"].apply(remove_diacritics)

        # 3. Elongated character collapsing
        logger.info(
            f"Collapsing elongated characters to max {elongation_max_repeat}..."
        )
        df["text"] = df["text"].apply(
            lambda x: collapse_elongation(x, elongation_max_repeat)
        )

        # 4. Latin character handling (normalize casing)
        logger.info("Normalizing Latin casing...")
        df["text"] = df["text"].apply(normalize_casing)

        # 5. Emoji handling
        # User explicitly requested to keep emojis, so we do nothing here.

        # 6. Deduplication
        # Exact first
        pre_exact_count = len(df)
        df = df.drop_duplicates(subset=["text"])
        exact_dupes_removed = pre_exact_count - len(df)
        logger.info(f"Exact duplicates removed: {exact_dupes_removed}")

        # Fuzzy near-duplicate
        pre_fuzzy_count = len(df)
        df = fuzzy_deduplicate(df, "text", dedup_similarity_threshold)
        near_dupes_removed = pre_fuzzy_count - len(df)
        logger.info(f"Fuzzy near-duplicates removed: {near_dupes_removed}")

        final_row_count = len(df)
        logger.info(f"Final row count: {final_row_count}")

        # Save output
        logger.info(f"Saving normalized data to {interim_data_path}")
        df.to_csv(interim_data_path, index=False)

        # Save metrics
        metrics = {
            "raw_row_count": raw_row_count,
            "near_empty_flagged_count": near_empty_count,
            "exact_duplicates_removed": exact_dupes_removed,
            "near_duplicates_removed": near_dupes_removed,
            "final_row_count": final_row_count,
            "percentages_pre_norm": {
                "latin_chars": round(pct_latin, 2),
                "diacritics": round(pct_diacritics, 2),
                "elongated": round(pct_elongation, 2),
                "emoji": round(pct_emoji, 2),
            },
        }

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

        logger.info(f"Metrics saved to {metrics_path}")
        mlflow.log_metrics(
            {
                "raw_rows": raw_row_count,
                "final_rows": final_row_count,
                "exact_dupes": exact_dupes_removed,
                "fuzzy_dupes": near_dupes_removed,
            }
        )


if __name__ == "__main__":
    main()
