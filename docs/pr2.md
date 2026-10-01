# PR #2 — Data Collection & Preprocessing Pipeline

## Summary

This phase focuses on acquiring the raw datasets and implementing a reproducible, version-controlled pipeline for data preprocessing and splitting. It integrates DVC for data versioning and MLFlow for experiment tracking.

---

## What Was Done

### 1. Data Collection

- Acquired two main datasets:
  - `arabic_sentiment_reviews.csv` (330K Arabic Sentiment Reviews)
  - `Final_Data.csv` (Arabic Customer Reviews)
- Configured **DVC** to track the raw datasets (`data/raw/`) while keeping them excluded from Git tracking via `.gitignore`.
- Set up a DVC remote storage (`~/dvc-storage/raay`) and pushed the initial data versions.

### 2. Preprocessing & Splitting Pipeline (DVC)

- Established a two-stage reproducible DVC pipeline (`dvc.yaml`):
  - **`preprocess` stage**: Runs `src/raay/data/preprocess.py` on `Final_Data.csv` to generate `data/interim/normalized.csv`. It applies Arabic text normalization, removes elongation, and filters rows by character length.
  - **`split` stage**: Runs `src/raay/data/split.py` on the normalized data to create `train.csv`, `val.csv`, and `test.csv` in `data/processed/`.
- Both stages utilize configuration parameters defined in `params.yaml` (e.g., `min_char_length`, `test_size`, `random_state`).
- Pipeline metrics (such as row counts and duplicate drops) are output to `reports/preprocess_metrics.json`.
- The entire pipeline is fully reproducible using `dvc repro`.

### 3. Experiment Tracking (MLFlow)

- Integrated **MLFlow** within the data processing scripts to automatically track parameters and metrics.
- Created the `raay_preprocessing` experiment, logging:
  - Pipeline configurations (min/max lengths, test sizes).
  - Data quality metrics (exact/fuzzy duplicate counts, row counts before and after processing).
  - Train, validation, and test split sizes.
- Tracking data can be analyzed locally via `uv run mlflow ui`.

---

## What's Next

- Proceed with Exploratory Data Analysis (EDA) on the normalized dataset.
- Fine-tune AraBERT on the processed training splits.
