# PR #3 — Modeling Baseline (AraBERT Fine-Tuning)

## Summary
This phase establishes the baseline sentiment classification model by fine-tuning **AraBERT v2** (`aubmindlab/bert-base-arabertv02`) on the preprocessed splits. Training ran on Kaggle (GPU T4) via a sweep script; the best run was registered in MLflow and evaluated on the held-out dialect-stratified test set.

---

## What Was Done

### 1. Training Script (`src/raay/training/train.py`)
- Hydra-driven config (`configs/train.yaml`) with overridable `lr` and `batch_size`.
- Applies `ArabertPreprocessor` before tokenization.
- HuggingFace `Trainer` with checkpoint saves every `save_steps`.
- Logs params, metrics, tokenizer, and model artifacts to the `raay_training` MLflow experiment.

### 2. Sweep on Kaggle (`scripts/kaggle_train_runs.py`)
- Cloned `modeling-baseline` branch onto Kaggle (GPU T4 accelerator).
- Ran **6 training runs** varying `lr` / `batch_size`.
- Best run registered as `ArabicSentiment → Production` in the MLflow Model Registry.
- MLflow `mlruns/` zipped and downloaded, paths rewritten from Kaggle absolutes → local `file:./mlflow/mlruns`, then merged into the local store and migrated to `sqlite:///mlflow.db`.

### 3. Evaluation (`src/raay/training/evaluate.py`)
- Ran on Kaggle against `data/processed/test.csv` (n = 7,209).
- Output written to `reports/eval_baseline.json`.

---

## Results (`reports/eval_baseline.json`)

| Metric | Value |
|---|---|
| Accuracy | **84.9 %** |
| F1 (macro) | **0.641** |
| F1 (weighted) | 0.841 |

**Per-class F1:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Positive | 0.880 | 0.909 | **0.895** | 4 152 |
| Negative | 0.846 | 0.853 | **0.850** | 2 691 |
| Neutral | 0.245 | 0.139 | **0.178** | 366 |

> Neutral is heavily underperforming due to class imbalance (only ~5 % of test set).

**Dialect breakdown (accuracy / F1 macro):**

| Dialect | n | Accuracy | F1 macro |
|---|---|---|---|
| MSA | 3 322 | 85.7 % | 0.605 |
| Gulf | 1 145 | 86.2 % | 0.682 |
| Egyptian | 1 011 | 82.4 % | 0.647 |
| Levantine | 1 129 | 82.4 % | 0.631 |
| Maghrebi | 357 | 92.2 % | 0.668 |
| Arabizi | 245 | 80.0 % | 0.538 |

---

## What's Next
- Address neutral-class imbalance (oversampling / class weights).
- Improve Arabizi coverage (possibly dedicated pre-processing or data augmentation).
- Build inference and serving layer (`inference/`, `serving/`).
