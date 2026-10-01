<div align="center">

# Raay راي

**Arabic E-Commerce Product Review Sentiment Analysis**

An end-to-end MLOps pipeline for classifying Arabic product reviews (Positive / Negative / Neutral) at scale — supporting Modern Standard Arabic and regional dialects (Egyptian, Gulf, Levantine).

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/package%20manager-uv-blueviolet)](https://docs.astral.sh/uv/)
[![Ruff](https://img.shields.io/badge/linter-ruff-orange)](https://docs.astral.sh/ruff/)
[![MLflow](https://img.shields.io/badge/tracking-MLflow-0194E2)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/data%20versioning-DVC-945DD6)](https://dvc.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Architecture](#project-architecture)
- [Datasets](#datasets)
- [Installation &amp; Usage](#installation--usage)
- [Data Pipeline](#data-pipeline)
- [Phase 3: Modeling Baseline](#phase-3-modeling-baseline)
- [Phase 4: Compression &amp; Optimization](#phase-4-compression--optimization)
- [Phase 5: Serving &amp; Operations](#phase-5-serving--operations)
- [Phase 6: CI/CD, Deployment &amp; Automation](#phase-6-cicd-deployment--automation)
- [Testing](#testing)
- [Limitations &amp; Honest Findings](#limitations--honest-findings)
- [Environment Variables](#environment-variables)
- [License](#license)

---

## Overview

**Raay** (Arabic: _رأي_, meaning "opinion") is a production-grade sentiment analysis system designed for Arab e-commerce platforms. It classifies ~50,000 Arabic product reviews per day into three sentiment classes — **Positive**, **Negative**, and **Neutral** — to power product ranking, seller rating aggregation, and customer-service ticket prioritization.

The project covers the full ML lifecycle: data versioning, experiment tracking, model training, inference optimization, and model serving.

**Current state:** an AraBERT teacher fine-tuned to **84.92 %** accuracy, distilled to a 6-layer student, exported to ONNX, dynamically quantized to INT8 (**85.03 %** accuracy, **136 MB**, **p50 ≈ 12 ms** single-call on CPU), served through a containerized BentoML API, re-scored nightly and watched by a **three-stage drift chain** — engineered input features, prediction drift with a triage that names which half moved, then a retrain trigger that dispatches to GitHub — rolled out via a **shadow-then-canary** deployment (5/95 → 100 % on Prometheus-fed stage gates), fed by an authenticated **customer-service feedback loop**, and guarded end to end by an automated **CI/CD pipeline** — a ±0.005 split-metrics gate on every PR, a Docker image built only on `main`, staged deploys with automatic rollback, a 14-gate promotion gate that is the only mover of the `Production` alias, and weekly retrain hooks that turn newly versioned data into a reviewable PR.

---

## Features

| Category                         | Details                                                                                                        |
| -------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| **3-Class Sentiment**      | Positive · Negative · Neutral classification with confidence scores                                          |
| **Arabic NLP**             | MSA + dialect support (Egyptian, Gulf, Levantine, Maghrebi, Arabizi/franco-arabe) with per-row dialect tagging |
| **Transformer-based**      | AraBERT v2 backbone with HuggingFace Transformers & Accelerate                                                 |
| **Model Optimization**     | Knowledge distillation (6-layer student), ONNX export, dynamic INT8 quantization                               |
| **Real-Time Serving**      | BentoML REST API (`/predict`, `/health`), containerized via Docker Compose                                 |
| **Near-Real-Time Scoring** | Client-side Redis micro-batch consumer (drain-by-size*or* drain-by-time)                                     |
| **Batch Scoring & Drift**  | Nightly re-scoring + Evidently PSI drift gate over 17 engineered columns (incl. a frozen PCA of AraBERT embeddings) |
| **Prediction Drift**      | Class-mix + confidence gates against two references, with a 2×2 triage naming *which half* moved          |
| **Retrain Trigger**       | Drift breach or confirmed seasonal event → GitHub `repository_dispatch`, with three deliberate non-firers |
| **Orchestration**          | Airflow DAG (daily 03:00 UTC) driving input → score → drift → prediction-drift → trigger → feedback      |
| **Canary Rollout**         | Standalone nginx-fronted project: shadow-mirror then 5/95→100 %, gated on Prometheus |
| **Scheduled Retraining**   | Weekly `retrain.yml` re-runs DVC on newly versioned data, gates proportions, opens a `dev` PR |
| **Human Feedback Loop**     | Authenticated CS override capture, two-agent QA ladder, train-only merge of hard negatives |
| **Experiment Tracking**    | MLflow runs, metrics, artifacts, and a Model Registry with `Production` / `Canary` aliases                  |
| **Data Versioning**        | DVC-tracked raw/interim/processed data with a configurable remote (local / S3)                                 |
| **Code Quality**           | Ruff linter & formatter, mypy static type checking, pre-commit hooks (pre-commit + pre-push)                   |
| **Testing**                | pytest suite (704 unit tests, hermetic — no GPU, no network, no servers)                                  |
| **Configuration**          | Hydra for training/distillation; `params.yaml` for the DVC data stages                                        |
| **Typed Schemas**          | Pydantic I/O models for the serving contract                                                                   |
| **Structured Logging**     | Loguru for structured, leveled logging                                                                         |

---

## Tech Stack

| Layer                | Tool                                          |
| -------------------- | --------------------------------------------- |
| Language             | Python 3.12+                                  |
| Package Manager      | [uv](https://docs.astral.sh/uv/)               |
| Deep Learning        | PyTorch, HuggingFace Transformers, Accelerate |
| Arabic Preprocessing | `arabert` (`ArabertPreprocessor`)         |
| Experiment Tracking  | MLflow (local SQLite store, `mlflow.db`)     |
| Data Versioning      | DVC (local / S3 remote)                       |
| Inference Runtime    | ONNX Runtime (FP32 + dynamic INT8)            |
| Model Serving        | BentoML + Docker Compose                      |
| Load Testing         | Locust (HTTP before/after)                    |
| Streaming / Queue    | Redis (`redis:7-alpine`)                    |
| Drift Monitoring     | Evidently (PSI) over 17 engineered columns        |
| Drift Embeddings     | AraBERT encoder + frozen PCA basis (fit once on the reference) |
| Orchestration        | Apache Airflow 2.10.5 (host-isolated)         |
| Traffic Shifting     | nginx (weighted upstream)                     |
| CI/CD & Triggers     | GitHub Actions (Actions, reusable composite action) |
| Config Management    | Hydra (`configs/`) + `params.yaml`        |
| Data Validation      | Pydantic                                      |
| Linting & Formatting | Ruff                                          |
| Type Checking        | mypy                                          |
| Testing              | pytest                                        |
| Logging              | Loguru                                        |

---

## Project Architecture

```
raay/
├── src/raay/                       # Main Python package
│   ├── data/                       #   Preprocessing, splitting, dialect, feedback QA
│   ├── training/                   #   Fine-tuning, distillation, evaluation
│   ├── inference/                  #   ONNX export/quantize, queue consumer, batch scoring,
│   │                               #     drift_features · prediction_drift · retrain_trigger
│   ├── serving/                    #   BentoML service, benchmark, canary agent, feedback capture
│   ├── config/                     #   Env loading, data config
│   └── enums/                      #   Shared constants (paths, experiments, models)
│
├── airflow/dags/                   # Nightly DAG: score → drift → prediction-drift → trigger → feedback-merge
├── configs/                        # Hydra configs (train.yaml, distill.yaml) + seasonal_events.yaml
├── deploy/                         # nginx_canary.conf (generated front) · prometheus.yml
│
├── data/                           # DVC-tracked (git-ignored)
│   ├── raw/                        #   Source CSVs (.dvc pointers committed)
│   ├── interim/                    #   normalized.csv
│   ├── processed/                  #   train/val/test splits + train_feedback.csv (train-only)
│   ├── feedback/                   #   raw/ captures · reviewed/ (DVC pointer, human-owned)
│   └── scoring/                    #   input/ · output/ · reference/ panels + frozen pca_basis.joblib
│
├── models/                         # Trained checkpoints + ONNX graphs
│   ├── baseline/ · distilled/      #   HF checkpoints
│   └── onnx/                       #   model.onnx · distilled.onnx · model_int8.onnx
│
├── reports/                        # Committed evidence (evals, benchmarks, drift/,
│                                   #   prediction_drift/, retrain_trigger/, locust, feedback_metrics)
├── scripts/                        # Sweeps, benchmarks, registry + canary + promote ops
├── tests/                          # Unit tests (hermetic)
├── notebooks/                      # EDA
├── docs/                           # pr1–pr7 write-ups · labeling_guidelines.md · commands_run_p*.txt
│
├── bentofile.yaml                  # Bento build recipe (serving-only deps + baked int8)
├── docker-compose.yml              # prod worker + redis
├── docker-compose.canary.yml       # standalone shadow/canary project (nginx + 2 workers + agent + Prometheus)
├── dvc.yaml · params.yaml          # Data pipeline stages + their parameters
├── .dvc/                           # DVC config & cache
├── .pre-commit-config.yaml         # ruff, ruff-format, mypy, DVC hooks
├── pyproject.toml · uv.lock        # Dependencies
└── AGENTS.md                       # Hard-won project notes for AI agents
```

> **Note:** `data/**` and model weights are git-ignored — only `.dvc` pointer files and reports are committed. Run `uv run dvc pull` before anything that touches data.

---

## Datasets

| Dataset                                                                                                                                                    | Size                | Role                                                                                  |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- | ------------------------------------------------------------------------------------- |
| **[Arabic Customer Reviews (`Final_Data.csv`)](https://www.kaggle.com/datasets/mohamedramadan2040/arabic-customer-reviews)**                        | ~40 k rows (4.4 MB) | **Primary** — drives the DVC pipeline, all training, and every reported metric |
| **[330K Arabic Sentiment Reviews (`arabic_sentiment_reviews.csv`)](https://www.kaggle.com/datasets/abdallaellaithy/330k-arabic-sentiment-reviews)** | 330 k rows (212 MB) | Secondary corpus for EDA / pretraining exploration (binary-labeled)                   |

Preprocessing on the primary set (`reports/preprocess_metrics.json`): 40 046 raw rows → **36 045** clean rows (1 939 exact + 2 062 near-duplicates removed, 5 952 near-empty flagged), then a label- **and** dialect-stratified split locked by `split.random_state` in `params.yaml`:

| Split     |   Rows | positive | negative | neutral |
| --------- | -----: | -------: | -------: | ------: |
| `train` | 25 231 |   14 533 |    9 419 |   1 279 |
| `val`   |  3 605 |    2 076 |    1 346 |     183 |
| `test`  |  7 209 |    4 152 |    2 691 |     366 |

---

## Installation & Usage

### Prerequisites

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** package manager
- **Docker** (for the serving stack)
- **Git**

### 1. Clone the Repository

```bash
git clone git@github.com:<your-org>/raay.git
cd raay
```

### 2. Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

### 3. Install Dependencies

```bash
uv sync --all-extras
```

### 4. Set Up Environment Variables

```bash
cp .env.example .env
# Edit .env and set:
#   MLFLOW_TRACKING_URI=sqlite:///mlflow.db   # local default
#   MLFLOW_ALLOW_FILE_STORE=true              # required by MLflow >= 3 for file: URIs
```

MLflow runs against a **local SQLite store** (`mlflow.db`) by default — no server needed:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://localhost:5000
```

> To use a remote tracking server instead, set `MLFLOW_TRACKING_URI=http://<host>:5000`; the code paths are identical.

### 5. Set Up DVC

`dvc init` and the `storage` remote are already committed in this repo — only the data pull is needed:

```bash
uv run dvc pull
```

<details>
<summary>First-time setup only (fresh clone without <code>.dvc/</code>)</summary>

```bash
uv run dvc init
mkdir -p ~/dvc-storage/raay
uv run dvc remote add -d storage ~/dvc-storage/raay
uv run dvc pull
```

</details>

### 6. Install Pre-commit Hooks

```bash
uv run pre-commit install
uv run pre-commit install --hook-type pre-push
```

### 7. Run Quality Checks

```bash
uv run ruff check .          # Lint
uv run ruff format --check . # Format check
uv run mypy src              # Type check
uv run pytest                # Tests (704)
```

---

## Data Pipeline

A DVC pipeline (`dvc.yaml`) turns the raw dataset into reproducible, dialect-tagged splits. Stages read `params.yaml` and auto-log to the `raay_preprocessing` MLflow experiment.

```bash
uv run dvc repro             # preprocess → split
```

| Stage          | Command                            | Outputs                                                                         |
| -------------- | ---------------------------------- | ------------------------------------------------------------------------------- |
| `preprocess` | `python -m raay.data.preprocess` | `data/interim/normalized.csv`, `reports/preprocess_metrics.json`            |
| `split`      | `python -m raay.data.split`      | `data/processed/{train,val,test}.csv` (+ `dialect`, `dialect_confidence`) |

---

## Phase 3: Modeling Baseline

AraBERT v2 (`aubmindlab/bert-base-arabertv02`) fine-tuned through a **6-run Kaggle GPU sweep** (varying `learning_rate` / `batch_size`); the best run was registered as `ArabicSentiment → Production`.

```bash
# Hydra-driven fine-tune (GPU: Kaggle T4)
uv run python -m raay.training.train learning_rate=2e-5 batch_size=32

# The sweep driver (registers the winner)
uv run python scripts/kaggle_train_runs.py --n 6

# Evaluate on the held-out test set
uv run python -m raay.training.evaluate \
  --model-dir models/baseline/final \
  --test-file data/processed/test.csv
```

### Results (`reports/eval_baseline.json`)

| Metric        | Value             |
| ------------- | ----------------- |
| Accuracy      | **84.92 %** |
| F1 (macro)    | **0.6407**  |
| F1 (weighted) | 0.8414            |

**Per-class**

| Class    | Precision | Recall | F1              | Support |
| -------- | --------- | ------ | --------------- | ------: |
| Positive | 0.880     | 0.909  | **0.895** |   4 152 |
| Negative | 0.846     | 0.853  | **0.850** |   2 691 |
| Neutral  | 0.245     | 0.139  | **0.178** |     366 |

> **Neutral underperforms** because it is only ~5 % of the test set — the dominant remaining error mode.

**Dialect breakdown**

| Dialect   |     n | Accuracy | F1 macro |
| --------- | ----: | -------: | -------: |
| MSA       | 3 322 |   85.7 % |    0.605 |
| Gulf      | 1 145 |   86.2 % |    0.682 |
| Egyptian  | 1 011 |   82.4 % |    0.647 |
| Levantine | 1 129 |   82.4 % |    0.631 |
| Maghrebi  |   357 |   92.2 % |    0.668 |
| Arabizi   |   245 |   80.0 % |    0.538 |

---

## Phase 4: Compression & Optimization

Cut inference cost without losing quality: **distillation → ONNX export → INT8 quantization**, with every variant benchmarked on accuracy *and* latency and tracked in MLflow.

### 4.1 Knowledge Distillation

A 6-layer student distilled from the fine-tuned teacher on Kaggle GPU:

```bash
uv run python scripts/kaggle_train_runs.py --module distill --n 6
uv run python -m raay.training.distill alpha=0.4 temperature=4.0   # Hydra overrides
```

| Metric     | Teacher (baseline) | Student (distilled) |
| ---------- | -----------------: | ------------------: |
| Accuracy   |            84.92 % |             83.04 % |
| F1 (macro) |             0.6407 |              0.5975 |
| Size       |           542.6 MB |            372.5 MB |

### 4.2 ONNX Export & INT8 Quantization

```bash
# Export teacher + student (dynamic batch/seq), validate PyTorch-vs-ORT parity
uv run python -m raay.inference.export_onnx
#   → models/onnx/model.onnx + models/onnx/distilled.onnx (+ external .onnx.data weights)
#   → reports/onnx_parity.json, logs to raay_training

# Dynamic INT8 quantization of the teacher graph (self-contained)
uv run python -m raay.inference.quantize_onnx
#   → models/onnx/model_int8.onnx
#   → reports/onnx_int8_parity.json + reports/eval_int8.json
```

Parity: FP32 graphs match PyTorch logits (`max_abs_diff ≈ 3.5e-06`, argmax identical); INT8 keeps **100 % label agreement** with a mean absolute logit deviation of `0.10`.

### 4.3 The 4-Variant Benchmark

```bash
uv run python scripts/benchmark.py            # regenerates reports/benchmark_table.{md,csv}
```

| Variant                 | Backend                |          Accuracy | F1 (macro) |               Size |      batch-1 p50 |       batch-1 p95 |
| ----------------------- | ---------------------- | ----------------: | ---------: | -----------------: | ---------------: | ----------------: |
| `baseline-torch`      | PyTorch CPU FP32       |           84.92 % |     0.6407 |           542.6 MB |          31.2 ms |           69.4 ms |
| `distilled-torch`     | PyTorch CPU FP32       |           83.04 % |     0.5975 |           372.5 MB |          22.2 ms |           45.8 ms |
| `onnx-fp32`           | ORT CPU FP32           |           84.92 % |     0.6407 |           540.9 MB |          19.0 ms |           43.0 ms |
| **`onnx-int8`** | **ORT CPU INT8** | **85.03 %** |     0.6369 | **136.1 MB** | **9.5 ms** | **27.6 ms** |

INT8 is **4× smaller** (541 MB → 136 MB) and **2× faster at p50** than the FP32 graph (19.0 → 9.5 ms; 1.6× at p95), while *gaining* 0.11 pp accuracy — quantization noise, not a real improvement. `onnx-fp32` accuracy is inherited from `baseline-torch` — identical weights.

### 4.4 MLflow Model Registry

```bash
uv run python scripts/log_variants_mlflow.py       # --winner onnx-int8
```

Each variant is logged as its own `raay_training` run (`stage=baseline|distilled|fp32|int8`) with accuracy, F1, latency, and size metrics, then the winner is registered and promoted.

| Version | Variant                           | Stage      | Alias                      |
| ------- | --------------------------------- | ---------- | -------------------------- |
| `v5`  | `distilled-fp32` (canary graph) | Production | `Production`, `Canary` |
| `v4`  | `onnx-int8`                     | Archived   | —                         |
| `v1`  | baseline transformers             | Archived   | —                         |

> Both the production INT8 graph and the canary distilled-FP32 graph live under the **same** registered model (`ArabicSentiment`); the alias is the promotion switch, not the graph source. The BentoML service resolves `models:/ArabicSentiment/Production` at worker start; `RAAY_ONNX_PATH` overrides it.

---

## Phase 5: Serving & Operations

### 5.1 The API

```bash
uv run bentoml serve src/raay/serving/serve.py:svc --reload    # dev server on :3000
```

```bash
curl -s http://localhost:3000/predict \
  -H 'Content-Type: application/json' \
  -d '{"texts": ["هذا المنتج ممتاز والجودة عالية جدا", "المنتج وصل متأخر والجودة رديئة"]}'
```

```json
{
  "predictions": [
    {"label": "positive", "score": 0.9914},
    {"label": "negative", "score": 0.9899}
  ]
}
```

`POST /predict` takes `{"texts": [...]}`; an empty list short-circuits to an empty 200 response, malformed payloads get 422 from pydantic. `GET /health` returns `{"status": "healthy"}`.

### 5.2 Latency & Load Testing

```bash
# In-process latency percentiles → reports/serving_benchmark.json
uv run python -m raay.serving.benchmark --samples 500

# HTTP before/after through the same service → reports/locust_{fp32,int8}.html
uv run python scripts/locust_run.py --users 8 --run-time 60
```

**Single-call INT8 latency** (500 samples, 2 runs): **p50 12.0 ms · p95 35.9 ms · p99 65.4 ms** — inside the product SLA, so CPU INT8 shipped.

**Locust** (8 users, 60 s, 3-review calls, 0 failures, `reports/locust_*_stats.csv`):

| Variant             | Requests |          Median |              p95 |              p99 | Failures |
| ------------------- | -------: | --------------: | ---------------: | ---------------: | -------: |
| FP32 ONNX           |      347 |           78 ms |           160 ms |           270 ms |        0 |
| **INT8 ONNX** |      364 | **54 ms** | **120 ms** | **160 ms** |        0 |

> **TensorRT** is retained as a future accelerator path. An engine must be built on the fixed production GPU / CUDA / TensorRT environment — never ahead of it.

### 5.3 Containerized Serving

```bash
uv run bentoml build -f bentofile.yaml
uv run bentoml containerize raay-sentiment:latest --image-tag raay-sentiment:latest
docker compose up -d           # prod :8000 · redis :6379
```

- The bento bakes `models/onnx/model_int8.onnx` + the baseline tokenizer, so the container is self-contained
- `docker-compose.yml` sets `RAAY_ONNX_PATH` / `RAAY_TOKENIZER_DIR` to the baked artifacts and probes `/health` with a Python `urllib` healthcheck
- Do **not** bind `env_file: .env` in compose — it holds dead Kaggle paths

### 5.4 Near-Real-Time Micro-Batch Consumer

Client-side micro-batching over Redis (there is no `batchable=True` in `serve.py` — batching happens on the client):

```bash
uv run python -m raay.inference.batch_consumer --mode producer --samples 10
uv run python -m raay.inference.batch_consumer --mode consumer

# In-memory benchmark (default) → reports/queue_benchmark.json
uv run python -m raay.inference.batch_consumer --mode benchmark \
  --samples 128 --max-batch 32 --overhead-ms 20
```

- Polls the `reviews` list, drains up to `--max-batch 32` (default) or `--max-wait-ms 100`, whichever fires first, and scores the whole micro-batch in **one** in-process INT8 ORT call
- Publishes `{"id"?, "text", "label", "score"}` to `reviews-results`
- 20 hermetic tests (`tests/test_batch_consumer.py`)

### 5.5 Nightly Batch Re-Scoring & Drift

```bash
uv run python -m raay.inference.batch_score --mode init-reference --samples 1200
uv run python -m raay.inference.batch_score --mode make-input --date 2026-09-24
uv run python -m raay.inference.batch_score --mode score     --date 2026-09-24
uv run python -m raay.inference.batch_score --mode drift    --date 2026-09-24
```

| Mode               | Reads                          | Writes                                                                                         |
| ------------------ | ------------------------------ | ---------------------------------------------------------------------------------------------- |
| `init-reference` | `data/processed/test.csv`    | `data/scoring/reference/reference.csv` (fixed seed)                                          |
| `make-input`     | `data/processed/test.csv`    | `data/scoring/input/{date}.csv` (seed = CRC32 of the date → idempotent)                     |
| `score`          | the day's input                | `data/scoring/output/{date}.csv` (3-class probs + `predicted_label` / `predicted_score`) |
| `drift`          | reference vs. the day's output | `reports/drift/{date}.json`                                                                  |

The drift gate is **Evidently PSI**: `< 0.1` PASS, `< 0.2` WARN, `≥ 0.2` FAIL (overall = worst column). Every mode logs to the `raay_batch` MLflow experiment. `--no-engineer` falls back to the two original output columns, so the job still runs on a runner that has no encoder weights.

> **Do not shrink `--samples` for `init-reference`.** The reference size sets the PSI noise floor: at 200 rows the gate returned **FAIL** with the tail PCs at 0.20/0.32/0.22, because pc8–pc10 carry only ~1.3% of the variance combined (pc1 alone is 0.72) and two small samples disagree on their bin proportions. At the production 1000 the same three read 0.024/0.042/0.037.

Latest verdicts: `2026-10-01` **PASS** (max PSI 0.0416) · `2026-10-02` **PASS** (max 0.0546, 17 columns gated, `oov_rate` SKIPPED — see 5.6).

### 5.6 Engineered Drift Features (Step 6)

The gate above only watched two model outputs. `src/raay/inference/drift_features.py` widens it to **17 gated columns**: `predicted_label`, `positive`, `confidence_score`, `text_length`, `oov_bucket`, `oov_rate`, `dialect_label`, and `embedding_pc1..10`.

`AraBertEmbedder` loads the **same** tokenizer and encoder the model was fine-tuned with (`models/baseline/final`, `local_files_only`, mask-aware mean pooling) and embeds the day's texts. PCA is **fitted once on the reference and frozen** to `data/scoring/reference/pca_basis.joblib` — never refit per day, so day-over-day scores are comparable. `ProjectionBasis.check_compatible` refuses a basis built from a different `model_dir`/`max_length`/`pooling`.

**The encoder is loaded for exactly two modes**, `init-reference` and `drift`. `make-input` and `score` must not pay ~2 GB and ~2 minutes for a model they never use, and the nightly DAG runs them every day.

The gate was validated with both a null and a positive control, and the numbers are the reason to trust it:

| control | result |
| --- | --- |
| **Null** — two independent 1000-row panels vs one 1000-row reference | max PSI **0.042 / 0.055** — the noise floor sits ~2× below the 0.1 WARN line |
| **Positive** — half the panel replaced with francophone-Arabizi | **all 16 non-`oov_bucket` columns FAIL**; dialect PSI 1.47, arabizi share 4.5%→51.5%, `text_length` 1.84 |

It fires when it should and stays quiet when it should. 51 hermetic tests in `tests/test_drift_features.py` (fake tokenizer/encoder/scorer in `tests/conftest.py`).

### 5.7 Prediction Drift & Triage (Step 6)

`src/raay/inference/prediction_drift.py`, run as `--mode predict-drift` → `reports/prediction_drift/{date}.json`. This watches the **outputs** rather than the inputs: the predicted class mix PSI'd against **two** references (the training label prior, read from `data/processed/train.csv`; and the reference panel's own predictions), mean `predicted_score` vs the frozen reference mean, plus a rolling z-score.

`classify_triage` is a 2×2 that names *which half moved*, which is the difference between "something is wrong" and "go look at this":

| input | output | triage |
| --- | --- | --- |
| PASS | PASS | `stable` |
| FAIL | PASS | `world_changed` |
| PASS | FAIL | `model_degraded` |
| FAIL | FAIL | `ambiguous` |
| missing | any | `indeterminate` |

A missing input-drift report returns `indeterminate` rather than assuming the inputs held. `escalate` — the coupled signal (confidence falling **AND** class PSI rising) — is kept as its own field and never folded into `overall`, so it cannot invent a threshold below the agreed one.

> **The brief's 45/35/20 class prior is wrong for this dataset, and the gate reads the real one instead.** Measured proportions are **57.6/37.3/5.1**, identical across all three splits (Neutral overstated 4×). A clean panel against 45/35/20 scores **PSI 0.41 (FAIL)**; against the real prior, **0.021 (PASS)** — the gate would have failed every night on a healthy model. Both numbers are pinned by tests so the wrong prior cannot be quietly reintroduced.

> **The ~0.02 baseline is the model's Neutral weakness, not drift.** It predicts Neutral 2.9% against a 5.1% prior (recall 0.139), so PSI against the true prior is never 0. Do not tighten thresholds below it. Separately, the rolling z-score needs **≥ 7 days** of history, not 3: at a 3-day window four panels that are provably just sampling noise produce a monotone decline and fire at z = −2.85. 31 hermetic tests in `tests/test_prediction_drift.py`.

### 5.8 Airflow Orchestration

Airflow is **host-isolated — not a project dependency**:

```bash
uv tool install "apache-airflow==2.10.5"
export PATH="$HOME/.local/bin:$PATH"
export AIRFLOW_HOME=/home/diab/Documents/Raay/airflow_runtime   # git-ignored runtime
airflow db migrate

# Both must keep running for the scheduler to fire
nohup airflow scheduler  >> airflow_runtime/logs/scheduler.out  2>&1 &
nohup airflow webserver --port 8080 >> airflow_runtime/logs/webserver.out 2>&1 &

airflow dags trigger raay_nightly_batch_scoring
```

The DAG `airflow/dags/raay_nightly_batch_scoring.py` runs daily at **03:00 UTC** as six chained tasks, each a `BashOperator` shelling out to `uv run python -m raay.inference.*`:

```
materialize_daily_input → score_daily_batch → run_drift_check
                        → run_prediction_drift_check → evaluate_retrain_trigger
                        → check_pending_feedback → merge_validated_feedback
```

The chain is ordered deliberately: `predict_drift` must run **after** `drift` or every night classifies as `indeterminate`, and the feedback merge must not run while the drift chain has failed — it would half-apply a batch on top of an incomplete night. `check_pending_feedback` is a `ShortCircuitOperator` reading the merged *file*, not a counter, so a retry after a crash cannot re-merge. 11 structural tests in `tests/test_nightly_dag.py`.

> `LocalExecutor` is hard-blocked on SQLite — the runtime uses `SequentialExecutor`. Only change the executor while the scheduler is **stopped**, and trust the scheduler's own `/proc/<pid>/environ` over `airflow config get-value`.

### 5.9 Shadow-then-Canary Rollout

A standalone compose project (`docker-compose.canary.yml`, `name: canary`) that
owns **:8000 through its own nginx front** while a rollout runs, then hands it
back. Two workers sit behind nginx, and `scripts/canary_promote.py` moves the
traffic split stage by stage, gating each stage on the compose-local
Prometheus that scrapes a canary-agent sidecar:

| Worker                    | Graph             | Registry alias        | Host port                                   |
| ------------------------- | ----------------- | --------------------- | ------------------------------------------- |
| `raay-sentiment`        | INT8 (production) | `Production` (alias)  | `8002` (direct health; nginx owns `8000`) |
| `raay-sentiment-canary` | distilled FP32    | `Canary`              | `8001` (direct health)                      |
| `raay-canary-agent`     | —                 | —                     | `9100` (`/ingest`, `/metrics`, `/health`)   |
| `prometheus`            | —                 | —                     | `9090`                                      |

Stage progression — each `advance` renders `deploy/nginx_canary.conf` and
reloads nginx, but **only after** the previous stage's gates pass:

| Stage      | Split        | Mirror? | Gate (over the trailing `--window`)                                     |
| ---------- | ------------ | ------- | ----------------------------------------------------------------------- |
| `shadow`   | 100 / 0      | yes     | label agreement ≥ 0.99, error rate ≤ 0.005, p95 ≤ 1.10× stable            |
| `canary-5` | 95 / 5       | no      | error rate + p95 ratio (`agreement` needs the mirror, so it stops here)   |
| `canary-25`| 75 / 25      | no      | error rate + p95 ratio                                                    |
| `canary-50`| 50 / 50      | no      | error rate + p95 ratio                                                    |
| `full`     | 0 / 100      | no      | online 50/50 gate **and** the Phase 6 offline gate (only code that    |
|            |              |         | moves the `Production` alias)                                             |

```bash
# 1. declare the candidate graph (idempotent) and start the project
uv run python scripts/canary_promote.py --mode declare
RAAY_IMAGE=ghcr.io/ahmeddiab1234/arabic-sentiment:<sha> \
  docker compose -f docker-compose.canary.yml up -d --no-build

# 2. enter shadow -- every /predict is mirrored to the candidate, 100/0
uv run python scripts/canary_promote.py --mode shadow

# 3. widen the slice only on a green gate (FAIL = exit 1, INCONCLUSIVE = 2)
uv run python scripts/canary_promote.py --mode advance --to canary-5
uv run python scripts/canary_promote.py --mode advance --to canary-25
uv run python scripts/canary_promote.py --mode advance --to canary-50
uv run python scripts/canary_promote.py --mode advance --to full   # runs the Phase 6 gate

# 4. rollback is one command: stable-only weights, then the pre-rollout alias
uv run python scripts/canary_promote.py --mode rollback
docker compose -f docker-compose.canary.yml down
```

Both variants live under the **same** registered model (`ArabicSentiment`); the
canary worker binds `distilled.onnx` + `.onnx.data` + `models/distilled/final`
read-only, so the registry alias is the promotion switch rather than the graph
source. Rollout state lives in `reports/canary_state.json` (git-ignored) —
`rollback` reads it, so a promotion and its rollback are one atomic story.

A note on **shadow pairing**: the nginx `mirror` subrequest is a separate
request object, so its `$request_id` differs from the main request's and
`proxy_set_header` on the main location is invisible to it (nginx clones only
client-sent headers). Correlation therefore happens in the agent on the
request content, which the mirror *does* clone: it pairs stable/candidate
events by `X-Request-ID` when a client supplies one, otherwise by the SHA-256
of the request `texts`. 46 hermetic tests in `tests/test_canary_nginx.py`
(renderer per stage, fake MLflow client, fake Prometheus — no nginx binary
needed).

### 5.10 Scheduled Retraining Hooks (Step 6)

`.github/workflows/retrain.yml` watches for **newly versioned raw data** and
turns it into a reviewable model-update PR. Weekly (Monday 04:12 UTC), on
demand, or on a `repository_dispatch` (`event_type: retrain` — the seam that
`raay.inference.retrain_trigger` pushes through nightly; see 5.11). It:

1. **Checks out `dev`**, because that is where data changes land. Note the
   scheduling caveat below: GitHub only honours `schedule:` and
   `repository_dispatch` for workflow files present on the **default branch**;
2. **Pulls the raw CSV and re-runs** `dvc pull data/raw/Final_Data.csv.dvc`
   → `dvc repro preprocess split` → verifies `dvc status preprocess split` is
   clean. Scoped to those two stages on purpose: a bare `dvc repro` runs every
   stage in `dvc.yaml` — including `feedback`, whose dep this runner never pulls
   — and would rewrite the git-tracked `reports/feedback_metrics.json`, tripping
   `git diff --exit-code dvc.lock` under a metrics table that explains nothing;
3. **Early-exits on no-op**: if `dvc repro` left `dvc.lock` unchanged (the lock
   embeds the raw input md5), nothing moved — the run stops, costing nothing;
4. **Gates the split** with the *same* `scripts/ci_metrics_gate.py` CI uses, at
   ±0.005, with one documented escape hatch: `--ignore '.*_size$'` removes the
   integer sizes/counts from the diff. A refresh is exactly the case where those
   SHOULD move; label/dialect **proportions** are still compared strictly, and a
   brand-new label/dialect class (a key with no baseline) fails the gate on
   purpose;
5. **On PASS** pushes branch `retrain/data-<date>` (`dvc.lock` + both metrics
   jsons) and opens a PR into `dev` with the gate table as its body — noting
   that CI's *own* gate will show the count rows red on that PR by design;
6. **On FAIL** prints `::error::` and exits 1 (an alarm, never a silent skip),
   uploading the report either way (→ `reports/data_refresh_<date>.{json,md}`,
   git-ignored).

A dispatch also runs an independent `notify` job (`if: repository_dispatch`,
`needs: refresh`, `issues: write`) that opens an idempotent
`Retrain candidate: <reason> on <date>` issue. It deliberately cannot retrain —
a structural test pins that it invokes no `uv` / `dvc` / `python -m` — because
a `psi_breach` means the raw data did not change, so the refresh job is correctly
a no-op and forcing past its early exit would reach `git commit` with nothing
staged.

The refresh threshold defaults to `0.005` and is dispatch-overridable via
`client_payload.threshold`; a dispatch `client_payload.reason` is echoed in the
run summary. 24 structural tests in `tests/test_retrain_workflow.py`; the
`--ignore` change to `ci_metrics_gate.py` leaves the default path byte-identical,
so CI's own gate is untouched (35 tests).

> **This workflow is currently inert, and so is every other cron in the repo.** `main` contains **no `.github/` directory at all** — it sits on a divergent lineage whose merge-base with `dev` is the very first commit, carrying none of `dev`'s 60+ commits. GitHub only runs `schedule:` and `repository_dispatch` for default-branch workflow files, so the weekly retrain, the weekly promotion re-gate, and all of CD have never executed. `push`/`pull_request` are unaffected, which is why CI looks healthy and hides it. Getting the workflow files onto the default branch is the single highest-impact fix outstanding.

### 5.11 Retrain Trigger (Step 6)

`src/raay/inference/retrain_trigger.py` is the nightly DAG's **fifth** task. It reads both drift reports and turns them into a decision, with precedence **`manual` > `psi_breach` > `scheduled` > `none`**, writing `reports/retrain_trigger/{date}.json` and logging `trigger_reason` as a queryable tag in `raay_batch`.

```bash
uv run python -m raay.inference.retrain_trigger --date 2026-10-03
```

Three deliberate **non**-firers, each a false alarm avoided:

1. **WARN does not fire.** Action is at 0.2–0.25, and a model sitting at its ~0.02 Neutral offset must never reach a trigger.
2. **`SKIPPED` / `ERROR` columns do not fire.** `oov_rate` is structurally 0.0 → `SKIPPED`; a null score is not a breach.
3. **Unconfirmed calendar events do not fire.** Ramadan and Eid are moon-sighting-set, so `confirmed: false` in `configs/seasonal_events.yaml` is the arming switch and a stale or guessed date cannot cause a retrain. Black Friday is resolved as *the Friday after the 4th Thursday*, deliberately not "last Friday of November" (2027-11-26 vs 2027-11-27 — a test pins both).

`dispatch_retrain` POSTs to the GitHub `dispatches` API with stdlib `urllib.request` (`requests` is only a transitive dep, and `deploy_staging` / `canary_promote` are stdlib-only for the same reason). The token is read from `airflow_runtime/secrets/github_dispatch_token` or `RAAY_GITHUB_DISPATCH_TOKEN`, **never argv**, and goes in an `Authorization: Bearer` header.

- **The required scope was measured, not assumed.** GitHub's endpoint page documents only the *classic* `repo` scope. A read-only fine-grained PAT returns **403** `Resource not accessible by personal access token`; a fine-grained token with **Contents: read and write** on the target repo returns **204**.
- **A failed dispatch exits 1; a missing token exits 0.** The report is written to disk first either way, and `dispatch.ok` / `status` / `error` land in it. A breach the operator asked to be notified about and wasn't is an infrastructure fault, so the task goes red and Airflow retries.
- **A 204 does not mean a workflow ran.** On this repo none does — see the caveat in 5.10.

`scripts/promote_model.py` stamps the reason onto the promoted version, **only at the Production flip**. A rejected candidate or a `--dry-run` is never tagged, and a clean night (`reason: none`) emits no tags at all, because tagging a version `trigger_reason=none` would assert a drift-motivated promotion that never happened. 30 hermetic tests in `tests/test_retrain_trigger.py`.

### 5.12 Customer-Service Feedback Loop (Step 6)

The last mile: production disagrees with the model, and the disagreement is a
**labelled example nobody had**. A CS agent corrects a sentiment label in their
tool; those corrections are the only real-traffic supervision this project has.

```
POST /feedback  ──►  data/feedback/raw/{date}.csv        (append-only, never edited)
                        │
                        │  --mode review        operator + adjudication happen here
                        ▼
                  data/feedback/reviewed/overrides.csv   (git-tracked, DVC-hashed, human-owned)
                        │
                        │  --mode merge         DVC stage, the only path into training data
                        ▼
                  data/processed/train_feedback.csv      (train-only sidecar)
                        │
                        ▼
                  train.py concatenates onto train.csv   (val and test untouched)
```

**Capture** — `src/raay/serving/feedback_service.py`, a standalone Starlette
sidecar. `POST /feedback` with a bearer token (`RAAY_FEEDBACK_TOKEN` or
`RAAY_FEEDBACK_TOKEN_FILE`), plus `/health` and `/stats`. Append-only CSV, one
file per UTC day, idempotent on a content hash of
`(text, model_label, corrected_label, agent_id)` — so a retrying client or a CS
tool that double-clicks produces one row, not two. The service **refuses to
start** without a token unless `RAAY_FEEDBACK_ALLOW_ANON=1`; an unauthenticated
write endpoint into the training set is a poisoning vector.

**QA ladder** — `--mode review`. An override is a *proposal*, not ground truth:
a support agent's judgement is formed under time pressure during a dispute, and
disputes are frequently about shipping rather than sentiment.

| Status | Meaning | Trains? |
| --- | --- | --- |
| `corroborated` | 2 **distinct agents** assert the same label for the same review | yes |
| `adjudicated` | a senior adjudicator ruled (the ruling wins over the agents) | yes |
| `single_agent` | only one assertion so far | no |
| `disputed` | ≥2 agents, no label reaches the bar | no |
| `confirmation` | `model_label == corrected_label` | **never** |
| `leak` | fuzzy-matches `data/processed/test.csv` | no |
| `duplicate` | already merged by an earlier run | no |
| `near_empty` | below `min_char_length` | no |

Three deliberate choices worth knowing:

- **Corroboration counts agents, not rows.** An agent who double-posts cannot
  corroborate themselves; an agent asserting two labels for one review is dropped
  as self-contradictory rather than counted as a vote.
- **One review yields one training row.** The second assertion is *evidence for
  the label* (`corroborated_by`), not a second copy — two identical
  `(text, label)` pairs would weight a single hard negative 2× for no
  informational reason.
- **Confirmations are never training data.** A confirmation's "corrected" label
  *is* the model's own prediction. It is kept instead as the **denominator** of
  `production_error_rate` in `reports/feedback_metrics.json` — the only
  production-measured error rate in the project, and the only Phase 6 signal not
  drawn from `test.csv`. It is uninterpretable if the CS tool posts only
  disputes, and the report says so in its own `note`.

**Routing, not gating.** A `model_score ≥ 0.9` sends a row to
`route=adjudicate_first`. A model that was confident *and wrong* is the hardest
case worth collecting, so confidence must never be able to discard a row. Neutral
overrides are capped at 50/batch because CS disputes skew Neutral and Neutral is
already the weakest class (recall 0.139).

**Train-only by design.** Merging into `data/interim/normalized.csv` would
re-run `split`, change `test.csv`, and make `promote_model.py:check_frozen_split`
exit 2 with no report — collapsing the 14-gate promotion story. So the merged
rows get their own file that `load_data` concatenates onto **train only**: `val`
drives `load_best_model_at_end` so overrides must not touch it, and the frozen
test split never moves.

**The two modes read different things, deliberately.** `--mode merge` is a DVC
stage whose only dependency is `reviewed/overrides.csv`; it must not read
`raw/`, because doing so would (a) make it unreproducible from its own declared
deps and (b) **rewrite the reviewed file, destroying every `adjudicated_label` a
person typed in**. `review` writes it, a human edits it, DVC hashes it, `merge`
consumes it. `merge` is idempotent — it appends to the existing sidecar and
drops texts already present, so a retry after a crash cannot double-count.

Nightly, the sixth Airflow task (`check_pending_feedback` →
`merge_validated_feedback`) chains off the end of the drift chain and
short-circuits when there is nothing new, so a no-op night does not churn
`dvc.lock`.

```bash
uv run python -m raay.data.feedback --mode review    # raw -> reviewed/overrides.csv (operator, then dvc add)
uv run dvc repro feedback                            # reviewed -> train_feedback.csv (the DVC stage)
uv run python -m raay.data.feedback --mode merge     # same command, without DVC
```

`ci.yml` and `retrain.yml` scope `dvc repro` to `preprocess split`, so a bare
`dvc repro` cannot sweep the feedback stage into a PR whose metrics gate would
then explain nothing. `dvc.lock` still covers its hashes, so a non-reproducible
merge surfaces as a lock diff. 67 tests in `tests/test_feedback.py`, 32 in
`tests/test_feedback_service.py`.

> The full drift-and-feedback story — every measured number, the null/positive
> controls, and the eight bugs the tests caught — is written up in
> [`docs/pr7.md`](docs/pr7.md), with the verbatim execution logs in
> [`commands_run_p7.txt`](docs/commands_run_p7.txt) and
> [`commands_run_p8.txt`](docs/commands_run_p8.txt).

---

## Phase 6: CI/CD, Deployment &amp; Automation

The pipeline is four workflow files with one shared quality gate and one mover of truth.

| Piece | File | Trigger | What it does |
| --- | --- | --- | --- |
| **CI** | `.github/workflows/ci.yml` | every PR / push to `dev` | lint + test, then `dvc repro` and the ±0.005 split-metrics gate as a PR comment |
| **CD** | `.github/workflows/cd.yml` | every push to `main` | builds the serving image, versioned `int8-<md5>`, smoke-tests it, pushes to GHCR |
| **Staging deploy** | `cd.yml` → `deploy-staging` job | after a published image | deploys to the staging box, gates it, auto-rolls back to the last smoke-tested sha |
| **Promotion gate** | `.github/workflows/promote.yml` | dispatch / weekly | 14-gate evaluation; the only code that moves the `Production` alias |
| **Retrain check** | `.github/workflows/retrain.yml` | weekly / dispatch | re-runs `dvc repro` on new raw data; opens a data-refresh PR (see 5.10) |

### 6.1 CI — gate every change to `dev`

`lint` (ruff check, ruff format --check, mypy) → `test` (pytest with coverage) → `pipeline` (`needs: [lint, test]`): `dvc pull data/raw/Final_Data.csv.dvc` → `dvc repro` → `dvc status preprocess split` must be clean (scoped to the pipeline stages — the job never pulls the DVC-tracked serving artifacts, and lock reproducibility is enforced by the `git diff --exit-code dvc.lock` step at the end) → `scripts/ci_metrics_gate.py --threshold 0.005` diffs `reports/split_metrics.json` against the merge-base and posts the table as a PR comment → `git diff --exit-code dvc.lock`. At ±0.005 the integer counts must match exactly; proportions may move half a point. Fork PRs run lint+test but skip the secret-dependent DVC steps.

### 6.2 CD — publish an image only from `main`

`main` **is not** a fast-forward of `dev` — it sits on a divergent lineage sharing only the very first commit, so CD has never actually run and the assumption that CI already gated these commits does not hold today. The image bakes the four DVC-served artifacts (the int8 graph + the baseline tokenizer, never the MLflow registry — see `docs/pr6.md` for why). `model_version` is derived from the graph's md5 (`int8-687d587004c6`) and ships twice: as an OCI label for `docker inspect`, and as a build-arg that the service reports on `/health` and `/predict`. A smoke test greps both routes for the version before anything is pushed.

### 6.3 Staging — deploy, verify, or roll back

`scripts/deploy_staging.py` (run over SSH by the CD job) pulls the image, waits for health, cross-checks the running container's revision label against the target sha, runs three Arabic predictions against fixed confidence floors, and asserts `{"texts":[1]}` returns 422. Only then is `last_known_good` promoted; any failure rolls back to the previous smoke-tested sha (never `latest`). Local rehearsal without a VM: `scripts/deploy_staging.py --registry localhost:5050 --repository <repo> --skip-login` against a `registry:2`.

### 6.4 Promotion — the only mover of `Production`

`scripts/promote_model.py` runs 14 gates (frozen-split hash, full-split evaluation, label order, F1/accuracy/recall vs Production and vs `eval_baseline.json`, absolute Neutral/Negative recall floors, interleaved null-controlled latency A/B, size, ONNX parity) with a calibrated `floor_tolerance=0.01`, then moves the `Production` alias only on a clean sweep. Locally:

```bash
uv run python scripts/promote_model.py --candidate-version 7 --skip-registry
```

`promote.yml` splits it into a measuring `gate` job and a human-approved `promote` job that re-runs the gate at flip time. Exit codes: `0` passed, `1` a gate failed, `2` the gate could not run at all (and produces no report, deliberately).

---

## Testing

**704 hermetic unit tests** (703 passing, 1 skipped) — no GPU, no network, no running servers:

| Suite                             | Tests | Covers                                                                    |
| --------------------------------- | ----: | ------------------------------------------------------------------------- |
| `tests/test_promote_model.py`     |    74 | Promotion gate (14 gates, latency A/B, dry-run safety)                     |
| `tests/test_feedback.py`          |    67 | Feedback QA ladder, corroboration, leakage guard, train-only merge         |
| `tests/test_drift_features.py`    |    51 | Engineered drift columns, frozen PCA basis, OOV comparability               |
| `tests/test_cd_workflow.py`       |    48 | CD workflow contract (build → inspect → smoke test → push)                  |
| `tests/test_canary_nginx.py`      |    46 | Conf renderer per stage, compose, rollout semantics, gate math              |
| `tests/test_batch_score.py`       |    37 | Input sampling, scoring, PSI drift verdicts, prediction drift               |
| `tests/test_retrain_trigger.py`   |    37 | Trigger precedence, seasonal calendar, `repository_dispatch` POST            |
| `tests/test_serving.py`           |    36 | BentoML service, health middleware, 422 validation, telemetry events        |
| `tests/test_deploy_staging.py`    |    36 | Staging deploy tool (flip/schema/unhealthy rollback paths)                  |
| `tests/test_ci_metrics_gate.py`   |    35 | Split-metrics diff gate (incl. `--ignore` refresh mode)                     |
| `tests/test_feedback_service.py`  |    32 | Starlette feedback sidecar (auth, idempotency, 422s)                        |
| `tests/test_prediction_drift.py`  |    31 | Class-mix gates, triage matrix, rolling z-score                             |
| `tests/test_promote_workflow.py`  |    29 | Promotion workflow contract (gate → human → re-gate)                        |
| `tests/test_ci_workflow.py`       |    27 | CI workflow contract (fork guard, repro scoping, token pinning)              |
| `tests/test_retrain_workflow.py`  |    24 | Retrain workflow contract + the `notify` job                                |
| `tests/test_canary_agent.py`      |    22 | Shadow pairing (request-id + content), gaps, Prometheus text                |
| `tests/test_batch_consumer.py`    |    20 | Micro-batch drain, fake Redis / in-memory queue, fake scorer                |
| `tests/test_nightly_dag.py`       |    11 | Nightly DAG task chain and ordering                                         |
| `tests/test_preprocess.py`        |     9 | Normalization, dedup, near-empty flagging                                   |
| `tests/test_export_onnx.py`       |     9 | Export parity helpers                                                       |
| `tests/test_dialect.py`           |     7 | Dialect heuristics + confidence                                             |
| `tests/test_distill.py`           |     7 | Distillation loss / config wiring                                           |
| `tests/test_split.py`             |     6 | Split sizes + label/dialect proportions                                     |
| `tests/test_quantize_onnx.py`     |     3 | Quantization + parity report                                                |

```bash
uv run pytest
```

---

## Limitations & Honest Findings

Stated plainly, because the numbers above are only useful with their caveats:

- **Micro-batching does not beat serial scoring on this box.** At batch 32 the INT8 graph costs ~58–84 ms *per item* vs ~12 ms single — a `pure_cpu` speedup of **0.25×** (0.50× once a 20 ms per-call overhead is modeled). Batching still wins on *call amortization* (128 reviews → 4 `session.run` calls) and on batch-throughput-bound backends (GPU / TensorRT) or high per-call HTTP overhead. On a 2-core CPU, treat `pure_cpu < 1` as expected, not a bug.
- **The canary latency gate compares like for like on shared cores.** Both workers run on the same box, so their p95s are a same-hardware A/B (fair) *and* both contend for the same cores (the candidate's sawtooth is partly CPU contention). The stage gates therefore require the *ratio* ≤ 1.10, never an absolute budget. The behavioral gate is the nightly drift chain (input drift → prediction drift → trigger).
- **Every cron in this repo is inert today.** `main` contains no `.github/` directory at all — it is on a divergent lineage from `dev`, not a fast-forward of it. GitHub only honours `schedule:` and `repository_dispatch` for workflow files on the **default branch**, so the weekly retrain, the weekly promotion re-gate, and all of CD (`push` → `main`) have never executed. `push`/`pull_request` are unaffected, which is why CI looks healthy and hides it. The `repository_dispatch` POST is implemented and returns 204, but that only means GitHub accepted the event — it does not mean a workflow ran, and here none does. Getting the workflow files onto the default branch is the highest-impact fix outstanding.
- **Both drift panels are seeded draws from `data/processed/test.csv`.** A PASS means "matches the training distribution", and the ~0.05 null floor is therefore optimistic. What is validated is the **mechanism**, not production-traffic behaviour. Two of the four triage cells (`world_changed`, `model_degraded`) are not constructible from one pool at all and are covered only by the parametrized matrix test.
- **`oov_rate` has no detection power on this corpus.** Every reference review — and the injected new-slang panel — has OOV rate exactly 0.0, because AraBERT v2's 64k WordPiece vocab covers this corpus including franco-Arabizi, so nothing reaches `[UNK]`. It reports decision `SKIPPED` with the reason rather than passing silently; `oov_bucket` is the gated form.
- **The seasonal trigger cannot be validated here.** No row in the corpus carries a timestamp, so whether firing before Ramadan helped is unknowable. Ramadan/Eid ship `confirmed: false` precisely so a guessed date cannot cause a retrain.
- **INT8 "gaining" accuracy is noise.** 85.03 % vs 84.92 % is within quantization noise on 7 209 samples; the honest claim is "no meaningful quality loss at 4× compression".
- **Neutral remains the weak class** (F1 0.178) — a class-imbalance problem, not a modeling one. This is also why the prediction-drift gate's ~0.02 baseline against the true prior is expected rather than drift, so its thresholds must not be tightened below it.
- **No local GPU.** Training and distillation run on Kaggle; local work is CPU-only inference.
- **The feedback loop is unexercised by real traffic.** No CS tool is deployed and no agent roster exists, so every number in `reports/feedback_metrics.json` is a rehearsal. Two-agent corroboration needs two real identities: until then `--mode review` legitimately returns 100 % `single_agent` and nothing is trainable. `train_feedback.csv` ships header-only, so no claim is made about its effect on F1. The leakage guard is tested against a synthetic `test.csv`, and rapidfuzz at 0.9 will not catch a semantically identical but lexically rewritten review.
- **`production_error_rate` is uninterpretable if the CS tool posts only disputes.** Confirmations are its denominator, so a dispute-only tool would report a meaningless number — the report says so in its own `note`. An unobserved class gets `rate: null`, never `0.0`.
- **`docs/labeling_guidelines.md` was wrong about the label encoding until v1.1.** It read Positive=2/Neutral=1/Negative=0; every graph uses the inverse. `promote_model.py`'s `label order vs Production` gate exists because of exactly this failure mode. The table is fixed and marked, and the order is now read from `constants.py` rather than retyped. Its §5 balance target (45/35/20) was also wrong for this dataset and was corrected to the measured 57.6/37.3/5.1.

---

## Environment Variables

| Variable                    | Description                                              | Default                                                                |
| --------------------------- | -------------------------------------------------------- | ---------------------------------------------------------------------- |
| `MLFLOW_TRACKING_URI`     | MLflow tracking store                                    | `sqlite:///mlflow.db` in `.env` (code fallback: `file:./mlruns`) |
| `MLFLOW_ALLOW_FILE_STORE` | Required by MLflow ≥ 3 for `file:` URIs                | auto-set when the URI is `file:`                                      |
| `RAAY_ONNX_PATH`          | Explicit ONNX graph override (skips registry resolution) | unset → resolves `models:/ArabicSentiment/Production`                |
| `RAAY_TOKENIZER_DIR`      | Tokenizer + `id2label` source                           | `models/baseline/final`                                              |
| `RAAY_MAX_LENGTH`         | Tokenizer truncation length                              | `128`                                                                |
| `RAAY_MODEL_NAME`         | `ArabertPreprocessor` model                            | `aubmindlab/bert-base-arabertv02`                                    |
| `RAAY_REGISTERED_MODEL`   | MLflow registered model name                             | `ArabicSentiment`                                                    |
| `RAAY_ALIAS`              | Registry alias resolved at worker start                  | `Production`                                                         |
| `DATA_ROOT`               | Kaggle snapshot dir holding the processed splits         | —                                                                     |
| `KAGGLE_TEACHER_DIR`      | Kaggle teacher checkpoint dir (distillation)             | —                                                                     |
| `RAAY_FEEDBACK_TOKEN`     | Bearer token for `POST /feedback` (write access to the training corpus) | unset → the service **refuses to start**                |
| `RAAY_FEEDBACK_TOKEN_FILE` | File holding that token (preferred; not visible in `ps`) | —                                                                     |
| `RAAY_FEEDBACK_ALLOW_ANON` | Permit an unauthenticated feedback service (local dev only) | `0`                                                            |
| `RAAY_GITHUB_DISPATCH_TOKEN` | Fine-grained PAT for `repository_dispatch` — needs **Contents: read and write** on the target repo, not a read-only token | unset → a breach is recorded, nothing is sent, exit 0 |
| `RAAY_GITHUB_DISPATCH_TOKEN_FILE` | File holding that token (preferred; never in argv / `ps`) | `airflow_runtime/secrets/github_dispatch_token` |
| `RAAY_GITHUB_REPOSITORY` | Target `owner/repo` for the dispatch | `ahmeddiab1234/Raay` (the git remote slug) |

---

## License

This project is licensed under the MIT License.
