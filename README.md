<div align="center">

# Raay راي

**Arabic E-Commerce Product Review Sentiment Analysis**

An end-to-end MLOps pipeline for classifying Arabic product reviews (Positive / Negative / Neutral) at scale — supporting Modern Standard Arabic and regional dialects (Egyptian, Gulf, Levantine).

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/package%20manager-uv-blueviolet)](https://docs.astral.sh/uv/)
[![Ruff](https://img.shields.io/badge/linter-ruff-orange)](https://docs.astral.sh/ruff/)
[![MLflow](https://img.shields.io/badge/tracking-MLflow-0194E2)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/data%20versioning-DVC-945DD6)](https://dvc.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#license)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Architecture](#project-architecture)
- [Datasets](#datasets)
- [Installation & Usage](#installation--usage)
- [Data Pipeline](#data-pipeline)
- [Phase 3: Modeling Baseline](#phase-3-modeling-baseline)
- [Phase 4: Compression & Optimization](#phase-4-compression--optimization)
- [Phase 5: Serving & Operations](#phase-5-serving--operations)
- [Testing](#testing)
- [Limitations & Honest Findings](#limitations--honest-findings)
- [Environment Variables](#environment-variables)
- [License](#license)

---

## Overview

**Raay** (Arabic: _رأي_, meaning "opinion") is a production-grade sentiment analysis system designed for Arab e-commerce platforms. It classifies ~50,000 Arabic product reviews per day into three sentiment classes — **Positive**, **Negative**, and **Neutral** — to power product ranking, seller rating aggregation, and customer-service ticket prioritization.

The project covers the full ML lifecycle: data versioning, experiment tracking, model training, inference optimization, and model serving.

**Current state:** an AraBERT teacher fine-tuned to **84.92 %** accuracy, distilled to a 6-layer student, exported to ONNX, dynamically quantized to INT8 (**85.03 %** accuracy, **136 MB**, **p50 ≈ 12 ms** single-call on CPU), served through a containerized BentoML API, re-scored nightly with a PSI drift gate, and rolled out behind a 95/5 canary split.

---

## Features

| Category | Details |
|---|---|
| **3-Class Sentiment** | Positive · Negative · Neutral classification with confidence scores |
| **Arabic NLP** | MSA + dialect support (Egyptian, Gulf, Levantine, Maghrebi, Arabizi/franco-arabe) with per-row dialect tagging |
| **Transformer-based** | AraBERT v2 backbone with HuggingFace Transformers & Accelerate |
| **Model Optimization** | Knowledge distillation (6-layer student), ONNX export, dynamic INT8 quantization |
| **Real-Time Serving** | BentoML REST API (`/predict`, `/health`), containerized via Docker Compose |
| **Near-Real-Time Scoring** | Client-side Redis micro-batch consumer (drain-by-size *or* drain-by-time) |
| **Batch Scoring & Drift** | Nightly re-scoring + Evidently PSI drift gate vs. a fixed reference panel |
| **Orchestration** | Airflow DAG (daily 03:00 UTC) driving input → score → drift |
| **Canary Rollout** | nginx 95/5 weighted split across two workers, health-gated registry promotion |
| **Experiment Tracking** | MLflow runs, metrics, artifacts, and a Model Registry with `Production` / `Canary` aliases |
| **Data Versioning** | DVC-tracked raw/interim/processed data with a configurable remote (local / S3) |
| **Code Quality** | Ruff linter & formatter, mypy static type checking, pre-commit hooks (pre-commit + pre-push) |
| **Testing** | pytest suite (96 unit tests, hermetic — no GPU, no network, no servers) |
| **Configuration** | Hydra for training/distillation; `params.yaml` for the DVC data stages |
| **Typed Schemas** | Pydantic I/O models for the serving contract |
| **Structured Logging** | Loguru for structured, leveled logging |

---

## Tech Stack

| Layer | Tool |
|---|---|
| Language | Python 3.12+ |
| Package Manager | [uv](https://docs.astral.sh/uv/) |
| Deep Learning | PyTorch, HuggingFace Transformers, Accelerate |
| Arabic Preprocessing | `arabert` (`ArabertPreprocessor`) |
| Experiment Tracking | MLflow (local SQLite store, `mlflow.db`) |
| Data Versioning | DVC (local / S3 remote) |
| Inference Runtime | ONNX Runtime (FP32 + dynamic INT8) |
| Model Serving | BentoML + Docker Compose |
| Load Testing | Locust (HTTP before/after) |
| Streaming / Queue | Redis (`redis:7-alpine`) |
| Drift Monitoring | Evidently (PSI) |
| Orchestration | Apache Airflow 2.10.5 (host-isolated) |
| Traffic Shifting | nginx (weighted upstream) |
| Config Management | Hydra (`configs/`) + `params.yaml` |
| Data Validation | Pydantic |
| Linting & Formatting | Ruff |
| Type Checking | mypy |
| Testing | pytest |
| Logging | Loguru |

---

## Project Architecture

```
raay/
├── src/raay/                       # Main Python package
│   ├── data/                       #   Preprocessing, splitting, dialect detection
│   ├── training/                   #   Fine-tuning, distillation, evaluation
│   ├── inference/                  #   ONNX export/quantize, queue consumer, batch scoring
│   ├── serving/                    #   BentoML service + latency benchmark
│   ├── config/                     #   Env loading, data config
│   └── enums/                      #   Shared constants (paths, experiments, models)
│
├── airflow/dags/                   # Nightly batch-scoring DAG
├── configs/                        # Hydra configs (train.yaml, distill.yaml)
├── deploy/                         # nginx_canary.conf (95/5 weighted front)
│
├── data/                           # DVC-tracked (git-ignored)
│   ├── raw/                        #   Source CSVs (.dvc pointers committed)
│   ├── interim/                    #   normalized.csv
│   ├── processed/                  #   train/val/test splits
│   └── scoring/                    #   input/ · output/ · reference/ panels
│
├── models/                         # Trained checkpoints + ONNX graphs
│   ├── baseline/ · distilled/      #   HF checkpoints
│   └── onnx/                       #   model.onnx · distilled.onnx · model_int8.onnx
│
├── reports/                        # Committed evidence (evals, benchmarks, drift, locust)
├── scripts/                        # Sweeps, benchmarks, registry + canary ops
├── tests/                          # Unit tests (hermetic)
├── notebooks/                      # EDA
├── docs/                           # project_analysis.md · labeling_guidelines.md
│
├── bentofile.yaml                  # Bento build recipe (serving-only deps + baked int8)
├── docker-compose.yml              # prod worker + canary worker + nginx + redis
├── dvc.yaml · params.yaml          # Data pipeline stages + their parameters
├── .dvc/                           # DVC config & cache
├── .pre-commit-config.yaml         # ruff, ruff-format, mypy, DVC hooks
├── pyproject.toml · uv.lock        # Dependencies
└── AGENTS.md                       # Hard-won project notes for AI agents
```

> **Note:** `data/**` and model weights are git-ignored — only `.dvc` pointer files and reports are committed. Run `uv run dvc pull` before anything that touches data.

---

## Datasets

| Dataset | Size | Role |
|---|---|---|
| **[Arabic Customer Reviews (`Final_Data.csv`)](https://www.kaggle.com/datasets/mohamedramadan2040/arabic-customer-reviews)** | ~40 k rows (4.4 MB) | **Primary** — drives the DVC pipeline, all training, and every reported metric |
| **[330K Arabic Sentiment Reviews (`arabic_sentiment_reviews.csv`)](https://www.kaggle.com/datasets/abdallaellaithy/330k-arabic-sentiment-reviews)** | 330 k rows (212 MB) | Secondary corpus for EDA / pretraining exploration (binary-labeled) |

Preprocessing on the primary set (`reports/preprocess_metrics.json`): 40 046 raw rows → **36 045** clean rows (1 939 exact + 2 062 near-duplicates removed, 5 952 near-empty flagged), then a label- **and** dialect-stratified split locked by `split.random_state` in `params.yaml`:

| Split | Rows | positive | negative | neutral |
|---|---:|---:|---:|---:|
| `train` | 25 231 | 14 533 | 9 419 | 1 279 |
| `val` | 3 605 | 2 076 | 1 346 | 183 |
| `test` | 7 209 | 4 152 | 2 691 | 366 |

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
uv run pytest                # Tests (96)
```

---

## Data Pipeline

A DVC pipeline (`dvc.yaml`) turns the raw dataset into reproducible, dialect-tagged splits. Stages read `params.yaml` and auto-log to the `raay_preprocessing` MLflow experiment.

```bash
uv run dvc repro             # preprocess → split
```

| Stage | Command | Outputs |
|---|---|---|
| `preprocess` | `python -m raay.data.preprocess` | `data/interim/normalized.csv`, `reports/preprocess_metrics.json` |
| `split` | `python -m raay.data.split` | `data/processed/{train,val,test}.csv` (+ `dialect`, `dialect_confidence`) |

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

| Metric | Value |
|---|---|
| Accuracy | **84.92 %** |
| F1 (macro) | **0.6407** |
| F1 (weighted) | 0.8414 |

**Per-class**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---:|
| Positive | 0.880 | 0.909 | **0.895** | 4 152 |
| Negative | 0.846 | 0.853 | **0.850** | 2 691 |
| Neutral | 0.245 | 0.139 | **0.178** | 366 |

> **Neutral underperforms** because it is only ~5 % of the test set — the dominant remaining error mode.

**Dialect breakdown**

| Dialect | n | Accuracy | F1 macro |
|---|---:|---:|---:|
| MSA | 3 322 | 85.7 % | 0.605 |
| Gulf | 1 145 | 86.2 % | 0.682 |
| Egyptian | 1 011 | 82.4 % | 0.647 |
| Levantine | 1 129 | 82.4 % | 0.631 |
| Maghrebi | 357 | 92.2 % | 0.668 |
| Arabizi | 245 | 80.0 % | 0.538 |

---

## Phase 4: Compression & Optimization

Cut inference cost without losing quality: **distillation → ONNX export → INT8 quantization**, with every variant benchmarked on accuracy *and* latency and tracked in MLflow.

### 4.1 Knowledge Distillation

A 6-layer student distilled from the fine-tuned teacher on Kaggle GPU:

```bash
uv run python scripts/kaggle_train_runs.py --module distill --n 6
uv run python -m raay.training.distill alpha=0.4 temperature=4.0   # Hydra overrides
```

| Metric | Teacher (baseline) | Student (distilled) |
|---|---:|---:|
| Accuracy | 84.92 % | 83.04 % |
| F1 (macro) | 0.6407 | 0.5975 |
| Size | 542.6 MB | 372.5 MB |

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

| Variant | Backend | Accuracy | F1 (macro) | Size | batch-1 p50 | batch-1 p95 |
|---|---|---:|---:|---:|---:|---:|
| `baseline-torch` | PyTorch CPU FP32 | 84.92 % | 0.6407 | 542.6 MB | 31.2 ms | 69.4 ms |
| `distilled-torch` | PyTorch CPU FP32 | 83.04 % | 0.5975 | 372.5 MB | 22.2 ms | 45.8 ms |
| `onnx-fp32` | ORT CPU FP32 | 84.92 % | 0.6407 | 540.9 MB | 19.0 ms | 43.0 ms |
| **`onnx-int8`** | **ORT CPU INT8** | **85.03 %** | 0.6369 | **136.1 MB** | **9.5 ms** | **27.6 ms** |

INT8 is **4× smaller** (541 MB → 136 MB) and **2× faster at p50** than the FP32 graph (19.0 → 9.5 ms; 1.6× at p95), while *gaining* 0.11 pp accuracy — quantization noise, not a real improvement. `onnx-fp32` accuracy is inherited from `baseline-torch` — identical weights.

### 4.4 MLflow Model Registry

```bash
uv run python scripts/log_variants_mlflow.py       # --winner onnx-int8
```

Each variant is logged as its own `raay_training` run (`stage=baseline|distilled|fp32|int8`) with accuracy, F1, latency, and size metrics, then the winner is registered and promoted.

| Version | Variant | Stage | Alias |
|---|---|---|---|
| `v5` | `distilled-fp32` (canary graph) | Production | `Production`, `Canary` |
| `v4` | `onnx-int8` | Archived | — |
| `v1` | baseline transformers | Archived | — |

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
    {"label": "positive", "score": 0.98},
    {"label": "negative", "score": 0.94}
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

| Variant | Requests | Median | p95 | p99 | Failures |
|---|---:|---:|---:|---:|---:|
| FP32 ONNX | 347 | 78 ms | 160 ms | 270 ms | 0 |
| **INT8 ONNX** | 364 | **54 ms** | **120 ms** | **160 ms** | 0 |

> **TensorRT** is retained as a future accelerator path. An engine must be built on the fixed production GPU / CUDA / TensorRT environment — never ahead of it.

### 5.3 Containerized Serving

```bash
uv run bentoml build -f bentofile.yaml
uv run bentoml containerize raay-sentiment:latest --image-tag raay-sentiment:latest
docker compose up -d           # prod :8000 · canary :8081 (nginx) · redis :6379
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

| Mode | Reads | Writes |
|---|---|---|
| `init-reference` | `data/processed/test.csv` | `data/scoring/reference/reference.csv` (fixed seed) |
| `make-input` | `data/processed/test.csv` | `data/scoring/input/{date}.csv` (seed = CRC32 of the date → idempotent) |
| `score` | the day's input | `data/scoring/output/{date}.csv` (3-class probs + `predicted_label` / `predicted_score`) |
| `drift` | reference vs. the day's output | `reports/drift/{date}.json` |

The drift gate is **Evidently PSI** on `predicted_label` + `positive`: `< 0.1` PASS, `< 0.2` WARN, `≥ 0.2` FAIL (overall = worst column). Every mode logs to the `raay_batch` MLflow experiment.

Latest verdicts: `2026-09-23` **PASS** (label PSI 0.0003, positive 0.0193) · `2026-09-24` **PASS** (label 0.0034, positive 0.0348).

### 5.6 Airflow Orchestration

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

The DAG `airflow/dags/raay_nightly_batch_scoring.py` runs daily at **03:00 UTC**: `materialize_daily_input → score_daily_batch → run_drift_check`, each a `BashOperator` shelling out to `uv run python -m raay.inference.batch_score`.

> `LocalExecutor` is hard-blocked on SQLite — the runtime uses `SequentialExecutor`. Only change the executor while the scheduler is **stopped**, and trust the scheduler's own `/proc/<pid>/environ` over `airflow config get-value`.

### 5.7 Canary Rollout

An additive nginx front on `:8081` splitting **95/5** across two compose workers:

| Worker | Graph | Registry alias | Host port |
|---|---|---|---|
| `raay-sentiment` | INT8 (production) | `Production` | `8000` |
| `raay-sentiment-canary` | distilled FP32 | `Canary` | none (nginx reaches it by service name) |

```bash
uv run python scripts/canary_promote.py --mode declare    # register the canary graph (idempotent)
docker compose up -d                                      # brings up both workers + nginx
curl -s :8081/health                                      # proves the 5 % slice is live

uv run python scripts/canary_promote.py --mode promote     # health-gate :8000 + :8081, then flip Production
uv run python scripts/canary_promote.py --mode rollback    # flip Production back to INT8
```

Both variants live under the **same** registered model; the canary worker binds `distilled.onnx` + `.onnx.data` + `models/distilled/final` read-only, so the alias is the promotion gate rather than the 5 % graph source. 14 hermetic tests in `tests/test_canary_nginx.py` (config-text parse + fake MLflow client — no nginx binary needed).

---

## Testing

96 hermetic unit tests — no GPU, no network, no running servers:

| Suite | Tests | Covers |
|---|---:|---|
| `tests/test_batch_consumer.py` | 20 | Micro-batch drain, fake Redis / in-memory queue, fake scorer |
| `tests/test_batch_score.py` | 14 | Input sampling, scoring, PSI drift verdicts |
| `tests/test_canary_nginx.py` | 14 | nginx config parse + canary promote/rollback against a fake MLflow client |
| `tests/test_serving.py` | 13 | BentoML service, health middleware, 422 validation |
| `tests/test_export_onnx.py` | 9 | Export parity helpers |
| `tests/test_preprocess.py` | 9 | Normalization, dedup, near-empty flagging |
| `tests/test_dialect.py` | 7 | Dialect heuristics + confidence |
| `tests/test_distill.py` | 7 | Distillation loss / config wiring |
| `tests/test_quantize_onnx.py` | 3 | Quantization + parity report |

```bash
uv run pytest
```

---

## Limitations & Honest Findings

Stated plainly, because the numbers above are only useful with their caveats:

- **Micro-batching does not beat serial scoring on this box.** At batch 32 the INT8 graph costs ~58–84 ms *per item* vs ~12 ms single — a `pure_cpu` speedup of **0.25×** (0.50× once a 20 ms per-call overhead is modeled). Batching still wins on *call amortization* (128 reviews → 4 `session.run` calls) and on batch-throughput-bound backends (GPU / TensorRT) or high per-call HTTP overhead. On a 2-core CPU, treat `pure_cpu < 1` as expected, not a bug.
- **The canary latency signal is not real.** Both workers share 2 cores, so the 5 % slice's latency sawtooth is CPU contention. The behavioral gate is the nightly PSI drift job, not nginx latency.
- **INT8 "gaining" accuracy is noise.** 85.03 % vs 84.92 % is within quantization noise on 7 209 samples; the honest claim is "no meaningful quality loss at 4× compression".
- **Neutral remains the weak class** (F1 0.178) — a class-imbalance problem, not a modeling one.
- **No CI.** `.github/workflows/` is empty; quality gates are pre-commit hooks plus the documented local command.
- **No local GPU.** Training and distillation run on Kaggle; local work is CPU-only inference.

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `MLFLOW_TRACKING_URI` | MLflow tracking store | `sqlite:///mlflow.db` in `.env` (code fallback: `file:./mlruns`) |
| `MLFLOW_ALLOW_FILE_STORE` | Required by MLflow ≥ 3 for `file:` URIs | auto-set when the URI is `file:` |
| `RAAY_ONNX_PATH` | Explicit ONNX graph override (skips registry resolution) | unset → resolves `models:/ArabicSentiment/Production` |
| `RAAY_TOKENIZER_DIR` | Tokenizer + `id2label` source | `models/baseline/final` |
| `RAAY_MAX_LENGTH` | Tokenizer truncation length | `128` |
| `RAAY_MODEL_NAME` | `ArabertPreprocessor` model | `aubmindlab/bert-base-arabertv02` |
| `RAAY_REGISTERED_MODEL` | MLflow registered model name | `ArabicSentiment` |
| `RAAY_ALIAS` | Registry alias resolved at worker start | `Production` |
| `DATA_ROOT` | Kaggle snapshot dir holding the processed splits | — |
| `KAGGLE_TEACHER_DIR` | Kaggle teacher checkpoint dir (distillation) | — |

---

## License

This project is licensed under the MIT License.

---

<div align="center">

**Raay راي** — Giving every Arabic review a voice.

</div>
