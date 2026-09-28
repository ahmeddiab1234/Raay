<div align="center">

---

## Overview

**Raay** (Arabic: _رأي_, meaning "opinion") is a production-grade sentiment analysis system designed for Arab e-commerce platforms. It classifies ~50,000 Arabic product reviews per day into three sentiment classes — **Positive**, **Negative**, and **Neutral** — to power product ranking, seller rating aggregation, and customer-service ticket prioritization.

The project covers the full ML lifecycle: data versioning, experiment tracking, model training, inference optimization, and model serving.

---

## Features

| Category                      | Details                                                                               |
| ----------------------------- | ------------------------------------------------------------------------------------- |
| **3-Class Sentiment**   | Positive · Negative · Neutral classification with confidence scores                 |
| **Arabic NLP**          | MSA + dialect support (Egyptian, Gulf, Levantine, Arabizi/franco-arabe)               |
| **Transformer-based**   | AraBERT backbone with HuggingFace Transformers & Accelerate                           |
| **Model Optimization**  | Knowledge distillation (6-layer student), ONNX export, dynamic INT8 quantization      |
| **Experiment Tracking** | MLflow for metric logging, model registry (`ArabicSentiment`), and artifact storage |
| **Data Versioning**     | DVC with configurable remote storage (local / S3)                                     |
| **Code Quality**        | Ruff linter & formatter, mypy static type checking, pre-commit hooks                  |
| **Testing**             | pytest + pytest-cov test suite                                                        |
| **Configuration**       | Hydra-based hierarchical config management                                            |
| **Serving**             | BentoML for model packaging and REST API serving                                      |
| **Typed Schemas**       | Pydantic models for data validation across the pipeline                               |
| **Structured Logging**  | Loguru for structured, leveled logging                                                |
| **CLI**                 | Typer-based command-line interface                                                    |

---

## Project Architecture

```
raay/
├── src/raay/                  # Main Python package
│   ├── data/                  #   Data loading, preprocessing, dialect handling
│   ├── training/              #   Training loops, fine-tuning, distillation
│   ├── inference/             #   Batch & real-time prediction
│   └── serving/               #   BentoML service definitions
│
├── configs/                   # Hydra YAML configuration files
├── data/
│   ├── raw/                   # Original scraped / downloaded reviews
│   ├── interim/               # Intermediate cleaned data
│   └── processed/             # Final train/val/test splits
│
├── models/                    # Trained model checkpoints
├── mlflow/                    # MLflow local artifacts
├── scripts/                   # Utility & automation scripts
├── tests/                     # Unit & integration tests
├── docs/                      # Documentation & guidelines
│   ├── project_analysis.md    #   Business & ML problem framing
│   ├── labeling_guidelines.md #   Annotation rules & edge cases
│   ├── pr1.md … pr4.md        #   Phase retrospectives & reports
│   └── commands_run_p*.txt    #   Phase command execution records
│
├── .github/workflows/         # CI/CD pipeline definitions
├── .dvc/                      # DVC configuration & cache
├── .pre-commit-config.yaml    # Pre-commit hook configuration
├── pyproject.toml             # Project metadata & dependencies
└── uv.lock                   # Locked dependency versions
```

---

## Tech Stack

| Layer                | Tool                                          |
| -------------------- | --------------------------------------------- |
| Language             | Python 3.12+                                  |
| Package Manager      | [uv](https://docs.astral.sh/uv/)               |
| Deep Learning        | PyTorch, HuggingFace Transformers, Accelerate |
| Experiment Tracking  | MLflow (Dockerized server)                    |
| Data Versioning      | DVC (S3 / local remote)                       |
| Model Serving        | BentoML                                       |
| Model Optimization   | ONNX, ONNX Runtime (FP32 / dynamic INT8)      |
| Config Management    | Hydra                                         |
| Data Validation      | Pydantic                                      |
| Linting & Formatting | Ruff                                          |
| Type Checking        | mypy                                          |
| Testing              | pytest, pytest-cov                            |
| Pre-commit           | pre-commit (ruff, mypy, DVC hooks)            |
| Logging              | Loguru                                        |
| CLI                  | Typer                                         |

---

## Datasets (Data Collection)

The system currently relies on the following key datasets for training and validation:

1. **[330K Arabic Sentiment Reviews (arabic_sentiment_reviews.csv)](https://www.kaggle.com/datasets/abdallaellaithy/330k-arabic-sentiment-reviews)**

   - **Size:** 330,000 reviews (212.23 MB)
   - **Description:** A large-scale binary sentiment dataset containing Arabic product reviews. It is labeled with `1` for positive and `0` for negative. This provides a massive foundation for training robust Arabic NLP models.
2. **[Arabic Customer Reviews (Final_Data.csv)](https://www.kaggle.com/datasets/mohamedramadan2040/arabic-customer-reviews)**

   - **Size:** ~36,000 reviews (4.44 MB)
   - **Description:** Customer reviews in Arabic collected from various companies and products. Includes review text, sentiment ratings, and the associated company. This dataset is actively used in the current preprocessing and modeling version.

*Note: The raw datasets are tracked via DVC and are not directly included in the Git repository.*

---

## Installation & Usage

### Prerequisites

- **Linux or Windows Subsystem for Linux (WSL)**
- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** package manager
- **Docker** (for MLflow server)
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
#   MLFLOW_TRACKING_URI=http://localhost:5000
```

### 5. Set Up MLflow Server

```bash
# Build and run the MLflow server container
mkdir -p ~/mlflow-server

# Create Dockerfile (see docs/commands_run.txt for full Dockerfile)
docker build -t mlflow-server ~/mlflow-server
docker run -d \
  --name mlflow-server \
  -p 5000:5000 \
  -v mlflow_data:/mlflow \
  mlflow-server

# Verify at http://localhost:5000
```

### 6. Set Up DVC

```bash
# Initialize DVC (already done if cloning this repo)
uv run dvc init

# Configure remote storage (local example)
mkdir -p ~/dvc-storage/raay
uv run dvc remote add -d storage ~/dvc-storage/raay

# Pull versioned data
uv run dvc pull
```

### 7. Install Pre-commit Hooks

```bash
uv run pre-commit install
uv run pre-commit install --hook-type pre-push
```

### 8. Run Quality Checks

```bash
uv run ruff check .          # Lint
uv run ruff format --check . # Format check
uv run mypy src              # Type check
uv run pytest                # Tests
```

### 9. Run the Data Pipeline (Preprocessing & Splitting)

We use a DVC pipeline (`dvc.yaml`) to run data preprocessing and splitting in a reproducible manner. This processes the raw datasets (`data/raw/`) into a normalized intermediate form, and splits them into `train.csv`, `val.csv`, and `test.csv` sets based on `params.yaml`.

```bash
# Reproduce the preprocessing and splitting steps
uv run dvc repro
```

This pipeline automatically tracks the steps using **MLflow** (logging metrics like min/max string length, exact/fuzzy duplicates removed, and final split row counts) and outputs reports to `reports/preprocess_metrics.json`.

---

## Baseline Model Training

AraBERT v2 (`aubmindlab/bert-base-arabertv02`) was fine-tuned on the processed splits via a **6-run [Kaggle](https://www.kaggle.com/code/codecaoch/raay-training/edit/run/346908504) GPU sweep** (varying `lr` and `batch_size`). The best run was registered as `ArabicSentiment → Production` in the MLflow Model Registry.

```bash
# Training script (Hydra config, run on Kaggle GPU)
uv run python -m raay.training.train [lr=2e-5 batch_size=32]

# Evaluate against the held-out test set
uv run python -m raay.training.evaluate \
  --model-dir models/baseline/final \
  --test-file data/processed/test.csv
```

### Baseline Results (`reports/eval_baseline.json`)

| Metric        | Value            |
| ------------- | ---------------- |
| Accuracy      | **84.9 %** |
| F1 (macro)    | **0.641**  |
| F1 (weighted) | 0.841            |

**Per-class F1:**

| Class    | Precision | Recall | F1              | Support |
| -------- | --------- | ------ | --------------- | ------- |
| Positive | 0.880     | 0.909  | **0.895** | 4 152   |
| Negative | 0.846     | 0.853  | **0.850** | 2 691   |
| Neutral  | 0.245     | 0.139  | **0.178** | 366     |

> **Note:** Neutral class underperforms due to heavy class imbalance (~5 % of test set).

**Dialect breakdown:**

| Dialect   | n     | Accuracy | F1 macro |
| --------- | ----- | -------- | -------- |
| MSA       | 3 322 | 85.7 %   | 0.605    |
| Gulf      | 1 145 | 86.2 %   | 0.682    |
| Egyptian  | 1 011 | 82.4 %   | 0.647    |
| Levantine | 1 129 | 82.4 %   | 0.631    |
| Maghrebi  | 357   | 92.2 %   | 0.668    |
| Arabizi   | 245   | 80.0 %   | 0.538    |

---

## Compression & Optimization

To cut inference cost without sacrificing quality, the baseline AraBERT model went through **knowledge distillation**, **ONNX export**, and **INT8 quantization** — with every variant benchmarked (accuracy vs. latency) and tracked in MLflow.

### 1. Knowledge Distillation

A 6-layer student was distilled from the fine-tuned teacher on a Kaggle GPU (`uv run python scripts/kaggle_train_runs.py --module distill --n 6`). The best student was evaluated on the held-out test set:

| Metric     | Teacher (baseline) | Student (distilled) |
| ---------- | -----------------: | ------------------: |
| Accuracy   |            84.92 % |             83.04 % |
| F1 (macro) |             0.6407 |              0.5975 |
| Size       |           542.6 MB |            372.5 MB |

### 2. ONNX Export & INT8 Quantization

Both checkpoints were exported to ONNX (dynamic batch/sequence dims, logit-parity validated vs. PyTorch via `raay.inference.export_onnx`), and the teacher graph was dynamically quantized to a self-contained INT8 model.

```bash
# Export teacher + student to ONNX, validate PyTorch-vs-ORT parity, log to raay_training
uv run python -m raay.inference.export_onnx

# Dynamic INT8 quantization (~136 MB self-contained vs 541 MB fp32 graph)
uv run python -m raay.inference.quantize_onnx
```

INT8 kept accuracy (85.03 %) at **4x compression** with no meaningful quality drop: `reports/onnx_parity.json`, `onnx_int8_parity.json`, `eval_int8.json`.

| Variant         |          Accuracy | F1 (macro) |               Size |   CPU batch-1 p95 |
| --------------- | ----------------: | ---------: | -----------------: | ----------------: |
| baseline-torch  |           84.92 % |     0.6407 |           542.6 MB |           69.4 ms |
| distilled-torch |           83.04 % |     0.5975 |           372.5 MB |           45.8 ms |
| onnx-fp32       |           84.92 % |     0.6407 |           540.9 MB |           43.0 ms |
| onnx-int8       | **85.03 %** |     0.6369 | **136.1 MB** | **27.6 ms** |

### 3. Serving & Latency Benchmark

The CPU INT8 graph is served through BentoML and benchmarked as the GPU/TensorRT decision gate.

```bash
# Serve CPU INT8 (default: MLflow registry alias models:/ArabicSentiment/Production)
uv run bentoml serve src/raay/serving/serve.py:svc --reload

# Single-call latency (p50/p95/p99) → reports/serving_benchmark.json + MLflow metrics
uv run python -m raay.serving.benchmark --samples 500

# Unified 4-variant table → reports/benchmark_table.{md,csv}
uv run python scripts/benchmark.py
# HTTP load test (fp32 vs int8) through the same BentoML service
uv run python scripts/locust_run.py --users 8 --run-time 60
```

The service exposes `POST /predict` — send a list of Arabic review texts and get back predictions with labels and confidence scores:

```bash
curl -s http://localhost:3000/predict -H 'Content-Type: application/json' \
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

Single-call INT8 latency was **p50 ≈ 12 ms, p95 ≈ 36 ms, p99 ≈ 65 ms**, inside the product SLA — so CPU INT8 shipped. **TensorRT** is retained as a future accelerator path; the engine must be built on the fixed production GPU/CUDA/TensorRT, never ahead of it.

Locust load test (8 users, 60 s, 3-review calls, 0 failures):

| Variant   |          Median |              p95 |              p99 | Failures |
| --------- | --------------: | ---------------: | ---------------: | -------: |
| FP32 ONNX |           56 ms |           130 ms |           160 ms |        0 |
| INT8 ONNX | **38 ms** | **110 ms** | **130 ms** |        0 |

### 4. MLflow Model Registry

All four variants were logged as `raay_training` runs (stage-tagged `baseline / distilled / fp32 / int8` with accuracy, F1, latency, and size metrics); the best tradeoff was registered and promoted.

```bash
uv run python scripts/log_variants_mlflow.py
```

- **`ArabicSentiment` v4 = Production** — onnx-int8 (acc 85.03 %, F1 0.6369, CPU p95 27.6 ms, 136.1 MB)
- **`ArabicSentiment` v1 = Archived** — baseline torch (Kaggle-backed)
- BentoML resolves the registry **`Production` alias** by default; `RAAY_ONNX_PATH` overrides it for benchmarks.

Full execution record: [`docs/commands_run_p4.txt`](docs/commands_run_p4.txt) · report: [`docs/pr4.md`](docs/pr4.md)

---

## Serving Phase

### 1. Containerized Serving

Package the CPU INT8 model into a BentoML image, containerize, and serve with health + liveness:

```bash
# Build the bento (bakes models/onnx/model_int8.onnx + baseline tokenizer)
uv run bentoml build -f bentofile.yaml
uv run bentoml containerize raay-sentiment:latest --image-tag raay-sentiment:latest
# Run the stack: service (:8000) + Redis (:6379)
docker compose up -d
```

- `GET /health` → `{"status": "healthy"}` (liveness probe; empty `texts: []` is a 200-empty by design, malformed payloads get 422 from pydantic)
- `POST /predict` → `{"texts": [...]}` → predictions with label + score
- Env-tunable: `RAAY_ONNX_PATH`, `RAAY_TOKENIZER_DIR`, `RAAY_MAX_LENGTH`, `RAAY_MODEL_NAME`, `RAAY_REGISTERED_MODEL`, `RAAY_ALIAS`
- Default ONNX source = registry alias `models:/ArabicSentiment/Production`; `RAAY_ONNX_PATH` overrides it (what the benchmark/Locust drivers set)

### 2. Near-Real-Time Micro-Batch Consumer

Client-side micro-batching over Redis — NOT `batchable=True` in `serve.py`:

```bash
# Benchmark (in-memory queue by default; --redis forces Redis)
uv run python -m raay.inference.batch_consumer --mode benchmark \
    --samples 128 --max-batch 32 --duration 10 --max-idle 2 --overhead-ms 20
# Producer / consumer modes against the compose Redis
uv run python -m raay.inference.batch_consumer --mode producer
uv run python -m raay.inference.batch_consumer --mode consumer
```

- Polls Redis list `reviews`, drains up to `--max-batch 32` or `--max-wait-ms` (default 100) — whichever comes first — and scores the micro-batch in ONE in-process INT8 ORT call
- Published results → `reviews-results` (`{"id"?,"text","label","score"}`)
- 20 hermetic tests in `tests/test_batch_consumer.py` (MemoryQueue/fake-Redis/fake-scorer)
- **Honest local finding**: on this 2-core CPU int8 is latency-optimal at batch 1 (batch-32 per-item ≈57–80 ms vs ~12 ms single → `pure_cpu` speedup ≈0.26x); batching wins via call amortization (128 reviews → 4 `session.run` calls) and on batch-throughput-bound backends (GPU/TensorRT) or high per-call overhead

### 3. Nightly Batch Re-Scoring + Drift

Daily re-score of a seeded `data/processed/test.csv` sample through the INT8 graph, then Evidently PSI drift check:

```bash
# Deterministic per-day input (seed = CRC32 of date)
uv run python -m raay.inference.batch_score --mode make-input --date YYYY-MM-DD --samples 1200
uv run python -m raay.inference.batch_score --mode score --date YYYY-MM-DD
uv run python -m raay.inference.batch_score --mode drift --date YYYY-MM-DD     # after init-reference
```

- Input: `data/scoring/input/{date}.csv` (adds a `day` column, penalizes missing text) · Output: `data/scoring/output/{date}.csv` (text + 3-class probs + predicted_label/score)
- `init-reference` seeds `data/scoring/reference/reference.csv` (fixed seed)
- Drift: `reports/drift/{date}.json` → `{"overall":"PASS|WARN|FAIL"}` (PSI <0.1 PASS, <0.2 WARN, ≥0.2 FAIL; overall = worst of `predicted_label` + `positive`)
- Scheduled daily 03:00 UTC via Airflow; every mode logs to MLflow `raay_batch`

### 4. Airflow Orchestration

Host-isolated Airflow 2.10.5, **not** in project deps (SQLite + `SequentialExecutor` — LocalExecutor is hard-blocked on SQLite):

```bash
uv tool install "apache-airflow==2.10.5"                      # binary `airflow`
export AIRFLOW_HOME=/home/diab/Documents/Raay/airflow_runtime # runtime (git-ignored)
airflow db migrate                                           # SQLite airflow.db
airflow users create --username admin --password admin --role Admin \
    --firstname admin --lastname raay --email admin@raay.local
# Both MUST keep running:
nohup airflow scheduler >> airflow_runtime/logs/scheduler.out 2>&1 &
nohup airflow webserver --port 8080 >> airflow_runtime/logs/webserver.out 2>&1 &
airflow dags trigger raay_nightly_batch_scoring
```

- DAG `airflow/dags/raay_nightly_batch_scoring.py` runs daily 03:00 UTC: materialize_daily_input → score_daily_batch → run_drift_check
- Reference drift result (2026-09-23): **PASS** — predicted_label PSI 0.0003, positive 0.0193
- Note: Airflow's `config get-value` reports can disagree with the file (`/proc/<scheduler_pid>/environ` is the source of truth); only change the executor while the scheduler is **stopped**

### 5. Canary Rollout

Additive nginx front on `:8081` splitting 95/5 over two compose workers — real second variant (distilled fp32), not theater:

```bash
# Register the distilled fp32 ONNX as the Canary alias (idempotent)
uv run python scripts/canary_promote.py --mode declare
# Build + bring up both workers + nginx front
uv run bentoml build -f bentofile.yaml && uv run bentoml containerize raay-sentiment:latest --image-tag raay-sentiment:latest
docker compose up -d                                              # :8081 health: curl -s :8081/health
# Promotion gate — health-checks :8000 + :8081, then flips Production alias
uv run python scripts/canary_promote.py --mode promote            # or --mode rollback
```

- `deploy/nginx_canary.conf`: `nginx:alpine`, `listen 8081`, request-level weights — `raay-sentiment:3000 weight=95` (Production int8) + `raay-sentiment-canary:3000 weight=5` (Canary distilled fp32)
- Canary worker has NO host port — nginx reaches it by compose service name; it binds `distilled.onnx` + `.onnx.data` + `models/distilled/final` read-only
- Both variants live under the **same** registered model; the registry alias is the promotion gate, not the 5% graph source
- **Honest caveat**: 2-core box → both workers share cores, so 5% slice latency sawtooth is CPU contention, not a canary signal; the behavioral gate is the nightly PSI drift job, not nginx latency
- 14 hermetic tests in `tests/test_canary_nginx.py` (conf text parse + fake-MlflowClient; no nginx binary)

Full execution record: [`docs/commands_run_p5.txt`](docs/commands_run_p5.txt)

---

## Environment Variables

| Variable                | Description       | Example                   |
| ----------------------- | ----------------- | ------------------------- |
| `MLFLOW_TRACKING_URI` | MLflow server URL | `http://localhost:5000` |

---

## License

This project is licensed under the MIT License.

---

<div align="center">
