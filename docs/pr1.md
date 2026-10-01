# PR #1 — Project Setup & Infrastructure Scaffolding

## Summary

the complete foundation for **Raay (راي)**, an Arabic e-commerce product review sentiment analysis system. It covers project initialization, dependency management, code quality tooling, experiment tracking infrastructure, data versioning, and project documentation.

---

## What Was Done

### 1. Project Initialization

- Initialized a new Python 3.12+ project using **uv** as the package manager (`uv init --package --name raay .`)
- Created the `src/raay/` package structure with four core modules:
  - `data/` — Data loading, preprocessing, and dialect handling
  - `training/` — Model training loops, fine-tuning, and distillation
  - `inference/` — Batch and real-time prediction pipelines
  - `serving/` — BentoML service definitions for REST API serving
- Set up `pyproject.toml` with all project metadata, dependencies, and build configuration
- Generated a locked dependency file (`uv.lock`) for reproducible installs

### 2. Dependency Stack

**Core dependencies:**


| Package           | Purpose                                |
| ------------------- | ---------------------------------------- |
| `torch`           | Deep learning framework                |
| `transformers`    | HuggingFace models (AraBERT, etc.)     |
| `datasets`        | HuggingFace datasets library           |
| `accelerate`      | Distributed / mixed-precision training |
| `pandas`, `numpy` | Data manipulation                      |
| `mlflow`          | Experiment tracking & model registry   |
| `dvc[s3]`         | Data versioning with S3 support        |
| `bentoml`         | Model serving & API packaging          |
| `pydantic`        | Data validation & typed schemas        |
| `hydra-core`      | Hierarchical configuration management  |
| `python-dotenv`   | Environment variable loading           |
| `loguru`          | Structured logging                     |
| `typer`           | CLI framework                          |

**Dev dependencies:**


| Package                 | Purpose                  |
| ------------------------- | -------------------------- |
| `pytest`, `pytest-cov`  | Testing & coverage       |
| `ruff`                  | Linting & formatting     |
| `mypy`                  | Static type checking     |
| `pre-commit`            | Git hook management      |
| `ipykernel`, `notebook` | Jupyter notebook support |

### 3. Code Quality & Pre-commit Hooks

Configured `.pre-commit-config.yaml` with the following hooks:

- **pre-commit-hooks** (v4.6.0):

  - `trailing-whitespace` — Remove trailing whitespace
  - `end-of-file-fixer` — Ensure files end with a newline
  - `check-yaml` — Validate YAML syntax
  - `check-added-large-files` — Block files > 5 MB
  - `check-merge-conflict` — Detect merge conflict markers
  - `mixed-line-ending` — Enforce consistent line endings
- **Ruff** (v0.5.6):

  - `ruff` — Linting with auto-fix (`--fix`)
  - `ruff-format` — Code formatting
- **mypy** (v1.10.1):

  - Static type checking with `--ignore-missing-imports`
  - Pydantic plugin enabled (`pydantic>=2.7.0`)
  - Tests excluded from type checking
- **DVC** (v3.67.1):

  - `dvc-pre-commit` — Run on commit stage
  - `dvc-pre-push` — Run on push stage
- **CI integration**: Auto-fix PRs enabled, monthly auto-update schedule

### 4. Data Directory Structure

```
data/
├── raw/          # Original scraped/downloaded reviews (.gitkeep)
├── interim/      # Intermediate cleaned data (.gitkeep)
└── processed/    # Final train/val/test splits
```

All data directories are excluded from Git via `.gitignore` and tracked through DVC.

### 5. MLflow Experiment Tracking Server

- Created a Dockerized MLflow server setup:
  - Base image: `python:3.11-slim`
  - MLflow version: `2.16.0`
  - Backend store: `/mlflow/mlruns` (file-based)
  - Artifact root: `/mlflow/artifacts`
  - Exposed on port `5000`
  - Persistent storage via Docker volume (`mlflow_data`)
- Environment variable: `MLFLOW_TRACKING_URI=http://localhost:5000`
- `.env.example` template provided for team onboarding

### 6. DVC Data Versioning

- Initialized DVC in the project (`dvc init`)
- Configured a default remote storage backend:
  - Local storage: `~/dvc-storage/raay` (for development)
  - S3 support available via `dvc[s3]` dependency
- Added `.dvc/` and `.dvcignore` to Git tracking
- DVC hooks integrated into pre-commit pipeline

### 7. Git & Version Control

- Initialized Git repository with `main` as the default branch
- Comprehensive `.gitignore` covering:
  - Python artifacts (`__pycache__/`, `*.py[cod]`, `*.egg-info/`)
  - Virtual environments (`.venv/`)
  - ML artifacts (`*.pt`, `*.onnx`, `*.engine`)
  - Data directories (`data/`)
  - MLflow runs (`mlruns/`)
  - IDE configs (`.idea/`, `.vscode/`)
  - Environment files (`.env`)

### 8. Documentation

- **`docs/project_analysis.md`** — Business problem framing, ML task definition, success metrics, constraints, and risks
- **`docs/labeling_guidelines.md`** — Annotation rules for 3-class sentiment labeling:
  - Label definitions (Positive=2, Negative=0, Neutral=1)
  - Edge case handling (sarcasm, mixed reviews, dialect, seasonal vocab, rating mismatches)
  - Annotation process (2 annotators + adjudicator, Cohen's Kappa ≥ 0.75)
  - Class balance targets (45% Pos / 35% Neg / 20% Neu for training)
  - Versioned guidelines with re-audit triggers
- **`docs/commands_run.txt`** — Complete reference of all setup commands

---

## Project Architecture Overview

```
raay/
├── src/raay/                  # Main Python package
│   ├── __init__.py            #   Package entry point
│   ├── data/                  #   Data pipeline module
│   ├── training/              #   Training module
│   ├── inference/             #   Inference module
│   └── serving/               #   Serving module
├── configs/                   # Hydra config files (to be populated)
├── data/{raw,interim,processed}/  # DVC-tracked data
├── models/                    # Trained checkpoints
├── mlflow/                    # MLflow local artifacts
├── scripts/                   # Automation scripts
├── tests/                     # Test suite
├── docs/                      # Project documentation
├── .github/workflows/         # CI/CD pipelines (to be added)
├── .dvc/                      # DVC configuration
├── .pre-commit-config.yaml    # Code quality hooks
├── pyproject.toml             # Project metadata & deps
└── uv.lock                   # Locked dependencies
```

---

## How to Verify

```bash
# Install dependencies
uv sync --all-extras

# Run all quality checks
uv run pre-commit run --all-files
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest

# Verify package import
uv run python -c "import raay; print(raay.__file__)"

# Start MLflow server (requires Docker)
docker start mlflow-server
# → http://localhost:5000
```

---

## What's Next

- [ ]  Data collection & preprocessing pipeline (`src/raay/data/`)
- [ ]  Hydra configuration files (`configs/`)
- [ ]  AraBERT fine-tuning training loop (`src/raay/training/`)
- [ ]  Model distillation (12-layer → 6-layer)
- [ ]  Batch & real-time inference (`src/raay/inference/`)
- [ ]  BentoML serving endpoint (`src/raay/serving/`)
- [ ]  CI/CD GitHub Actions workflows
- [ ]  Unit & integration tests
