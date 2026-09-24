"""Nightly batch re-scoring job with an Evidently PSI drift check.

Phase 5 step 3: a day's worth of reviews is scored offline in LARGE batches
directly against the INT8 ONNX graph (``serve.predict_probs``, in-process --
no REST call), the predictions are written to ``data/scoring/output/`` and an
Evidently PSI drift check is run against a scored reference set. Airflow
orchestrates the three steps nightly (see ``airflow/dags/raay_nightly_batch_scoring.py``).

Modes
-----
- ``make-input``: emulate "a day's worth of reviews" by deterministically
  sampling ``data/processed/test.csv`` (seeded by date) into
  ``data/scoring/input/{date}.csv``. There is no live scraper in this repo, so
  the pool stands in for new production reviews.
- ``score``: score an input CSV in ``--batch-size`` chunks and write
  ``data/scoring/output/{date}.csv`` with per-class probabilities, the argmax
  label and its score. Enforces the ``--min-samples`` (default 1000) floor.
- ``init-reference``: build the drift reference once by scoring a seeded slice
  of the pool into ``data/scoring/reference/reference.csv``.
- ``drift``: Evidently PSI (``ColumnDriftMetric`` + ``psi_stat_test``) on the
  ``predicted_label`` and ``positive`` columns of the current day vs the
  reference; each column is PASS (<0.1) / WARN (<0.2) / FAIL (>=0.2) and the
  verdicts are written to ``reports/drift/{date}.json``.

Every run also logs params/metrics/artifacts to the ``raay_batch`` MLflow
experiment (unless ``--no-mlflow``).

Run (from the repo root any of these):

    uv run python -m raay.inference.batch_score --mode make-input --date 2026-09-24
    uv run python -m raay.inference.batch_score --mode score --date 2026-09-24
    uv run python -m raay.inference.batch_score --mode drift --date 2026-09-24
    uv run python -m raay.inference.batch_score --mode init-reference --samples 2000

Honest caveat: the pool is a single dataset, so real-world PSI drift will be ~0
by construction; the mechanism (and the Phase-5 monitoring hook it provides) is
the deliverable.
"""

from __future__ import annotations

import argparse
import json
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import pandas as pd
from loguru import logger
from transformers import AutoConfig, AutoTokenizer

from raay.config.env import load_environment, mlflow_tracking_uri
from raay.enums.constants import DefaultPaths, Experiments, Models
from raay.serving.serve import predict_probs


def _date_seed(date: str) -> int:
    """Deterministic uint32 seed for a date string (same day -> same sample)."""
    return zlib.crc32(date.encode("utf-8"))


@dataclass
class Scorer:
    """Lazy INT8 ONNX scorer sharing ``serve.predict_probs``.

    ``score`` returns the full row-softmax probability matrix so the nightly
    job can persist every class probability, not just the argmax.
    """

    onnx_path: str = DefaultPaths.ONNX_INT8_MODEL.value
    tokenizer_dir: str = DefaultPaths.BASELINE_MODEL.value
    model_name: str = Models.TEACHER.value
    max_length: int = 128
    batch_size: int = 64
    _session: Any = field(init=False, repr=False, default=None)
    _tokenizer: Any = field(init=False, repr=False, default=None)
    _id2label: dict[int, str] = field(init=False, repr=False, default_factory=dict)
    _loaded: bool = field(init=False, repr=False, default=False)

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._session = ort.InferenceSession(
            self.onnx_path, providers=["CPUExecutionProvider"]
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_dir)
        config = AutoConfig.from_pretrained(self.tokenizer_dir)
        raw = getattr(config, "id2label", None) or {}
        self._id2label = {int(k): v for k, v in raw.items()}
        self._loaded = True
        logger.info(
            f"Batch scorer loaded {self.onnx_path} labels={self._id2label} "
            f"providers={self._session.get_providers()}"
        )

    @property
    def label_columns(self) -> list[str]:
        return [self._id2label[i] for i in sorted(self._id2label)]

    def score(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, len(self.label_columns)))
        self._ensure_loaded()
        assert self._session is not None and self._tokenizer is not None
        return predict_probs(
            self._session,
            self._tokenizer,
            texts,
            self.model_name,
            self.max_length,
            batch_size=self.batch_size,
        )


def make_input(
    pool_csv: str,
    date: str,
    samples: int,
    out_csv: str,
    seed: int | None = None,
) -> pd.DataFrame:
    """Sample ``samples`` reviews from the pool (seeded by date) into ``out_csv``."""
    df = pd.read_csv(pool_csv)
    if len(df) < samples:
        raise ValueError(
            f"pool has {len(df)} rows, need at least {samples}; pick a smaller --samples"
        )
    rng = np.random.default_rng(seed if seed is not None else _date_seed(date))
    sample = df.sample(n=samples, random_state=rng).copy()
    sample.insert(0, "day", date)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out_csv, index=False)
    logger.info(f"Wrote {len(sample)} reviews to {out_csv}")
    return sample


def score_input(
    input_csv: str,
    out_csv: str,
    scorer: Scorer,
    min_samples: int,
    target_label: str = "label",
) -> dict[str, Any]:
    """Score every text in ``input_csv`` in batches and persist predictions."""
    df = pd.read_csv(input_csv)
    if "text" not in df.columns:
        raise ValueError(
            f"input CSV must have a 'text' column (got {list(df.columns)})"
        )
    n = len(df)
    if n < min_samples:
        raise ValueError(
            f"input has {n} rows; nightly batch should re-score at least "
            f"{min_samples} (got --min-samples {min_samples})"
        )
    t0 = time.perf_counter()
    probs = scorer.score(df["text"].astype(str).tolist())
    elapsed = time.perf_counter() - t0
    cols = scorer.label_columns
    for i, col in enumerate(cols):
        df[col] = probs[:, i]
    arg = np.argmax(probs, axis=1)
    df["predicted_label"] = [cols[int(i)] for i in arg]
    df["predicted_score"] = probs[np.arange(n), arg]
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    logger.info(f"Scored {n} reviews in {elapsed:.2f}s -> {out_csv}")
    prop = df["predicted_label"].value_counts().to_dict()
    return {
        "n_reviews": n,
        "elapsed_sec": elapsed,
        "batch_size": scorer.batch_size,
        "label_columns": cols,
        "class_proportions": {k: int(v) for k, v in prop.items()},
        "target_label": target_label,
    }


def init_reference(
    pool_csv: str,
    samples: int,
    out_csv: str,
    scorer: Scorer,
) -> pd.DataFrame:
    """Score a fixed slice of the pool once; the drift reference for all days."""
    make_input(pool_csv, "reference", samples, out_csv, seed=0)
    score_input(out_csv, out_csv, scorer, min_samples=1)
    return pd.read_csv(out_csv)


def drift_check(
    reference_csv: str,
    current_csv: str,
    out_json: str,
    date: str,
    thresholds: tuple[float, float] = (0.1, 0.2),
    drift_columns: tuple[str, ...] = ("predicted_label", "positive"),
) -> dict[str, Any]:
    """Evidently PSI drift between reference and a scored day; write verdicts."""
    from evidently.legacy.calculations.stattests.psi import psi_stat_test
    from evidently.legacy.metrics import ColumnDriftMetric
    from evidently.legacy.report import Report

    ref = pd.read_csv(reference_csv)
    cur = pd.read_csv(current_csv)
    report = Report(
        metrics=[
            ColumnDriftMetric(column_name=col, stattest=psi_stat_test)
            for col in drift_columns
        ]
    )
    report.run(reference_data=ref, current_data=cur)

    lo, hi = thresholds
    columns: dict[str, dict[str, Any]] = {}
    for entry in report.as_dict()["metrics"]:
        result = entry["result"]
        col = result["column_name"]
        score = float(result["drift_score"])
        decision = "PASS" if score < lo else ("WARN" if score < hi else "FAIL")
        columns[col] = {
            "drift_score": round(score, 4),
            "stattest": result["stattest_name"],
            "stattest_threshold": float(result["stattest_threshold"]),
            "drift_detected": bool(result["drift_detected"]),
            "decision": decision,
        }
    severity = {"PASS": 0, "WARN": 1, "FAIL": 2}
    decisions = [columns[c]["decision"] for c in columns]
    overall = max(decisions, key=severity.__getitem__)
    verdict = {
        "date": date,
        "thresholds": {"warn": lo, "fail": hi},
        "reference": reference_csv,
        "current": current_csv,
        "columns": columns,
        "overall": overall,
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(verdict, f, indent=2, ensure_ascii=False)
    logger.info(f"Drift check {overall}: {columns}")
    return verdict


def _log_run(
    run_name: str,
    metrics: dict[str, float],
    artifacts: list[tuple[str, str]],
    run_type: str,
) -> None:
    """Idempotently log a batch/drift run to the ``raay_batch`` experiment."""
    import mlflow

    tracking_uri = mlflow_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(Experiments.BATCH.value)
    if experiment is None:
        experiment_id = mlflow.create_experiment(Experiments.BATCH.value)
    else:
        experiment_id = experiment.experiment_id
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        mlflow.set_tag("run_type", run_type)
        for key, value in metrics.items():
            mlflow.log_metric(key, float(value))
        for artifact_path, local_path in artifacts:
            mlflow.log_artifact(local_path, artifact_path=artifact_path)
        logger.info(f"Logged raay_batch run {run.info.run_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["make-input", "score", "init-reference", "drift"],
        default="score",
    )
    parser.add_argument("--date", default=None, help="YYYY-MM-DD (default today)")
    parser.add_argument("--pool", default=DefaultPaths.TEST_SPLIT.value)
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--current", default=None, help="drift current-day CSV")
    parser.add_argument("--reference", default=DefaultPaths.SCORING_REFERENCE.value)
    parser.add_argument("--drift-out", default=None)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--min-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--onnx-path", default=DefaultPaths.ONNX_INT8_MODEL.value)
    parser.add_argument("--tokenizer-dir", default=DefaultPaths.BASELINE_MODEL.value)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging.")
    parser.add_argument("--report-out", default=None)
    args = parser.parse_args()

    load_environment()
    from datetime import datetime

    day = args.date or datetime.now().astimezone().date().isoformat()
    input_csv = args.input or str(Path(DefaultPaths.SCORING_INPUT.value) / f"{day}.csv")
    output_csv = args.output or str(
        Path(DefaultPaths.SCORING_OUTPUT.value) / f"{day}.csv"
    )
    stats_json = args.report_out or f"reports/batch_score_{day}.json"
    drift_json = args.drift_out or f"reports/drift/{day}.json"
    scorer = Scorer(
        onnx_path=args.onnx_path,
        tokenizer_dir=args.tokenizer_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    if args.mode == "make-input":
        make_input(args.pool, day, args.samples, input_csv)
        if not args.no_mlflow:
            _log_run(
                f"make-input-{day}",
                {"n_reviews": float(args.samples)},
                [(f"input/{day}", input_csv)],
                "make-input",
            )
        return

    if args.mode == "init-reference":
        init_reference(args.pool, args.samples, args.reference, scorer)
        if not args.no_mlflow:
            _log_run(
                "init-reference",
                {"n_reference": float(args.samples)},
                [("reference", args.reference)],
                "init-reference",
            )
        return

    if args.mode == "score":
        stats = score_input(input_csv, output_csv, scorer, min_samples=args.min_samples)
        Path(stats_json).parent.mkdir(parents=True, exist_ok=True)
        with open(stats_json, "w") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        if not args.no_mlflow:
            _log_run(
                f"score-{day}",
                {
                    "n_reviews": float(stats["n_reviews"]),
                    "elapsed_sec": stats["elapsed_sec"],
                    "mean_predicted_score": float(
                        pd.read_csv(output_csv)["predicted_score"].mean()
                    ),
                },
                [(f"score/{day}", output_csv), ("stats_today", stats_json)],
                "score",
            )
        return

    verdict = drift_check(
        args.reference,
        args.current or output_csv,
        drift_json,
        day,
    )
    if not args.no_mlflow:
        _log_run(
            f"drift-{day}",
            {col: data["drift_score"] for col, data in verdict["columns"].items()},
            [("drift", drift_json)],
            "drift",
        )


if __name__ == "__main__":
    main()
