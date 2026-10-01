"""Nightly batch re-scoring job with an Evidently PSI drift check.

Phase 5 step 3: a day's worth of reviews is scored offline in LARGE batches
directly against the INT8 ONNX graph (``serve.predict_probs``, in-process --
no REST call), the predictions are written to ``data/scoring/output/`` and an
Evidently PSI drift check is run against a scored reference set. Airflow
orchestrates the steps nightly (see ``airflow/dags/raay_nightly_batch_scoring.py``).

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
  of the pool into ``data/scoring/reference/reference.csv``. It then writes two
  more artifacts: ``reference_engineered.csv`` (the reference plus the Phase-6
  engineered input features) and ``pca_basis.joblib`` (the PCA projection
  frozen on the reference embeddings).
- ``drift``: Evidently PSI (``ColumnDriftMetric`` + ``psi_stat_test``) on the
  current day vs the reference; each column is PASS (<0.1) / WARN (<0.2) /
  FAIL (>=0.2) and the verdicts are written to ``reports/drift/{date}.json``.
- ``predict-drift``: Phase 6 step 2 -- watches the **outputs** rather than the
  inputs (see :mod:`raay.inference.prediction_drift`): the predicted class mix
  PSI'd against the training label prior, mean confidence as a series, and a
  triage classification pairing this verdict with the same day's input-drift
  verdict. Writes ``reports/prediction_drift/{date}.json``.

Phase 6 step 1 extended the drift columns from the model's own outputs
(``predicted_label``, ``positive``) to the **inputs** as well: AraBERT
embedding principal components, the ``[UNK]`` rate, the dialect mix, review
length and confidence. Those come from ``raay.inference.drift_features`` and
need the fine-tuned encoder, so they are opt-out via ``--no-engineer``.

Phase 6 step 2 splits the two directions apart. ``drift`` is the input side;
``predict-drift`` is the output side and ends in a triage verdict, because the
useful question when something moves is *which half moved* -- a class-mix shift
and a world change look identical from the outputs alone.

Layout
------
``batch_score_args``  the parser
``batch_scoring``     make-input / score / the Scorer
``batch_reference``   init-reference + its engineered/PCA artifacts
``batch_modes``       one function per mode, MLflow metrics included
``batch_tracking``    the ``raay_batch`` logging wrapper

This module is the shell: parse, wire the scorer + engineered spec, dispatch.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from raay.config.env import load_environment
from raay.enums.constants import DefaultPaths
from raay.inference.batch_modes import (
    run_drift,
    run_init_reference,
    run_predict_drift,
    run_score,
    write_json,
)
from raay.inference.batch_reference import make_input
from raay.inference.batch_score_args import build_parser
from raay.inference.batch_scoring import Scorer
from raay.inference.batch_tracking import _log_run
from raay.inference.builder import DriftFeatureBuilder
from raay.inference.drift_engine import EngineeredDriftSpec


def _engineered_spec(
    args: argparse.Namespace, day: str, output_csv: str
) -> EngineeredDriftSpec | None:
    """Build the engineered-feature spec, or ``None`` when not needed.

    The encoder is loaded for exactly two modes. ``make-input`` samples a CSV
    and ``score`` runs the INT8 graph; neither has any use for it, and loading
    it costs ~2 GB of resident weights plus a few seconds that the nightly
    Airflow DAG would pay twice per day.
    """
    if args.mode not in ("init-reference", "drift") or args.no_engineer:
        return None
    current_csv = args.current or output_csv
    stem = Path(current_csv)
    return EngineeredDriftSpec(
        builder=DriftFeatureBuilder.from_config(
            tokenizer_dir=args.tokenizer_dir,
            n_components=args.n_pcs,
            max_length=args.max_length,
            pooling=args.pooling,
            batch_size=args.batch_size,
        ),
        pca_path=args.pca_path,
        reference_engineered=args.reference_engineered,
        current_engineered=args.engineered_current
        or str(stem.with_name(stem.stem + "_engineered.csv")),
    )


def _resolve_paths(args: argparse.Namespace, day: str) -> dict[str, str]:
    return {
        "input": args.input
        or str(Path(DefaultPaths.SCORING_INPUT.value) / f"{day}.csv"),
        "output": args.output
        or str(Path(DefaultPaths.SCORING_OUTPUT.value) / f"{day}.csv"),
        "stats": args.report_out or f"reports/batch_score_{day}.json",
        "drift": args.drift_out or f"reports/drift/{day}.json",
        "predict_drift": args.predict_drift_out
        or str(Path(DefaultPaths.PREDICTION_DRIFT_REPORTS.value) / f"{day}.json"),
    }


def main() -> None:
    args = build_parser(__doc__).parse_args()
    load_environment()

    day = args.date or datetime.now().astimezone().date().isoformat()
    paths = _resolve_paths(args, day)
    scorer = Scorer(
        onnx_path=args.onnx_path,
        tokenizer_dir=args.tokenizer_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    engineered = _engineered_spec(args, day, paths["output"])

    if args.mode == "make-input":
        make_input(args.pool, day, args.samples, paths["input"])
        if not args.no_mlflow:
            _log_run(
                f"make-input-{day}",
                {"n_reviews": float(args.samples)},
                [(f"input/{day}", paths["input"])],
                "make-input",
            )
        return

    if args.mode == "init-reference":
        metrics, artifacts = run_init_reference(args, scorer, engineered)
        if not args.no_mlflow:
            _log_run("init-reference", metrics, artifacts, "init-reference")
        return

    if args.mode == "score":
        metrics, artifacts = run_score(args, day, paths, scorer)
        if not args.no_mlflow:
            _log_run(f"score-{day}", metrics, artifacts, "score")
        return

    if args.mode == "predict-drift":
        metrics, artifacts = run_predict_drift(args, day, paths)
        if not args.no_mlflow:
            _log_run(f"predict-drift-{day}", metrics, artifacts, "predict-drift")
        return

    metrics, artifacts = run_drift(args, day, paths, engineered)
    if not args.no_mlflow:
        _log_run(f"drift-{day}", metrics, artifacts, "drift")


__all__ = ["main", "write_json"]


if __name__ == "__main__":
    main()
