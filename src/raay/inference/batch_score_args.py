"""Argument parsing for ``python -m raay.inference.batch_score``.

Split out from the dispatcher so the flag surface can be read (and changed)
without scrolling through the mode branches. The defaults that matter:

* ``--samples`` / ``--min-samples`` default to 1000, and ``--min-samples`` is a
  hard floor enforced by ``score_input`` -- a nightly panel of 40 rows is not a
  distribution and would produce a PSI nobody should read.
* ``--min-history-days`` defaults to 7, not 3, because the rolling z-score is
  only as stable as its denominator.
"""

from __future__ import annotations

import argparse

from raay.enums.constants import DefaultPaths

MODES = ("make-input", "score", "init-reference", "drift", "predict-drift")


def build_parser(description: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    _add_mode_flags(parser)
    _add_engineering_flags(parser)
    _add_predict_drift_flags(parser)
    return parser


def _add_mode_flags(parser: argparse.ArgumentParser) -> None:
    """The mode, the panel paths, and the scoring knobs shared by all modes."""
    parser.add_argument("--mode", choices=list(MODES), default="score")
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


def _add_engineering_flags(parser: argparse.ArgumentParser) -> None:
    """Phase 6 step 1: engineered input features (embedding PCs, OOV, dialect)."""
    parser.add_argument(
        "--no-engineer",
        action="store_true",
        help=(
            "Skip the engineered input features and gate only the model outputs "
            "(predicted_label, positive). Needs no encoder weights."
        ),
    )
    parser.add_argument(
        "--pca-path",
        default=DefaultPaths.DRIFT_PCA_BASIS.value,
        help="Frozen PCA basis fitted by init-reference (never refit per day).",
    )
    parser.add_argument(
        "--reference-engineered",
        default=DefaultPaths.SCORING_REFERENCE_ENGINEERED.value,
        help="Cached engineered reference panel written by init-reference.",
    )
    parser.add_argument(
        "--engineered-current",
        default=None,
        help="Where to write the engineered current panel (default alongside it).",
    )
    parser.add_argument(
        "--n-pcs", type=int, default=10, help="PCA components kept (10-20 is sane)."
    )
    parser.add_argument(
        "--pooling", choices=["mean", "cls"], default="mean", help="Encoder pooling."
    )
    parser.add_argument(
        "--drift-columns",
        default=None,
        help="Comma-separated override of the gated columns.",
    )


def _add_predict_drift_flags(parser: argparse.ArgumentParser) -> None:
    """Phase 6 step 2: the output-side watch."""
    parser.add_argument(
        "--train-prior",
        default=DefaultPaths.TRAIN_SPLIT.value,
        help=(
            "Label prior for the Phase-6-step-2 class-distribution PSI. Read "
            "from the train split's labels rather than hardcoded, because the "
            "brief's 45/35/20 does not describe this dataset (it is "
            "57.6/37.3/5.1) and fails a clean panel at PSI 0.41."
        ),
    )
    parser.add_argument(
        "--predict-drift-out",
        default=None,
        help="Prediction-drift report path (default reports/prediction_drift).",
    )
    parser.add_argument(
        "--input-drift-report",
        default=None,
        help="Input-drift report to pair with for triage (default reports/drift).",
    )
    parser.add_argument(
        "--history-window",
        type=int,
        default=14,
        help="Days of confidence history in the rolling baseline.",
    )
    parser.add_argument(
        "--min-history-days",
        type=int,
        default=7,
        help="Days of history before the rolling z-score is reported.",
    )
