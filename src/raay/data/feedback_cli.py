"""CLI for the feedback loop: ``--mode review`` and ``--mode merge``."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from raay.config.env import load_environment
from raay.data.feedback_merge import build_merged, merge_reviewed, read_merged
from raay.data.feedback_metrics import (
    build_report,
    log_run,
    write_report,
)
from raay.data.feedback_review import read_raw, review_rows, write_reviewed
from raay.data.feedback_schema import (
    ALL_STATUSES,
    FeedbackConfig,
    ReviewResult,
)
from raay.enums.constants import DefaultPaths


def load_params(path: str) -> dict[str, Any]:
    import yaml

    if not Path(path).exists():
        logger.warning(f"No {path}; using FeedbackConfig defaults.")
        return {}
    with open(path) as handle:
        return yaml.safe_load(handle) or {}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Customer-service feedback QA and merge"
    )
    parser.add_argument("--mode", choices=["review", "merge"], default="merge")
    parser.add_argument("--raw", default=DefaultPaths.FEEDBACK_RAW.value)
    parser.add_argument("--reviewed", default=DefaultPaths.FEEDBACK_REVIEWED.value)
    parser.add_argument("--merged", default=DefaultPaths.FEEDBACK_MERGED.value)
    parser.add_argument("--test-split", default=DefaultPaths.TEST_SPLIT.value)
    parser.add_argument("--metrics-out", default=DefaultPaths.FEEDBACK_METRICS.value)
    parser.add_argument("--params", default=DefaultPaths.PARAMS.value)
    parser.add_argument("--dedup-threshold", type=float, default=None)
    parser.add_argument("--max-neutral", type=int, default=None)
    parser.add_argument("--no-mlflow", action="store_true")
    return parser.parse_args(argv)


def _run_review(
    args: argparse.Namespace, config: FeedbackConfig, threshold: float, elongation: int
) -> ReviewResult:
    """Raw assertions -> the reviewed file. The operator-facing half.

    Reads the append-only capture directory and rewrites ``reviewed/overrides.csv``
    with a status per row. It also marks rows an earlier run already merged, so a
    re-review cannot resurrect them as fresh candidates.
    """
    raw = read_raw(args.raw)
    existing = read_merged(args.merged)
    review = review_rows(raw, config, args.test_split, threshold, existing, elongation)
    write_reviewed(review, args.reviewed)
    return review


def _review_for_report(raw: pd.DataFrame, config: FeedbackConfig) -> ReviewResult:
    """Re-derive statuses from already-reviewed rows.

    ``--mode merge`` must not re-review (that would re-run the ladder, and the
    reviewed file's statuses are the operator-approved truth). This only rebuilds
    the ``ReviewResult`` envelope so ``build_report`` has status counts to report,
    reading the statuses straight off the rows rather than recomputing them -- a
    recompute here could disagree with the file DVC just hashed.
    """
    if raw.empty or "status" not in raw.columns:
        return ReviewResult(frame=pd.DataFrame(), counts={})
    counts = {status: int((raw["status"] == status).sum()) for status in ALL_STATUSES}
    return ReviewResult(frame=raw, counts=counts)


def _read_reviewed_for_report(reviewed_csv: str) -> pd.DataFrame:
    """The reviewed file, for the error-rate block in ``--mode merge``."""
    if not Path(reviewed_csv).exists():
        return pd.DataFrame()
    return pd.read_csv(reviewed_csv, dtype=str, keep_default_na=False)


def _resolve_thresholds(
    args: argparse.Namespace, config: FeedbackConfig, params: dict[str, Any]
) -> tuple[float, int]:
    if args.max_neutral is not None:
        config.max_neutral_per_batch = args.max_neutral
    preprocessing = params.get("preprocessing", {})
    threshold = (
        args.dedup_threshold
        if args.dedup_threshold is not None
        else float(preprocessing.get("dedup_similarity_threshold", 0.9))
    )
    elongation = int(preprocessing.get("elongation_max_repeat", 2))
    return threshold, elongation


def main(argv: list[str] | None = None) -> int:
    load_environment()
    args = parse_args(argv)
    params = load_params(args.params)
    config = FeedbackConfig.from_params(params)
    threshold, elongation = _resolve_thresholds(args, config, params)

    # The two modes read *different* things and must not share a path.
    #
    # `--mode merge` is a DVC stage whose only declared dependency is
    # `reviewed/overrides.csv`. If it also read `data/feedback/raw/` it would be
    # unreproducible from its own deps (a runner has no raw files, since they are
    # git-ignored), and worse, it would *rewrite* the reviewed file -- destroying
    # every `adjudicator_id` / `adjudicated_label` the operator typed in by hand
    # after `--mode review`, silently demoting every adjudication back to
    # `single_agent`. So merge consumes the reviewed artifact as-is.
    #
    # The reviewed file is the human-in-the-loop boundary: review writes it, a
    # person edits it, DVC hashes it, merge consumes it.
    if args.mode == "review":
        review = _run_review(args, config, threshold, elongation)
        # Report what *would* be trainable without touching data/processed/. That
        # is the mode an operator runs before `dvc add`.
        _, summary = build_merged(review.frame, config)
        existing = read_merged(args.merged)
        summary["merged_path"] = None
        summary["cumulative_rows"] = 0 if existing is None else len(existing)
        raw = read_raw(args.raw)
        logger.info(
            f"Review only: {review.counts}; {summary['n_accepted']} would be trainable"
        )
    else:
        summary = merge_reviewed(args.reviewed, args.merged, config)
        logger.info(
            f"Merged {summary['n_accepted']} rows into {args.merged} "
            f"(collapsed {summary['n_collapsed_to_one_row_per_review']} "
            f"per-review duplicates, cap suppressed "
            f"{summary['n_suppressed_by_cap']} neutral)"
        )
        # The merge has no business reading raw captures, so the error-rate block
        # is built from the reviewed file -- the same assertions the merge saw.
        raw = _read_reviewed_for_report(args.reviewed)

    report = build_report(
        raw,
        review if args.mode == "review" else _review_for_report(raw, config),
        summary,
        config,
    )
    write_report(report, args.metrics_out)
    if not args.no_mlflow:
        log_run(args.mode, report, args.metrics_out)
    return 0
