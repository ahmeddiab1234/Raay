"""``reports/feedback_metrics.json`` and its MLflow logging.

``model_label`` on a production review is a *free* labelled error record. Every
other Phase 6 signal is a seeded draw from ``data/processed/test.csv`` and says
so in its own ``caveat`` field; ``production_error_rate`` is the only
production-measured error rate in the project.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from raay.data.feedback_merge import label_proportions, read_merged
from raay.data.feedback_schema import (
    _PROPORTION_DP,
    ALL_STATUSES,
    ROUTE_ADJUDICATE_FIRST,
    TRAINABLE_STATUSES,
    FeedbackConfig,
    ReviewResult,
)
from raay.data.feedback_text import _text_key_only, coerce_label
from raay.enums.constants import LABELS, Experiments

_ERROR_RATE_NOTE = (
    "Measured on captured production traffic, not on a draw from "
    "data/processed/test.csv. Only interpretable if the CS tool posts "
    "confirmations as well as disputes: a confirmation (model_label == "
    "corrected_label) is the denominator, and a tool that posts only disputes "
    "makes this number uninterpretable."
)


def production_error_rate(raw: pd.DataFrame) -> dict[str, Any]:
    """Per-class share of captured reviews the model got wrong.

    ``corrections_where_model_said_c / total_where_model_said_c``.

    Returns ``None`` for a class never observed rather than 0.0, because "the
    model was never wrong about a positive" and "we never saw a positive" are
    different claims and a report that conflates them is worse than one with a
    hole in it.
    """
    empty: dict[str, Any] = {
        "note": _ERROR_RATE_NOTE,
        "by_class": {},
        "overall": None,
        "n_observed": 0,
    }
    if raw.empty or "model_label" not in raw.columns:
        return empty

    predicted = raw["model_label"].map(coerce_label)
    corrected = raw["corrected_label"].map(coerce_label)
    by_class: dict[str, Any] = {}
    total_seen = 0
    total_wrong = 0
    for label in LABELS:
        seen = predicted == label
        n_seen = int(seen.sum())
        n_wrong = int((corrected[seen] != label).sum()) if n_seen else 0
        total_seen += n_seen
        total_wrong += n_wrong
        by_class[label] = {
            "n_observed": n_seen,
            "n_wrong": n_wrong,
            "rate": round(n_wrong / n_seen, _PROPORTION_DP) if n_seen else None,
        }
    return {
        "note": _ERROR_RATE_NOTE,
        "by_class": by_class,
        "overall": round(total_wrong / total_seen, _PROPORTION_DP)
        if total_seen
        else None,
        "n_observed": total_seen,
    }


def build_report(
    raw: pd.DataFrame,
    review: ReviewResult,
    merge_summary: dict[str, Any],
    config: FeedbackConfig,
) -> dict[str, Any]:
    """Assemble ``reports/feedback_metrics.json``."""
    counts = {status: 0 for status in ALL_STATUSES}
    counts.update(review.counts)
    error_rate = production_error_rate(raw)
    n_corrections = sum(entry["n_wrong"] for entry in error_rate["by_class"].values())
    n_raw = len(raw)
    return {
        "config": {
            "min_corroborating_agents": config.min_corroborating_agents,
            "suspicious_model_score": config.suspicious_model_score,
            "max_neutral_per_batch": config.max_neutral_per_batch,
            "min_char_length": config.min_char_length,
        },
        "captured": {
            "n_raw": n_raw,
            "n_corrections": n_corrections,
            "n_confirmations": n_raw - n_corrections,
            "model_label_mix": (
                label_proportions(raw["model_label"].map(coerce_label)) if n_raw else {}
            ),
        },
        "qa": {
            "counts": counts,
            "n_trainable": int(sum(counts[status] for status in TRAINABLE_STATUSES)),
            "n_route_adjudicate_first": (
                int((review.frame["route"] == ROUTE_ADJUDICATE_FIRST).sum())
                if not review.frame.empty and "route" in review.frame.columns
                else 0
            ),
        },
        "production_error_rate": error_rate,
        "merge": merge_summary,
        "caveat": (
            "There is no deployed service or CS tool yet, so every value here is a "
            "rehearsal until real traffic arrives. Two-agent corroboration also "
            "needs a roster with at least two real agent ids: until one exists "
            "every row is `single_agent` and nothing is trainable."
        ),
    }


def write_report(report: dict[str, Any], out_json: str) -> None:
    """Write the metrics JSON with a trailing newline.

    The file is git-tracked (``cache: false``) so its md5 lands in ``dvc.lock``.
    Without the newline, pre-commit's ``end-of-file-fixer`` rewrites the file
    after the stage runs and every subsequent ``dvc repro feedback`` shows a
    spurious diff -- which ``git diff --exit-code dvc.lock`` in CI then fails on.
    """
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def log_run(mode: str, report: dict[str, Any], out_json: str) -> None:
    """Log to the ``raay_batch`` experiment, matching ``batch_score._log_run``."""
    import mlflow

    mlflow.set_experiment(Experiments.BATCH.value)
    with mlflow.start_run(run_name=f"feedback-{mode}"):
        mlflow.set_tag("run_type", f"feedback_{mode}")
        error_rate = report["production_error_rate"]
        if error_rate["overall"] is not None:
            mlflow.log_metric("feedback_production_error_rate", error_rate["overall"])
        for label, entry in error_rate["by_class"].items():
            if entry["rate"] is not None:
                mlflow.log_metric(f"feedback_error_rate_{label}", entry["rate"])
        mlflow.log_metric("feedback_n_raw", float(report["captured"]["n_raw"]))
        mlflow.log_metric(
            "feedback_n_corrections", float(report["captured"]["n_corrections"])
        )
        mlflow.log_metric(
            "feedback_n_confirmations", float(report["captured"]["n_confirmations"])
        )
        mlflow.log_metric("feedback_n_trainable", float(report["qa"]["n_trainable"]))
        mlflow.log_artifact(out_json, artifact_path="feedback")


def has_new_feedback(reviewed_csv: str, merged_csv: str) -> bool:
    """Any reviewed row that has not been merged yet.

    The Airflow predicate for skipping the nightly merge. Reading the merged file
    rather than trusting a counter is what makes a re-run after a crash correct:
    a row that landed in ``train_feedback.csv`` before the task failed is not
    re-merged, so the retry is a no-op instead of a duplicate.
    """
    if not Path(reviewed_csv).exists():
        return False
    reviewed = pd.read_csv(reviewed_csv, dtype=str, keep_default_na=False)
    if reviewed.empty or "status" not in reviewed.columns:
        return False
    trainable = reviewed[reviewed["status"].isin(TRAINABLE_STATUSES)]
    if trainable.empty:
        return False
    merged = read_merged(merged_csv)
    if merged is None or merged.empty:
        return True
    seen = {_text_key_only(t) for t in merged["text"]}
    return not bool(trainable["text"].map(_text_key_only).isin(seen).all())
