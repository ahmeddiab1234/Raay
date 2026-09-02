"""Evaluation harness for the fine-tuned AraBERT baseline.

Produces ``reports/eval_baseline.json`` with:
    - overall accuracy / macro-F1 / weighted-F1,
    - per-class (positive/negative/neutral) precision, recall, F1,
    - the raw confusion matrix (JSON-serializable),
    - a per-dialect stratified breakdown (accuracy + macro-F1 per dialect),
    - the label -> id mapping and model metadata.

Run from the repo root:

    uv run python -m raay.training.evaluate \
        --model-dir models/baseline/final \
        --test-file data/processed/test.csv \
        --output reports/eval_baseline.json \
        --experiment raay_training

Also writes ``reports/mlflow_comparison.png`` (a bar chart comparing f1_macro
across the ``raay_training`` MLflow runs) when ``--comparison-plot`` is set.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import mlflow
from raay.data.dialect import add_dialect_column

warnings.filterwarnings("ignore", category=SyntaxWarning)

try:
    from arabert.preprocess import ArabertPreprocessor
except ImportError:  # pragma: no cover - import path guard
    ArabertPreprocessor = None


def _preprocess(text: str, model_name: str) -> str:
    if ArabertPreprocessor is not None:
        return ArabertPreprocessor(model_name=model_name).preprocess(text)
    return str(text)


def load_model(model_dir: str):
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    id2label = getattr(model.config, "id2label", None)
    return model, tokenizer, id2label


def predict(
    model: Any, tokenizer: Any, texts: list[str], model_name: str, max_length: int
) -> np.ndarray:
    probs = model(
        **tokenizer(
            [_preprocess(t, model_name) for t in texts],
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
    ).logits
    return np.argmax(probs.detach().numpy(), axis=-1)


def _per_class(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list[int], names: list[str]
) -> dict[str, dict]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    precision, recall, f1 = (
        np.atleast_1d(precision),
        np.atleast_1d(recall),
        np.atleast_1d(f1),
    )
    support = (
        np.atleast_1d(support)
        if support is not None
        else np.zeros_like(precision, dtype=int)
    )
    result = {}
    for i, name in enumerate(names):
        result[name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
    return result


def evaluate_on_split(
    df: pd.DataFrame,
    model: Any,
    tokenizer: Any,
    model_name: str,
    max_length: int,
) -> dict[str, Any]:
    id2label = getattr(model.config, "id2label", None)
    raw_labels = df["label"].tolist()
    # Map string labels to indices via id2label if present, else textual order.
    if id2label and raw_labels and isinstance(raw_labels[0], str):
        id2label = {int(k): v for k, v in id2label.items()}
        label_to_id = {v: k for k, v in id2label.items()}
        y_true = np.array([label_to_id[lab] for lab in raw_labels])
        label_names = [id2label[i] for i in sorted(id2label)]
    else:
        labels_sorted = sorted(set(raw_labels))
        label_to_id = {lab: i for i, lab in enumerate(labels_sorted)}
        y_true = np.array([label_to_id[lab] for lab in raw_labels])
        label_names = labels_sorted
    label_ids = list(range(len(label_names)))

    y_pred = predict(model, tokenizer, df["text"].tolist(), model_name, max_length)

    cm = confusion_matrix(y_true, y_pred, labels=label_ids)
    per_class = _per_class(y_true, y_pred, label_ids, label_names)
    accuracy = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    report = {
        "label_names": label_names,
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "sample_size": len(df),
    }
    return report


def dialect_breakdown(
    df: pd.DataFrame,
    model: Any,
    tokenizer: Any,
    model_name: str,
    max_length: int,
) -> dict[str, dict[str, float]]:
    if "dialect" not in df.columns:
        df = add_dialect_column(df, text_col="text")

    breakdown: dict[str, dict[str, float]] = {}
    for dialect, group in df.groupby("dialect"):
        group = group.copy()
        if group["label"].nunique() < 2 or len(group) < 2:
            # Degenerate slice: report count only.
            breakdown[str(dialect)] = {"count": len(group)}
            continue
        rep = evaluate_on_split(group, model, tokenizer, model_name, max_length)
        breakdown[str(dialect)] = {
            "count": len(group),
            "accuracy": rep["accuracy"],
            "f1_macro": rep["f1_macro"],
        }
    return breakdown


def plot_mlflow_comparison(
    experiment_name: str,
    output_path: str,
    tracking_uri: str | None = None,
) -> None:
    """Bar chart of f1_macro per MLflow run in the training experiment."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        logger.warning(f"Experiment {experiment_name!r} not found; skipping plot.")
        return
    runs = client.search_runs(experiment_ids=[exp.experiment_id])
    if not runs:
        logger.warning(f"No runs found for {experiment_name!r}; skipping plot.")
        return

    labels: list[str] = []
    f1s: list[float] = []
    for run in runs:
        metrics = run.data.metrics
        if "f1_macro" not in metrics and "eval_f1_macro" not in metrics:
            continue
        key = "f1_macro" if "f1_macro" in metrics else "eval_f1_macro"
        f1s.append(float(metrics[key]))
        params = run.data.params
        lr = params.get("learning_rate", "?")
        bs = params.get("batch_size", "?")
        labels.append(
            f"{run.data.tags.get('mlflow.runName', run.info.run_id[:8])}\nlr={lr} bs={bs}"
        )

    if not f1s:
        logger.warning("No runs had f1_macro metrics; skipping plot.")
        return

    plt.figure(figsize=(max(8, 0.9 * len(labels)), 5))
    bars = plt.bar(range(len(f1s)), f1s, color="#4C72B0")
    best = int(np.argmax(f1s))
    bars[best].set_color("#55A868")
    plt.xticks(range(len(f1s)), labels, rotation=0)
    plt.ylabel("f1_macro")
    plt.title(f"MLflow run comparison — {experiment_name}")
    plt.ylim(0, 1.0)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120)
    plt.close()
    logger.info(f"Wrote comparison plot: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="models/baseline/final")
    parser.add_argument("--test-file", default="data/processed/test.csv")
    parser.add_argument("--model-name", default="aubmindlab/bert-base-arabertv02")
    parser.add_argument("--output", default="reports/eval_baseline.json")
    parser.add_argument("--experiment", default="raay_training")
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--comparison-plot", default=None)
    args = parser.parse_args()

    os.environ.setdefault("MLFLOW_TRACKING_URI", args.tracking_uri or "file:./mlruns")
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
    if tracking_uri.startswith("file:"):
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

    logger.info(f"Loading test data from {args.test_file}")
    test_df = pd.read_csv(args.test_file)

    logger.info(f"Loading model from {args.model_dir}")
    model, tokenizer, _ = load_model(args.model_dir)

    metadata = {
        "model_dir": args.model_dir,
        "model_name": args.model_name,
        "max_length": args.max_length,
        "tokenizer_version": getattr(tokenizer, "vocab_size", None),
    }

    report = evaluate_on_split(
        test_df, model, tokenizer, args.model_name, args.max_length
    )
    report["dialect_breakdown"] = dialect_breakdown(
        test_df, model, tokenizer, args.model_name, args.max_length
    )
    report["metadata"] = metadata

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Wrote evaluation report: {args.output}")
    logger.info(
        f"Test accuracy={report['accuracy']:.4f} f1_macro={report['f1_macro']:.4f}"
    )

    if args.comparison_plot:
        plot_mlflow_comparison(
            args.experiment, args.comparison_plot, tracking_uri=args.tracking_uri
        )


if __name__ == "__main__":
    main()
