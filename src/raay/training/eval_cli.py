"""CLI for the frozen-split evaluation harness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from loguru import logger
from transformers import AutoTokenizer

from raay.config.env import load_environment, mlflow_tracking_uri
from raay.enums.constants import DefaultPaths, Experiments, Models
from raay.training.eval_metrics import (
    dialect_breakdown,
    evaluate_on_split,
    load_model,
    load_onnx_session,
)
from raay.training.eval_plot import plot_mlflow_comparison

_DESCRIPTION = (
    "Evaluate a checkpoint (or an exported ONNX graph) on the frozen test split."
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=_DESCRIPTION)
    parser.add_argument("--model-dir", default=DefaultPaths.BASELINE_MODEL.value)
    parser.add_argument(
        "--onnx-path",
        default=None,
        help=(
            "If set, run inference through this exported ONNX model via "
            "onnxruntime instead of the PyTorch checkpoint in --model-dir."
        ),
    )
    parser.add_argument("--test-file", default=DefaultPaths.TEST_SPLIT.value)
    parser.add_argument("--model-name", default=Models.TEACHER.value)
    parser.add_argument("--output", default=DefaultPaths.EVAL_BASELINE.value)
    parser.add_argument("--experiment", default=Experiments.TRAINING.value)
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--comparison-plot", default=None)
    return parser


def _load_backend(
    args: argparse.Namespace, metadata: dict[str, Any]
) -> tuple[Any, Any]:
    """Return the (model, tokenizer) pair plus the metadata describing it."""
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if args.onnx_path:
        logger.info(f"Loading ONNX model from {args.onnx_path}")
        metadata["onnx_path"] = args.onnx_path
        metadata["backend"] = "onnxruntime"
        metadata["tokenizer_version"] = getattr(tokenizer, "vocab_size", None)
        return load_onnx_session(args.onnx_path), tokenizer

    logger.info(f"Loading model from {args.model_dir}")
    model, tokenizer, _ = load_model(args.model_dir)
    metadata["tokenizer_version"] = getattr(tokenizer, "vocab_size", None)
    return model, tokenizer


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    load_environment()
    tracking_uri = (
        args.tracking_uri
        if args.tracking_uri
        else mlflow_tracking_uri(default="file:./mlruns")
    )
    mlflow.set_tracking_uri(tracking_uri)

    logger.info(f"Loading test data from {args.test_file}")
    test_df = pd.read_csv(args.test_file)

    metadata: dict[str, Any] = {
        "model_dir": args.model_dir,
        "model_name": args.model_name,
        "max_length": args.max_length,
        "tokenizer_version": None,
    }
    model, tokenizer = _load_backend(args, metadata)

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
