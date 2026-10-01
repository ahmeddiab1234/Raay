"""Quantize an exported ONNX graph to INT8, verify parity, evaluate, log to MLflow.

    uv run python -m raay.inference.quantize_onnx
    uv run python -m raay.inference.quantize_onnx \
        --onnx-path models/onnx/distilled.onnx --tokenizer-dir models/distilled/final

Emits ``models/onnx/model_int8.onnx`` (self-contained), ``reports/onnx_int8_parity.json``
and ``reports/eval_int8.json`` (same schema as ``eval_baseline.json``). The
quantization + parity math is in ``quantize_core``; this module is the run.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from loguru import logger
from onnxruntime.quantization import QuantType
from transformers import AutoConfig, AutoTokenizer

from raay.config.env import load_environment, mlflow_tracking_uri
from raay.enums.constants import DefaultPaths, Experiments, Models
from raay.inference.export_onnx import _repair_artifact_locations
from raay.inference.quantize_core import (
    _tokenizer_dir_for,
    parity_report,
    quantize_to_int8,
)
from raay.training.evaluate import (
    dialect_breakdown,
    evaluate_on_split,
    load_onnx_session,
)

warnings.filterwarnings("ignore", category=SyntaxWarning)

__all__ = ["_tokenizer_dir_for", "main", "parity_report", "quantize_to_int8"]


def _write_json(results: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-path", default="models/onnx/model.onnx")
    parser.add_argument("--output", default="models/onnx/model_int8.onnx")
    parser.add_argument("--tokenizer-dir", default=None)
    parser.add_argument("--test-file", default=DefaultPaths.TEST_SPLIT.value)
    parser.add_argument("--model-name", default=Models.TEACHER.value)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--per-channel", action="store_true", default=True)
    parser.add_argument("--eval-output", default="reports/eval_int8.json")
    parser.add_argument("--parity-output", default="reports/onnx_int8_parity.json")
    parser.add_argument("--experiment", default=Experiments.TRAINING.value)
    parser.add_argument("--tracking-uri", default=None)
    args = parser.parse_args()

    load_environment()
    tracking_uri = (
        args.tracking_uri
        if args.tracking_uri
        else mlflow_tracking_uri(default="file:./mlruns")
    )
    mlflow.set_tracking_uri(tracking_uri)
    repaired = _repair_artifact_locations(tracking_uri)
    if repaired:
        logger.info(
            f"Repointed {repaired} experiment(s) with stale /kaggle artifact "
            f"roots to the local ./mlruns store"
        )

    tokenizer_dir = args.tokenizer_dir or _tokenizer_dir_for(args.onnx_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    logger.info(f"Quantizing {args.onnx_path} -> {args.output}")
    output_path = quantize_to_int8(args.onnx_path, args.output, args.per_channel)

    parity = parity_report(args.onnx_path, output_path, tokenizer, args.max_length)
    logger.info(
        f"FP32-vs-INT8 parity: max_abs_diff={parity['max_abs_diff']:.3e} "
        f"label_agreement={parity['label_agreement']:.4f}"
    )
    _write_json(parity, args.parity_output)

    logger.info(f"Evaluating INT8 model on {args.test_file}")
    test_df = pd.read_csv(args.test_file)
    model = load_onnx_session(output_path)
    # ORT sessions have no `.config`; evaluate_on_split reads id2label from it,
    # so expose the checkpoint config to keep label decoding correct.
    model.config = AutoConfig.from_pretrained(tokenizer_dir)
    report = evaluate_on_split(
        test_df, model, tokenizer, args.model_name, args.max_length
    )
    report["dialect_breakdown"] = dialect_breakdown(
        test_df, model, tokenizer, args.model_name, args.max_length
    )
    report["metadata"] = {
        "onnx_path": args.onnx_path,
        "quantized_path": output_path,
        "backend": "onnxruntime-int8",
        "quantization": "dynamic",
        "weight_type": "QInt8",
        "per_channel": args.per_channel,
        "model_name": args.model_name,
        "max_length": args.max_length,
        "tokenizer_version": getattr(tokenizer, "vocab_size", None),
    }
    report["int8_parity"] = parity
    _write_json(report, args.eval_output)
    logger.info(
        f"INT8 test accuracy={report['accuracy']:.4f} f1_macro={report['f1_macro']:.4f}"
    )

    size_bytes = Path(output_path).stat().st_size
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=f"quantize-onnx-{Path(output_path).stem}") as run:
        mlflow.log_params(
            {
                "onnx_path": args.onnx_path,
                "output_name": Path(output_path).stem,
                "quantization": "dynamic",
                "weight_type": str(QuantType.QInt8),
                "per_channel": args.per_channel,
                "max_length": args.max_length,
                "onnx_int8_size_bytes": size_bytes,
                "onnx_fp32_size_bytes": Path(args.onnx_path).stat().st_size,
            }
        )
        mlflow.log_metrics(
            {
                "accuracy": report["accuracy"],
                "f1_macro": report["f1_macro"],
                "f1_weighted": report["f1_weighted"],
                "max_logits_abs_diff": parity["max_abs_diff"],
                "mean_logits_abs_diff": parity["mean_abs_diff"],
                "label_agreement": parity["label_agreement"],
            }
        )
        mlflow.log_artifact(output_path, artifact_path="onnx")
        logger.info(f"MLflow run {run.info.run_id} logged {output_path}")


if __name__ == "__main__":
    main()
