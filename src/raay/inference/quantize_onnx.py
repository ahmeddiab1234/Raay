"""Quantize an exported ONNX model to INT8 and re-evaluate it.

Phase 3 (compression/optimization) step 4: apply dynamic quantization
(``onnxruntime.quantization.quantize_dynamic`` — the simplest, most reliable
option for transformers) to a frozen ONNX graph and re-run the held-out
evaluation through the quantized model.

Run from the repo root:

    uv run python -m raay.inference.quantize_onnx

Quantizes ``models/onnx/model.onnx`` -> ``models/onnx/model_int8.onnx``:

    - verifies the INT8 graph still runs and compares its logits against the
      FP32 graph on a fixed set of Arabic reviews (writes
      ``reports/onnx_int8_parity.json``),
    - re-evaluates on ``data/processed/test.csv`` through the INT8 model,
      reusing the exact ``evaluate.py`` metrics so ``reports/eval_int8.json``
      has the same schema as ``eval_baseline.json`` / ``eval_distilled.json``,
    - logs the quantized artifact + parity/eval metrics to ``raay_training``.

Point ``--onnx-path`` at ``models/onnx/distilled.onnx`` (+ ``--tokenizer-dir
models/distilled/final``) to quantize the student instead.

Requires deps: onnx, onnxruntime, onnxscript.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
from loguru import logger
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoConfig, AutoTokenizer

import mlflow
from raay.config.env import load_environment, mlflow_tracking_uri
from raay.enums.constants import DefaultPaths, Experiments, Models
from raay.inference.export_onnx import _preprocess, _repair_artifact_locations
from raay.training.evaluate import (
    dialect_breakdown,
    evaluate_on_split,
    load_onnx_session,
)

warnings.filterwarnings("ignore", category=SyntaxWarning)

_SAMPLE_TEXTS: tuple[str, ...] = (
    "هذا المنتج ممتاز والجودة عالية جدا",
    "المنتج وصل متأخر والجودة رديئة",
    "الطلبية وصلت بسرعة والحاجة تمام جدا شكرا",
    "حسبي الله ونعم الوكيل ياخي الجودة خايسة",
    "المنتج محايد شكله عادي",
    "الخدمة ممتازة وسعر مناسب لكن التوصيل بطيء",
)

_TOKENIZER_DIRS = {
    "model": DefaultPaths.BASELINE_MODEL.value,
    "distilled": DefaultPaths.DISTILLED_MODEL.value,
}


def _tokenizer_dir_for(onnx_path: str) -> str:
    return _TOKENIZER_DIRS.get(
        Path(onnx_path).stem, str(Path(onnx_path).parent.parent / "final")
    )


def _encode(tokenizer: Any, texts: list[str], max_length: int) -> dict[str, Any]:
    enc = tokenizer(
        [_preprocess(t) for t in texts],
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {key: enc[key] for key in ("input_ids", "attention_mask")}


def _run(session: ort.InferenceSession, enc: dict[str, Any]) -> np.ndarray:
    return session.run(
        ["logits"],
        {key: value.numpy() for key, value in enc.items()},
    )[0]


def quantize_to_int8(onnx_path: str, output_path: str, per_channel: bool = True) -> str:
    """Dynamically quantize ``onnx_path`` to INT8 at ``output_path``.

    The torch dynamo exporter leaves stale ``value_info`` whose declared shapes
    conflict with onnx shape inference; onnxruntime's quantizer re-runs a
    strict file-based inference and raises on those mismatches, and its
    external-weights churn breaks across temp dirs. We therefore (a) inline the
    weights into a self-contained FP32 graph with the conflicting ``value_info``
    dropped so fresh inference cannot clash, and (b) quantize that. The
    quantized output is likewise self-contained (INT8 weights inlined)."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    model = onnx.load(onnx_path)
    del model.graph.value_info[:]
    embedded = str(Path(output_path).with_name(f"{Path(onnx_path).stem}.fp32.onnx"))
    onnx.save_model(model, embedded, save_as_external_data=False)
    try:
        quantize_dynamic(
            model_input=embedded,
            model_output=output_path,
            weight_type=QuantType.QInt8,
            per_channel=per_channel,
            reduce_range=False,
        )
    finally:
        Path(embedded).unlink(missing_ok=True)
    logger.info(f"Quantized {onnx_path} -> {output_path}")
    return output_path


def parity_report(
    fp32_path: str, int8_path: str, tokenizer: Any, max_length: int
) -> dict[str, Any]:
    """Compare FP32 vs INT8 ONNX logits on a fixed sample set."""
    fp32 = load_onnx_session(fp32_path)
    int8 = load_onnx_session(int8_path)
    enc = _encode(tokenizer, list(_SAMPLE_TEXTS), max_length)
    fp32_out = _run(fp32, enc)
    int8_out = _run(int8, enc)
    diff = np.abs(fp32_out - int8_out)
    return {
        "n_samples": len(_SAMPLE_TEXTS),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "label_agreement": float(
            np.mean(np.argmax(fp32_out, axis=-1) == np.argmax(int8_out, axis=-1))
        ),
    }


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
