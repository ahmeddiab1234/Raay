"""Export a single checkpoint to ONNX and log the parity result."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import onnxruntime as ort
from loguru import logger

from raay.inference.export_config import _SAMPLE_TEXTS
from raay.inference.export_parity import (
    export_to_onnx,
    load_model,
    run_ort,
    torch_logits,
    verify_parity,
)
from raay.inference.export_preprocess import _preprocess


def _tokenize_sample_texts(tokenizer: Any, max_length: int) -> dict[str, Any]:
    texts = [_preprocess(t) for t in _SAMPLE_TEXTS]
    return tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )


def export_one(
    model_dir: str,
    name: str,
    output_dir: str,
    max_length: int,
    opset: int,
    tolerance: float,
    experiment: str,
) -> dict[str, Any]:
    onnx_path = str(Path(output_dir) / f"{name}.onnx")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading model from {model_dir}")
    model, tokenizer, id2label = load_model(model_dir)
    enc = _tokenize_sample_texts(tokenizer, max_length)

    export_to_onnx(model, enc, onnx_path, opset)

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    pt = torch_logits(model, enc)
    ort_out = run_ort(session, enc)
    parity = verify_parity(pt, ort_out, tolerance)
    logger.info(
        f"Parity [{name}]: max_abs_diff={parity['max_abs_diff']:.3e} "
        f"passed={parity['passed']}"
    )
    if not parity["passed"]:
        raise RuntimeError(f"ONNX parity check FAILED for {onnx_path}: {parity}")

    # torch's dynamo exporter writes weights to external `${name}.onnx.data`.
    externals: list[str] = []
    data_file = Path(onnx_path).with_name(f"{name}.onnx.data")
    if data_file.exists():
        externals.append(str(data_file))
    size_bytes = Path(onnx_path).stat().st_size + sum(
        Path(p).stat().st_size for p in externals
    )
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=f"export-onnx-{name}") as run:
        mlflow.log_params(
            {
                "model_dir": model_dir,
                "model_name": str(id2label or name),
                "opset": opset,
                "max_length": max_length,
                "onnx_size_bytes": size_bytes,
                "output_name": name,
                "n_sample_texts": len(_SAMPLE_TEXTS),
            }
        )
        mlflow.log_metrics(
            {
                "max_logits_abs_diff": parity["max_abs_diff"],
                "mean_logits_abs_diff": parity["mean_abs_diff"],
            }
        )
        mlflow.log_artifact(onnx_path, artifact_path="onnx")
        for ext in externals:
            mlflow.log_artifact(ext, artifact_path="onnx")
        logger.info(f"MLflow run {run.info.run_id} logged {onnx_path}")

    return {
        "model_dir": model_dir,
        "onnx_path": onnx_path,
        "opset": opset,
        "max_length": max_length,
        "onnx_size_bytes": size_bytes,
        **parity,
    }
