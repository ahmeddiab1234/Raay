"""CPU latency/accuracy benchmark for the ONNX service graph.

Measures the real runtime cost of one ``/predict``-shaped call over held-out
data at several ORT batch sizes and writes ``reports/serving_benchmark.json``.
This is the evidence gate for GPU/TensorRT investment: compare p50/p95/p99
latency against the product SLA — if already within budget, ship CPU INT8 on
BentoML and never build a TensorRT engine.

The timings wrap the exact code path ``serve.predict_probs`` uses
(arabet preprocess + tokenization + a single ORT ``session.run``), so the
numbers are representative of one online call at that batch size. Batch sizes
> 1 show throughput headroom if the service ever adopts request batching.

Run from the repo root:

    uv run python -m raay.serving.benchmark

Defaults: models/onnx/model_int8.onnx tokenized with models/baseline/final,
1000 samples from data/processed/test.csv, batch sizes 1/8/32.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import pandas as pd
from loguru import logger
from transformers import AutoConfig, AutoTokenizer

import mlflow
from raay.config.env import load_environment, mlflow_tracking_uri
from raay.enums.constants import DefaultPaths, Experiments, Models
from raay.serving.serve import predict_probs


def load_session(tokenizer_dir: str, onnx_path: str):
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    config = AutoConfig.from_pretrained(tokenizer_dir)
    raw = getattr(config, "id2label", None) or {}
    id2label = {int(k): v for k, v in raw.items()}
    return session, tokenizer, id2label


def benchmark_batch_size(
    session: Any,
    tokenizer: Any,
    texts: list[str],
    y_true: np.ndarray,
    label_to_id: dict[str, int],
    model_name: str,
    max_length: int,
    batch_size: int,
    runs: int,
) -> dict[str, Any]:
    """Time full passes over ``texts`` at one batch size.

    Each chunk is one tokenize+ORT call (one online call at that size). Warm
    up one chunk first, then record every chunk across ``runs`` passes.
    """
    n = len(texts)
    first = texts[:batch_size]
    predict_probs(session, tokenizer, first, model_name, max_length)

    chunk_ms: list[float] = []
    argmax_seen: list[int] = []
    for pass_idx in range(1, runs + 1):
        for i in range(0, n, batch_size):
            chunk = texts[i : i + batch_size]
            t0 = time.perf_counter()
            probs = predict_probs(session, tokenizer, chunk, model_name, max_length)
            chunk_ms.append((time.perf_counter() - t0) * 1000.0)
            if pass_idx == 1:
                argmax_seen.extend(np.argmax(probs, axis=-1).tolist())

    chunk_ms = sorted(chunk_ms)
    total_sec = sum(chunk_ms) / 1000.0
    total_items = n * runs
    per_item = [ms / batch_size for ms in chunk_ms]

    y_pred = np.array(argmax_seen[:n])
    accuracy = float(np.mean(y_pred == y_true))

    def pct(samples: list[float], p: float) -> float:
        return float(np.percentile(samples, p))

    return {
        "batch_size": batch_size,
        "n_samples": n,
        "n_calls": len(chunk_ms),
        "n_runs": runs,
        "chunk_latency_ms": {
            "mean": float(statistics.mean(chunk_ms)),
            "p50": pct(chunk_ms, 50),
            "p95": pct(chunk_ms, 95),
            "p99": pct(chunk_ms, 99),
        },
        "latency_ms_per_req_equivalent": {
            "mean": float(statistics.mean(per_item)),
            "p50": pct(per_item, 50),
            "p95": pct(per_item, 95),
            "p99": pct(per_item, 99),
        },
        "throughput_req_per_s": float(total_items / total_sec),
        "accuracy": accuracy,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-path", default=DefaultPaths.ONNX_INT8_MODEL.value)
    parser.add_argument("--tokenizer-dir", default=DefaultPaths.BASELINE_MODEL.value)
    parser.add_argument("--test-file", default=DefaultPaths.TEST_SPLIT.value)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--model-name", default=Models.TEACHER.value)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument(
        "--batch-sizes",
        default="1,8,32",
        help="Comma-separated ORT batch sizes to benchmark.",
    )
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--output", default=DefaultPaths.SERVING_BENCHMARK.value)
    parser.add_argument("--experiment", default=Experiments.TRAINING.value)
    parser.add_argument("--tracking-uri", default=None)
    args = parser.parse_args()

    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    load_environment()
    tracking_uri = (
        args.tracking_uri
        if args.tracking_uri
        else mlflow_tracking_uri(default="file:./mlruns")
    )
    mlflow.set_tracking_uri(tracking_uri)

    logger.info(f"Loading test data from {args.test_file}")
    df = pd.read_csv(args.test_file).head(args.samples)

    logger.info(f"Loading session {args.onnx_path} with tokenizer {args.tokenizer_dir}")
    session, tokenizer, id2label = load_session(args.tokenizer_dir, args.onnx_path)
    label_to_id = {v: k for k, v in id2label.items()}
    y_true = np.array([label_to_id[lab] for lab in df["label"]])
    texts = df["text"].tolist()

    results: dict[int, dict[str, Any]] = {}
    for batch_size in batch_sizes:
        logger.info(f"Benchmarking batch_size={batch_size} over {len(texts)} samples")
        results[batch_size] = benchmark_batch_size(
            session,
            tokenizer,
            texts,
            y_true,
            label_to_id,
            args.model_name,
            args.max_length,
            batch_size,
            args.runs,
        )
        r = results[batch_size]
        logger.info(
            f"  batch={batch_size}: p50={r['chunk_latency_ms']['p50']:.1f}ms "
            f"p95={r['chunk_latency_ms']['p95']:.1f}ms "
            f"p99={r['chunk_latency_ms']['p99']:.1f}ms "
            f"acc={r['accuracy']:.4f}"
        )

    report = {
        "metadata": {
            "onnx_path": args.onnx_path,
            "tokenizer_dir": args.tokenizer_dir,
            "backend": "onnxruntime-cpu",
            "model_name": args.model_name,
            "max_length": args.max_length,
            "n_samples": len(texts),
            "runs": args.runs,
        },
        "batch_sizes": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Wrote benchmark report: {args.output}")

    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name="benchmark-serving-cpu") as run:
        mlflow.log_params(
            {
                "onnx_path": args.onnx_path,
                "tokenizer_dir": args.tokenizer_dir,
                "batch_sizes": args.batch_sizes,
                "runs": args.runs,
                "n_samples": len(texts),
                "max_length": args.max_length,
            }
        )
        for batch_size, r in results.items():
            mlflow.log_metrics(
                {
                    f"batch{batch_size}_p50_ms": r["chunk_latency_ms"]["p50"],
                    f"batch{batch_size}_p95_ms": r["chunk_latency_ms"]["p95"],
                    f"batch{batch_size}_p99_ms": r["chunk_latency_ms"]["p99"],
                    f"batch{batch_size}_throughput_req_per_s": r[
                        "throughput_req_per_s"
                    ],
                    f"batch{batch_size}_accuracy": r["accuracy"],
                }
            )
        mlflow.log_artifact(args.output)
        logger.info(f"MLflow run {run.info.run_id} logged benchmark to {args.output}")


if __name__ == "__main__":
    main()
