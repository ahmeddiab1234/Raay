"""Model-variant benchmark table generator (Phase 3 step 6).

Loads every trained/served model variant and assembles the README-ready
comparison table: accuracy + F1-macro from the generated eval JSONs, p50/p95
latency at batch=1 and batch=32 (measured fresh over the same
preprocess -> tokenize -> run path the BentoML service uses), and model size
on disk.

Writes ``reports/benchmark_table.md`` and ``reports/benchmark_table.csv``.
Run from the repo root:

    uv run python scripts/benchmark.py [--samples 300]

Variant rows:
    - baseline-torch  PyTorch checkpoint, models/baseline/final
    - distilled-torch PyTorch checkpoint, models/distilled/final
    - onnx-fp32       exported FP32 graph (+ external weights)
    - onnx-int8       dynamically quantized INT8 graph (self-contained)

Accuracy for ``onnx-fp32`` is inherited from ``eval_baseline.json`` because the
graph holds the identical weights; a ``note`` column marks that.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from raay.enums.constants import DefaultPaths, Models

warnings.filterwarnings("ignore", category=SyntaxWarning)

try:
    from arabert.preprocess import ArabertPreprocessor
except ImportError:  # pragma: no cover - import path guard
    ArabertPreprocessor = None

_PREPROCESSORS: dict[str, Any] = {}


def _preprocess(text: str, model_name: str) -> str:
    if ArabertPreprocessor is None:
        return str(text)
    proc = _PREPROCESSORS.get(model_name)
    if proc is None:
        proc = _PREPROCESSORS.setdefault(
            model_name, ArabertPreprocessor(model_name=model_name)
        )
    return proc.preprocess(text)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


@dataclass
class Variant:
    name: str
    kind: str
    eval_json: str
    path: str
    tokenizer_dir: str
    model_name: str
    extra_weights: tuple[str, ...] = ()
    note: str = ""


_BASELINE = DefaultPaths.BASELINE_MODEL.value
_DISTILLED = DefaultPaths.DISTILLED_MODEL.value
_TEACHER = Models.TEACHER.value

VARIANTS: tuple[Variant, ...] = (
    Variant(
        name="baseline-torch",
        kind="torch",
        eval_json=DefaultPaths.EVAL_BASELINE.value,
        path=_BASELINE,
        tokenizer_dir=_BASELINE,
        model_name=_TEACHER,
        note="FP32 PyTorch checkpoint",
    ),
    Variant(
        name="distilled-torch",
        kind="torch",
        eval_json=DefaultPaths.EVAL_DISTILLED.value,
        path=_DISTILLED,
        tokenizer_dir=_DISTILLED,
        model_name=_TEACHER,
        note="FP32 distilled PyTorch checkpoint",
    ),
    Variant(
        name="onnx-fp32",
        kind="ort",
        eval_json=DefaultPaths.EVAL_BASELINE.value,
        path=DefaultPaths.ONNX_MODEL.value,
        tokenizer_dir=_BASELINE,
        model_name=_TEACHER,
        extra_weights=("models/onnx/model.onnx.data",),
        note="Accuracy/F1 inherited from baseline-torch (identical weights)",
    ),
    Variant(
        name="onnx-int8",
        kind="ort",
        eval_json=DefaultPaths.EVAL_INT8.value,
        path=DefaultPaths.ONNX_INT8_MODEL.value,
        tokenizer_dir=_BASELINE,
        model_name=_TEACHER,
        note="Dynamic INT8 quantization (self-contained)",
    ),
)


def _checkpoint_size_mb(dir_path: str) -> float:
    return sum(p.stat().st_size for p in Path(dir_path).rglob("*") if p.is_file()) / 1e6


def _onnx_size_mb(graph_path: str, extra_weights: tuple[str, ...]) -> float:
    return (
        Path(graph_path).stat().st_size
        + sum(Path(w).stat().st_size for w in extra_weights)
    ) / 1e6


def _load_variant(variant: Variant) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(variant.tokenizer_dir)
    if variant.kind == "torch":
        model = AutoModelForSequenceClassification.from_pretrained(variant.path)
        return model, tokenizer
    session = ort.InferenceSession(variant.path, providers=["CPUExecutionProvider"])
    return session, tokenizer


def _chunk_probs(
    model: Any,
    tokenizer: Any,
    batch: list[str],
    model_name: str,
    max_length: int,
) -> np.ndarray:
    processed = [_preprocess(text, model_name) for text in batch]
    enc = tokenizer(
        processed,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    if isinstance(model, ort.InferenceSession):
        logits = model.run(
            ["logits"],
            {
                "input_ids": enc["input_ids"].numpy(),
                "attention_mask": enc["attention_mask"].numpy(),
            },
        )[0]
    else:
        with torch.no_grad():
            logits = model(**enc).logits.detach().numpy()
    return _softmax(np.asarray(logits))


def _timed_chunks(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    model_name: str,
    max_length: int,
    batch_size: int,
    runs: int,
) -> list[float]:
    _chunk_probs(model, tokenizer, texts[:batch_size], model_name, max_length)
    chunk_ms: list[float] = []
    for _ in range(runs):
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            t0 = time.perf_counter()
            _chunk_probs(model, tokenizer, chunk, model_name, max_length)
            chunk_ms.append((time.perf_counter() - t0) * 1000.0)
    return sorted(chunk_ms)


def measure_latency(
    variant: Variant,
    texts: list[str],
    batch_sizes: list[int],
    max_length: int,
    runs: int,
) -> dict[str, float]:
    model, tokenizer = _load_variant(variant)
    result: dict[str, float] = {}
    for batch_size in batch_sizes:
        samples = _timed_chunks(
            model, tokenizer, texts, variant.model_name, max_length, batch_size, runs
        )
        result[f"batch{batch_size}_p50_ms"] = float(np.percentile(samples, 50))
        result[f"batch{batch_size}_p95_ms"] = float(np.percentile(samples, 95))
    return result


def _eval_metrics(variant: Variant) -> tuple[float, float]:
    with open(variant.eval_json) as f:
        report = json.load(f)
    return float(report["accuracy"]), float(report["f1_macro"])


def _build_rows(
    texts: list[str],
    batch_sizes: list[int],
    max_length: int,
    runs: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    baseline_size = _checkpoint_size_mb(VARIANTS[0].path)
    backends = {
        "baseline-torch": "pytorch-cpu-fp32",
        "distilled-torch": "pytorch-cpu-fp32",
        "onnx-fp32": "onnxruntime-cpu-fp32",
        "onnx-int8": "onnxruntime-cpu-int8",
    }
    for variant in VARIANTS:
        accuracy, f1_macro = _eval_metrics(variant)
        if variant.kind == "torch":
            size_mb = _checkpoint_size_mb(variant.path)
            size_ratio = "1.0x"
        else:
            size_mb = _onnx_size_mb(variant.path, variant.extra_weights)
            size_ratio = f"{size_mb / baseline_size:.3f}x"
        logger.info(f"Measuring latency for {variant.name}")
        latency = measure_latency(variant, texts, batch_sizes, max_length, runs)
        rows.append(
            {
                "variant": variant.name,
                "backend": backends[variant.name],
                "accuracy": accuracy,
                "f1_macro": f1_macro,
                **latency,
                "size_mb": round(size_mb, 1),
                "size_vs_baseline": size_ratio,
                "note": variant.note,
            }
        )
    return rows


def _markdown_table(
    rows: list[dict[str, Any]],
    samples: int,
    batch_sizes: list[int],
    runs: int,
    max_length: int,
) -> str:
    header = [
        "variant",
        "backend",
        "accuracy",
        "f1_macro",
        "batch1 p50 (ms)",
        "batch1 p95 (ms)",
        f"batch{max(batch_sizes)} p50 (ms)",
        f"batch{max(batch_sizes)} p95 (ms)",
        "size (MB)",
        "size vs baseline",
        "note",
    ]
    batch_size = max(batch_sizes)
    lines = ["# Benchmark table", ""]
    lines.append(
        f"_Auto-generated by `uv run python scripts/benchmark.py`. {samples} "
        f"held-out samples, {runs} pass(es), max_length={max_length}. batch1 = one "
        f"single-sample call; batch{max(batch_sizes)} = one call over a batch of "
        f"{max(batch_sizes)} (chunk-level latency)._"
    )
    lines += [
        "",
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for row in rows:
        cells = [
            row["variant"],
            row["backend"],
            f"{row['accuracy']:.4f}",
            f"{row['f1_macro']:.4f}",
            f"{row['batch1_p50_ms']:.1f}",
            f"{row['batch1_p95_ms']:.1f}",
            f"{row[f'batch{batch_size}_p50_ms']:.1f}",
            f"{row[f'batch{batch_size}_p95_ms']:.1f}",
            f"{row['size_mb']:.1f}",
            str(row["size_vs_baseline"]),
            str(row["note"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def _write_outputs(rows: list[dict[str, Any]], md: str, output_path: Path) -> None:
    out_dir = output_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{output_path.stem}.md").write_text(md)
    with open(output_path, "w", newline="") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Wrote {output_path.parent / f'{output_path.stem}.md'}")
    logger.info(f"Wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--batch-sizes", default="1,32")
    parser.add_argument("--test-file", default=DefaultPaths.TEST_SPLIT.value)
    parser.add_argument("--output", default="reports/benchmark_table.csv")
    args = parser.parse_args()

    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    logger.info(f"Loading {args.samples} samples from {args.test_file}")
    text_df = pd.read_csv(args.test_file).head(args.samples)
    texts = text_df["text"].tolist()

    rows = _build_rows(texts, batch_sizes, args.max_length, args.runs)
    for row in rows:
        logger.info(
            f"{row['variant']:16s} acc={row['accuracy']:.4f} "
            f"f1={row['f1_macro']:.4f} "
            f"b1 p50/p95={row['batch1_p50_ms']:.1f}/{row['batch1_p95_ms']:.1f}ms "
            f"b{batch_sizes[-1]} p50/p95={row[f'batch{batch_sizes[-1]}_p50_ms']:.1f}/"
            f"{row[f'batch{batch_sizes[-1]}_p95_ms']:.1f}ms "
            f"size={row['size_mb']} MB"
        )
    md = _markdown_table(rows, args.samples, batch_sizes, args.runs, args.max_length)
    _write_outputs(rows, md, Path(args.output))


if __name__ == "__main__":
    main()
