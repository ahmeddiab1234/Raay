"""Load a variant and measure it over the same path the service uses.

The point of measuring here rather than quoting ``serving_benchmark.json`` is that
this is the real production path: ``ArabertPreprocessor`` -> tokenizer -> torch
forward or ORT run. A warm-up call is issued per (variant, batch size) before
timing, because the first ORT call in a process pays allocation costs that would
otherwise be attributed to the model.
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from benchmark_variants import (
    BACKENDS,
    VARIANTS,
    Variant,
    checkpoint_size_mb,
    onnx_size_mb,
)
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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


def build_rows(
    texts: list[str],
    batch_sizes: list[int],
    max_length: int,
    runs: int,
) -> list[dict[str, Any]]:
    """One table row per variant: accuracy, latency, and size on disk."""
    rows: list[dict[str, Any]] = []
    baseline_size = checkpoint_size_mb(VARIANTS[0].path)
    for variant in VARIANTS:
        accuracy, f1_macro = _eval_metrics(variant)
        if variant.kind == "torch":
            size_mb = checkpoint_size_mb(variant.path)
            size_ratio = "1.0x"
        else:
            size_mb = onnx_size_mb(variant.path, variant.extra_weights)
            size_ratio = f"{size_mb / baseline_size:.3f}x"
        logger.info(f"Measuring latency for {variant.name}")
        latency = measure_latency(variant, texts, batch_sizes, max_length, runs)
        rows.append(
            {
                "variant": variant.name,
                "backend": BACKENDS[variant.name],
                "accuracy": accuracy,
                "f1_macro": f1_macro,
                **latency,
                "size_mb": round(size_mb, 1),
                "size_vs_baseline": size_ratio,
                "note": variant.note,
            }
        )
    return rows
