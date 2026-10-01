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
import warnings
from pathlib import Path

import pandas as pd
from benchmark_measure import build_rows as _build_rows
from benchmark_report import markdown_table as _markdown_table
from benchmark_report import write_outputs as _write_outputs
from loguru import logger

from raay.enums.constants import DefaultPaths

warnings.filterwarnings("ignore", category=SyntaxWarning)


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
