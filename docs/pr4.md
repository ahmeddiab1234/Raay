# PR #4 -Compression / Optimization

## Summary

This phase reduces the inference cost of the AraBERT sentiment model while preserving its predictive quality. A 6-layer student was trained through knowledge distillation, the teacher and student were exported to ONNX, the teacher was dynamically quantized to INT8, and all variants were benchmarked and tracked in MLflow.

## What Was Done

### 1. Knowledge Distillation

- Ran the distillation sweep on a Kaggle GPU using `scripts/kaggle_train_runs.py --module distill --n 6`.
- Trained a 6-layer student from the fine-tuned AraBERT teacher.
- Merged the Kaggle MLflow runs into the local tracking store and restored the distilled model and tokenizer under `models/distilled/final/`.
- Evaluated the student on the held-out dialect-stratified test split, writing `reports/eval_distilled.json`.

### 2. ONNX Export and Quantization

- Exported both the baseline teacher and distilled student to ONNX with dynamic batch and sequence dimensions.
- Validated PyTorch-versus-ONNX Runtime logit parity on Arabic samples and wrote `reports/onnx_parity.json`.
- Logged ONNX graphs, external weights, export parameters, and metrics to the `raay_training` MLflow experiment.
- Dynamically quantized the teacher ONNX graph to a self-contained INT8 model at `models/onnx/model_int8.onnx`.
- Confirmed INT8 parity and evaluated the quantized model, writing `reports/onnx_int8_parity.json` and `reports/eval_int8.json`.

### 3. Serving and Benchmarking

The unified benchmark compared baseline PyTorch, distilled PyTorch, FP32 ONNX, and INT8 ONNX using accuracy, macro F1, p50/p95 latency, batch-32 p95 latency, and disk size. Results were written to `reports/benchmark_table.csv` and `reports/benchmark_table.md`.

| Variant          |          Accuracy | F1 (macro) |               Size | CPU single-call p95 |
| ---------------- | ----------------: | ---------: | -----------------: | ------------------: |
| Baseline PyTorch |           84.92 % |     0.6407 |                 — |                  — |
| ONNX INT8        | **85.03 %** |     0.6369 | **136.1 MB** |   **27.6 ms** |

Through the same BentoML service, the 8-user Locust run produced:

| Variant   |          Median |              p95 |              p99 | Failures |
| --------- | --------------: | ---------------: | ---------------: | -------: |
| FP32 ONNX |           56 ms |           130 ms |           160 ms |        0 |
| INT8 ONNX | **38 ms** | **110 ms** | **130 ms** |        0 |

The CPU INT8 benchmark met the observed serving target closely enough to make it the current production choice. TensorRT was retained as a future accelerator path; an engine should be built on the fixed production GPU, CUDA, and TensorRT environment.

### 4. MLflow Registration

- Logged each of the four variants as a separate `raay_training` run with stage tags: `baseline`, `distilled`, `fp32`, and `int8`.
- Logged accuracy, macro F1, latency, and size metrics for model comparison.
- Registered the best tradeoff as `ArabicSentiment` version 4.
- Promoted `onnx-int8` to the `Production` alias and archived the previous production version.
- Configured the BentoML service to resolve the promoted registry model by default, while retaining `RAAY_ONNX_PATH` as an explicit override for benchmarks.

## Artifacts

- `reports/eval_distilled.json`
- `reports/onnx_parity.json`
- `reports/onnx_int8_parity.json`
- `reports/eval_int8.json`
- `reports/serving_benchmark.json`
- `reports/benchmark_table.{csv,md}`
- `models/onnx/model.onnx`
- `models/onnx/distilled.onnx`
- `models/onnx/model_int8.onnx`

## Verification

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest
```

The complete execution record is available in [`docs/commands_run_p4.txt`](commands_run_p4.txt).
