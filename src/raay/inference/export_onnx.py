"""Export the fine-tuned AraBERT checkpoints to ONNX and validate parity.

Phase 3 (compression/optimization) step 3: freeze each checkpoint's BERT
backbone into an ``onnxruntime``-runnable graph via ``torch.onnx.export`` and
confirm the exported logits match PyTorch within a tolerance.

Run from the repo root:

    uv run python -m raay.inference.export_onnx

By default it exports BOTH local checkpoints:
    - models/baseline/final   -> models/onnx/model.onnx
    - models/distilled/final  -> models/onnx/distilled.onnx

Use ``--model-dir`` (repeatable) to override and ``--name`` to pick the output
basename (fallback: parent basename, e.g. ``models/onnx/distilled.onnx``).

The graph only contains the BERT classification head with inputs
``{input_ids, attention_mask}`` (``token_type_ids`` is folded as a constant
zero tensor at trace time — single-sentence classification). The
``ArabertPreprocessor`` text normalization and the HF fast tokenizer stay in
Python, exactly as in the training/eval pipeline, so every row fed to ONNX
uses the same preprocessed text.

Validates parity by running a fixed set of Arabic sample reviews through both
PyTorch and ONNX Runtime and comparing logits (max abs diff vs tolerance, plus
argmax agreement), writes ``reports/onnx_parity.json`` and logs the ONNX
artifacts + parity metrics to the MLflow experiment.

A local ``mlflow.db`` copied from a Kaggle GPU session records ``/kaggle``
artifact roots for its experiments; before exporting, the script idempotently
repoints those dead roots to the local ``./mlruns`` store so artifacts land on
disk (skipped when a real writable Kaggle workspace is present).

Implementation lives in siblings -- ``export_config`` (sample texts + output
names), ``export_preprocess`` (Arabert normalization), ``export_parity`` (model
loading, torch/ORT logits, parity check), ``export_artifacts`` (output naming +
MLflow artifact-root repair), ``export_one`` (one checkpoint end to end), and
``export_cli`` (argument parsing) -- re-exported here so callers keep one
import path.

Requires deps: onnx, onnxruntime, onnxscript.
"""

from __future__ import annotations

import warnings

from raay.inference.export_artifacts import (
    _output_name_for,
    _repair_artifact_locations,
    _report_results,
    output_name_for,
)
from raay.inference.export_cli import main
from raay.inference.export_config import (
    _DEFAULT_OUTPUT_NAMES,
    _SAMPLE_TEXTS,
)
from raay.inference.export_one import export_one
from raay.inference.export_parity import (
    export_to_onnx,
    load_model,
    run_ort,
    torch_logits,
    verify_parity,
)
from raay.inference.export_preprocess import _preprocess, preprocess_text

warnings.filterwarnings("ignore", category=SyntaxWarning)

__all__ = [
    "export_one",
    "export_to_onnx",
    "load_model",
    "main",
    "output_name_for",
    "preprocess_text",
    "run_ort",
    "torch_logits",
    "verify_parity",
]

# Underscored names below stay importable from this module because the tests
# and ``quantize_onnx`` import them directly.
_PRIVATE_REEXPORTS = (
    _DEFAULT_OUTPUT_NAMES,
    _SAMPLE_TEXTS,
    _output_name_for,
    _preprocess,
    _repair_artifact_locations,
    _report_results,
)


if __name__ == "__main__":
    main()
