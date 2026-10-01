"""Evaluate a checkpoint on the frozen test split.

Run from the repo root:

    uv run python -m raay.training.evaluate \
        --model-dir models/baseline/final \
        --output reports/eval_baseline.json

    uv run python -m raay.training.evaluate \
        --model-dir models/baseline/final --onnx-path models/onnx/model_int8.onnx

By default it evaluates the fine-tuned PyTorch checkpoint in ``--model-dir``;
``--onnx-path`` switches the same split to ONNX Runtime inference. Both write a
JSON report with accuracy, macro/weighted F1, per-class precision/recall/F1, the
confusion matrix, a dialect-stratified breakdown, and the metadata describing
which backend produced it.

Two label-encoding details are load-bearing:

- ``evaluate_on_split`` reads ``id2label`` off ``model.config``. An ONNX Runtime
  session has no ``.config``, so the caller must attach one (``promote_model.py``
  does) or the labels fall back to alphabetical order and Neutral/Negative are
  silently swapped.
- The id order is the one in ``raay.enums.constants.LABELS``
  (positive=0, negative=1, neutral=2), never retyped.

Implementation lives in siblings -- ``eval_metrics`` (backends + split metrics),
``eval_plot`` (the MLflow comparison bar chart) and ``eval_cli`` (argument
parsing + report writing) -- re-exported here so ``raay.training.evaluate.<name>``
keeps one import path. The promotion gate's structural tests read this module's
public surface, not its layout.
"""

from __future__ import annotations

from raay.training.eval_cli import main
from raay.training.eval_metrics import (
    _per_class,
    _preprocess,
    dialect_breakdown,
    evaluate_on_split,
    load_model,
    load_onnx_session,
    predict,
)
from raay.training.eval_plot import plot_mlflow_comparison

__all__ = [
    "dialect_breakdown",
    "evaluate_on_split",
    "load_model",
    "load_onnx_session",
    "main",
    "plot_mlflow_comparison",
    "predict",
]

_PRIVATE_REEXPORTS = (_per_class, _preprocess)


if __name__ == "__main__":
    main()
