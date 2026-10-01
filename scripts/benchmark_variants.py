"""The four variants the benchmark table compares.

``onnx-fp32``'s accuracy/F1 is *inherited* from ``eval_baseline.json`` because the
exported graph holds identical weights to the torch checkpoint; the ``note``
column marks that rather than re-measuring, so the table can never disagree with
the baseline row.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from raay.enums.constants import DefaultPaths, Models


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

BACKENDS = {
    "baseline-torch": "pytorch-cpu-fp32",
    "distilled-torch": "pytorch-cpu-fp32",
    "onnx-fp32": "onnxruntime-cpu-fp32",
    "onnx-int8": "onnxruntime-cpu-int8",
}


def checkpoint_size_mb(dir_path: str) -> float:
    """Sum every file under a checkpoint dir (weights + tokenizer + config)."""
    return sum(p.stat().st_size for p in Path(dir_path).rglob("*") if p.is_file()) / 1e6


def onnx_size_mb(graph_path: str, extra_weights: tuple[str, ...]) -> float:
    """Graph size plus any external ``.onnx.data`` sidecar."""
    return (
        Path(graph_path).stat().st_size
        + sum(Path(w).stat().st_size for w in extra_weights)
    ) / 1e6
