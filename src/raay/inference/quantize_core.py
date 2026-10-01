"""INT8 dynamic quantization + FP32-vs-INT8 parity for an exported ONNX graph.

Two workarounds here are load-bearing, both caused by the torch dynamo exporter:

* **Stale ``value_info``.** The exporter leaves ``value_info`` entries whose
  declared shapes conflict with onnx's file-based shape inference, and
  ``quantize_dynamic`` re-runs a *strict* file-based inference and raises on those
  mismatches. Dropping ``value_info`` before quantizing removes the clash.
* **External weights.** The exporter writes weights as a sibling ``.onnx.data``.
  The quantizer's external-weights churn does not survive being pointed at a temp
  dir, so the FP32 graph is first inlined (``save_as_external_data=False``) and the
  INT8 output is likewise self-contained.

Output artifact ``models/onnx/model_int8.onnx`` is ~136 MB with everything inside
it, which is what lets the bento image bake a single file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from loguru import logger
from onnxruntime.quantization import QuantType, quantize_dynamic

from raay.enums.constants import DefaultPaths
from raay.inference.export_onnx import _preprocess
from raay.training.evaluate import load_onnx_session

SAMPLE_TEXTS: tuple[str, ...] = (
    "هذا المنتج ممتاز والجودة عالية جدا",
    "المنتج وصل متأخر والجودة رديئة",
    "الطلبية وصلت بسرعة والحاجة تمام جدا شكرا",
    "حسبي الله ونعم الوكيل ياخي الجودة خايسة",
    "المنتج محايد شكله عادي",
    "الخدمة ممتازة وسعر مناسب لكن التوصيل بطيء",
)

TOKENIZER_DIRS = {
    "model": DefaultPaths.BASELINE_MODEL.value,
    "distilled": DefaultPaths.DISTILLED_MODEL.value,
}


def _tokenizer_dir_for(onnx_path: str) -> str:
    """Pick the tokenizer dir matching the graph, else `<parent>/../final`."""
    return TOKENIZER_DIRS.get(
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

    Inlines the FP32 weights and drops ``value_info`` first (see module docstring),
    then quantizes that self-contained graph.
    """
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
    enc = _encode(tokenizer, list(SAMPLE_TEXTS), max_length)
    fp32_out = _run(fp32, enc)
    int8_out = _run(int8, enc)
    diff = np.abs(fp32_out - int8_out)
    return {
        "n_samples": len(SAMPLE_TEXTS),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "label_agreement": float(
            np.mean(np.argmax(fp32_out, axis=-1) == np.argmax(int8_out, axis=-1))
        ),
    }
