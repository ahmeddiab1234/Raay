"""Export constants and sample texts for ONNX parity checks."""

from __future__ import annotations

from raay.enums.constants import DefaultPaths

_SAMPLE_TEXTS: tuple[str, ...] = (
    "هذا المنتج ممتاز والجودة عالية جدا",
    "المنتج وصل متأخر والجودة رديئة",
    "الطلبية وصلت بسرعة والحاجة تمام جدا شكرا",
    "حسبي الله ونعم الوكيل ياخي الجودة خايسة",
    "المنتج محايد شكله عادي",
    "الخدمة ممتازة وسعر مناسب لكن التوصيل بطيء",
)

_DEFAULT_OUTPUT_NAMES = {
    DefaultPaths.BASELINE_MODEL.value: "model",
    DefaultPaths.DISTILLED_MODEL.value: "distilled",
}
