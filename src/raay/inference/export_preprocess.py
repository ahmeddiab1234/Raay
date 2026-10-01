"""Text preprocessing helper for ONNX export."""

from __future__ import annotations

import warnings

from raay.enums.constants import Models

warnings.filterwarnings("ignore", category=SyntaxWarning)

try:
    from arabert.preprocess import ArabertPreprocessor
except ImportError:  # pragma: no cover - import path guard
    ArabertPreprocessor = None


def preprocess_text(text: str) -> str:
    if ArabertPreprocessor is not None:
        return ArabertPreprocessor(model_name=Models.TEACHER.value).preprocess(text)
    return str(text)


# Backward-compatible alias for underscored import surface.
_preprocess = preprocess_text
