"""A tiny real ONNX graph + tokenizer, shared by the serving unit tests.

Hermetic like everything else in ``tests/``: the model is a 2-layer BERT with a
32-wide hidden state built on the spot and exported to a ``tmp_path``, so
``predict_probs`` is exercised end to end (torch -> ONNX -> ORT) without a
checkpoint or a 136 MB graph on disk.
"""

from __future__ import annotations

import torch
from transformers import BertConfig, BertForSequenceClassification

from raay.inference.export_onnx import export_to_onnx

ID2LABEL = {0: "positive", 1: "negative", 2: "neutral"}


class FakeTokenizer:
    """Minimal callable tokenizer standing in for the HF fast tokenizer."""

    vocab_size = 512
    max_len = 32

    def __call__(
        self, texts, truncation=True, padding=True, max_length=128, return_tensors="pt"
    ):
        n = len(texts)
        length = min(max_length, self.max_len)
        return {
            "input_ids": torch.randint(
                0, self.vocab_size, (n, length), dtype=torch.long
            ),
            "attention_mask": torch.ones((n, length), dtype=torch.long),
        }


def tiny_config() -> BertConfig:
    """A 2-layer / 32-hidden BERT with the project's real label ordering."""
    return BertConfig(
        vocab_size=512,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
        num_labels=3,
        id2label={str(i): s for i, s in enumerate(ID2LABEL.values())},
    )


def tiny_session(tmp_path, name: str = "tiny.onnx"):
    """Export :func:`tiny_config` to ``tmp_path`` and return an ORT session."""
    import onnxruntime as ort

    torch.manual_seed(0)
    model = BertForSequenceClassification(tiny_config())
    model.eval()
    input_ids = torch.randint(0, 512, (2, 16), dtype=torch.long)
    enc = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
    onnx_path = str(tmp_path / name)
    export_to_onnx(model, enc, onnx_path, opset=17)
    return ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
