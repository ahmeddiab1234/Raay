import numpy as np
import torch
from transformers import BertConfig, BertForSequenceClassification

from raay.inference.export_onnx import export_to_onnx
from raay.inference.quantize_onnx import (
    _tokenizer_dir_for,
    parity_report,
    quantize_to_int8,
)


def test_tokenizer_dir_for():
    assert _tokenizer_dir_for("models/onnx/model.onnx") == "models/baseline/final"
    assert _tokenizer_dir_for("models/onnx/distilled.onnx") == "models/distilled/final"
    assert _tokenizer_dir_for("models/onnx/custom.onnx") == "models/final"


def _tiny_model_and_onnx(tmp_path):
    torch.manual_seed(0)
    config = BertConfig(
        vocab_size=512,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
        num_labels=3,
        id2label={str(i): s for i, s in enumerate(["positive", "negative", "neutral"])},
    )
    model = BertForSequenceClassification(config)
    model.eval()
    input_ids = torch.randint(0, 512, (2, 16), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    onnx_path = str(tmp_path / "tiny.onnx")
    export_to_onnx(
        model, {"input_ids": input_ids, "attention_mask": attention_mask}, onnx_path, 17
    )
    return onnx_path


def test_quantize_to_int8_smoke(tmp_path):
    onnx_path = _tiny_model_and_onnx(tmp_path)
    int8_path = str(tmp_path / "tiny_int8.onnx")

    quantize_to_int8(onnx_path, int8_path)

    import onnxruntime as ort

    fp32 = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    int8 = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])
    feed = {
        "input_ids": torch.randint(0, 512, (3, 16)).numpy(),
        "attention_mask": np.ones((3, 16), dtype=np.int64),
    }
    fp32_out = fp32.run(["logits"], feed)[0]
    int8_out = int8.run(["logits"], feed)[0]

    assert np.array_equal(np.argmax(fp32_out, axis=-1), np.argmax(int8_out, axis=-1))
    assert int8_path.endswith("tiny_int8.onnx")


def test_parity_report_within_tolerance(tmp_path):
    onnx_path = _tiny_model_and_onnx(tmp_path)
    int8_path = str(tmp_path / "tiny_int8.onnx")
    quantize_to_int8(onnx_path, int8_path)

    class FakeTokenizer:
        def __call__(self, texts, **kwargs):
            return {
                "input_ids": torch.randint(0, 512, (len(texts), 16)),
                "attention_mask": torch.ones((len(texts), 16), dtype=torch.long),
            }

        def __getattr__(self, item):
            return None

    result = parity_report(onnx_path, int8_path, FakeTokenizer(), max_length=16)
    assert result["n_samples"] == 6
    assert 0.0 <= result["label_agreement"] <= 1.0
    assert result["max_abs_diff"] >= 0.0
