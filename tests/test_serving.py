import numpy as np
import torch
from transformers import BertConfig, BertForSequenceClassification

from raay.inference.export_onnx import export_to_onnx
from raay.serving.serve import (
    _resolve_onnx_path,
    predict_probs,
    softmax,
    to_predictions,
)


def test_resolve_uses_raay_onnx_path_override():
    env = {"RAAY_ONNX_PATH": "/tmp/override.onnx"}
    source, path, detail = _resolve_onnx_path(env.get, download=lambda uri: uri)
    assert source == "RAAY_ONNX_PATH"
    assert path == "/tmp/override.onnx"
    assert detail == "/tmp/override.onnx"


def test_resolve_from_registry_alias(tmp_path):
    model_dir = tmp_path / "registered"
    model_dir.mkdir()
    (model_dir / "model.onnx").write_bytes(b"fake")
    env = {}
    source, path, _ = _resolve_onnx_path(env.get, download=lambda uri: str(model_dir))
    assert source == "models:/ArabicSentiment/Production"
    assert path == str(model_dir / "model.onnx")


def test_resolve_registry_alias_missing_onnx_raises(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    try:
        _resolve_onnx_path({}.get, download=lambda uri: str(empty))
    except RuntimeError as exc:
        assert "model.onnx" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for missing model.onnx")


def test_resolve_raises_when_download_fails():
    def boom(_uri):
        raise RuntimeError("no such model")

    try:
        _resolve_onnx_path({}.get, download=boom)
    except RuntimeError as exc:
        assert "RAAY_ONNX_PATH" in str(exc)
    else:
        raise AssertionError("expected RuntimeError when the alias is unresolvable")


def test_softmax_rows_sum_to_one_and_argmax_preserved():
    logits = np.array([[1.0, 2.0, 0.5], [0.1, 0.2, 3.0], [-1.0, 4.0, 5.0]])
    probs = softmax(logits)
    np.testing.assert_allclose(probs.sum(axis=-1), np.ones(3), atol=1e-6)
    np.testing.assert_array_equal(np.argmax(probs, axis=-1), np.argmax(logits, axis=-1))


def test_to_predictions_maps_labels_in_id_order():
    probs = np.array([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]])
    id2label = {0: "positive", 1: "negative", 2: "neutral"}
    predicted = to_predictions(probs, id2label)
    assert [p["label"] for p in predicted] == ["negative", "positive"]
    assert abs(predicted[0]["score"] - 0.7) < 1e-6
    assert abs(predicted[1]["score"] - 0.8) < 1e-6


class _FakeTokenizer:
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


def _tiny_session(tmp_path, with_labels=False):
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
    enc = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
    onnx_path = str(tmp_path / "tiny.onnx")
    export_to_onnx(model, enc, onnx_path, opset=17)
    import onnxruntime as ort

    return ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])


def test_predict_probs_runs_tiny_onnx(tmp_path):
    session = _tiny_session(tmp_path)
    texts = ["هذا ممتاز", "الجودة رديئة"]
    probs = predict_probs(
        session, _FakeTokenizer(), texts, model_name="tiny", max_length=32
    )
    assert probs.shape == (2, 3)
    np.testing.assert_allclose(probs.sum(axis=-1), np.ones(2), atol=1e-6)


def test_predict_probs_batches_cleanly(tmp_path):
    session = _tiny_session(tmp_path)
    texts = [f"مراجعة رقم {i}" for i in range(7)]
    probs = predict_probs(
        session,
        _FakeTokenizer(),
        texts,
        model_name="tiny",
        max_length=32,
        batch_size=4,
    )
    assert probs.shape == (7, 3)


def test_to_predictions_roundtrip_via_probs(tmp_path):
    session = _tiny_session(tmp_path)
    id2label = {0: "positive", 1: "negative", 2: "neutral"}
    probs = predict_probs(
        session, _FakeTokenizer(), ["شيء ممتاز"], model_name="tiny", max_length=32
    )
    predictions = to_predictions(probs, id2label)
    assert len(predictions) == 1
    assert predictions[0]["label"] in id2label.values()
    assert 0.0 <= predictions[0]["score"] <= 1.0
