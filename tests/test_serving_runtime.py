"""Graph resolution and the shared predict/softmax pipeline.

The ONNX-backed tests export a tiny real graph per test
(:mod:`tests.tiny_graph`), so ``predict_probs`` is exercised end to end without
a checkpoint on disk.
"""

import numpy as np
import pytest
from tiny_graph import ID2LABEL, FakeTokenizer, tiny_session

from raay.serving.runtime import (
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
    with pytest.raises(RuntimeError, match="model.onnx"):
        _resolve_onnx_path({}.get, download=lambda uri: str(empty))


def test_resolve_raises_when_download_fails():
    def boom(_uri):
        raise RuntimeError("no such model")

    with pytest.raises(RuntimeError, match="RAAY_ONNX_PATH"):
        _resolve_onnx_path({}.get, download=boom)


def test_softmax_rows_sum_to_one_and_argmax_preserved():
    logits = np.array([[1.0, 2.0, 0.5], [0.1, 0.2, 3.0], [-1.0, 4.0, 5.0]])
    probs = softmax(logits)
    np.testing.assert_allclose(probs.sum(axis=-1), np.ones(3), atol=1e-6)
    np.testing.assert_array_equal(np.argmax(probs, axis=-1), np.argmax(logits, axis=-1))


def test_to_predictions_maps_labels_in_id_order():
    probs = np.array([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]])
    predicted = to_predictions(probs, ID2LABEL)
    assert [p["label"] for p in predicted] == ["negative", "positive"]
    assert abs(predicted[0]["score"] - 0.7) < 1e-6
    assert abs(predicted[1]["score"] - 0.8) < 1e-6


def test_predict_probs_runs_tiny_onnx(tmp_path):
    probs = predict_probs(
        tiny_session(tmp_path),
        FakeTokenizer(),
        ["هذا ممتاز", "الجودة رديئة"],
        model_name="tiny",
        max_length=32,
    )
    assert probs.shape == (2, 3)
    np.testing.assert_allclose(probs.sum(axis=-1), np.ones(2), atol=1e-6)


def test_predict_probs_batches_cleanly(tmp_path):
    """Seven texts through a batch of four: the chunk loop must not drop or
    duplicate a row, which a ``range(0, n, batch_size)`` off-by-one would."""
    probs = predict_probs(
        tiny_session(tmp_path),
        FakeTokenizer(),
        [f"مراجعة رقم {i}" for i in range(7)],
        model_name="tiny",
        max_length=32,
        batch_size=4,
    )
    assert probs.shape == (7, 3)


def test_to_predictions_roundtrip_via_probs(tmp_path):
    probs = predict_probs(
        tiny_session(tmp_path),
        FakeTokenizer(),
        ["شيء ممتاز"],
        model_name="tiny",
        max_length=32,
    )
    predictions = to_predictions(probs, ID2LABEL)
    assert len(predictions) == 1
    assert predictions[0]["label"] in ID2LABEL.values()
    assert 0.0 <= predictions[0]["score"] <= 1.0
