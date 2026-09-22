from pathlib import Path

import numpy as np
import torch
from transformers import BertConfig, BertForSequenceClassification

from raay.inference.export_onnx import (
    _output_name_for,
    _repair_artifact_locations,
    export_to_onnx,
    verify_parity,
)


def test_output_name_defaults():
    assert _output_name_for("models/baseline/final", None) == "model"
    assert _output_name_for("models/distilled/final", None) == "distilled"
    assert _output_name_for("models/other/final", None) == "other"
    assert _output_name_for("models/distilled/final", "custom") == "custom"


def test_verify_parity_passes_within_tolerance():
    logits = np.array([[1.0, 2.0, 0.5], [0.1, 0.2, 3.0]])
    result = verify_parity(logits, logits + 1e-5, tolerance=1e-4)
    assert result["passed"] is True
    assert result["argmax_match"] is True
    assert result["max_abs_diff"] <= 1e-4


def test_verify_parity_fails_beyond_tolerance():
    logits = np.array([[1.0, 2.0, 0.5]])
    result = verify_parity(logits, logits + 0.1, tolerance=1e-4)
    assert result["passed"] is False
    assert result["max_abs_diff"] > 1e-4


def test_verify_parity_fails_on_argmax_flip():
    logits = np.array([[1.0, 2.0, 0.5]])
    flipped = np.array([[0.0, 1.0, 3.0]])
    result = verify_parity(logits, flipped, tolerance=10.0)
    assert result["passed"] is False
    assert result["argmax_match"] is False


def test_verify_parity_tolerance_boundary_semantics():
    logits = np.array([[1.0, 2.0, 0.5]])
    margin = 1e-9
    assert (
        verify_parity(logits, logits + (1e-4 - margin), tolerance=1e-4)["passed"]
        is True
    )
    assert (
        verify_parity(logits, logits + (1e-4 + margin), tolerance=1e-4)["passed"]
        is False
    )


def test_repair_artifact_locations_sqlite(tmp_path):
    db = tmp_path / "mlflow.db"
    uri = f"sqlite:///{db}"
    import sqlite3

    con = sqlite3.connect(db)
    con.execute("create table experiments (experiment_id text, artifact_location text)")
    con.executemany(
        "insert into experiments values (?, ?)",
        [
            ("1", "file:///kaggle/working/mlruns/1"),
            ("2", "./mlruns/2"),
        ],
    )
    con.commit()
    con.close()

    assert _repair_artifact_locations(uri) == 1

    con = sqlite3.connect(db)
    rows = dict(con.execute("select experiment_id, artifact_location from experiments"))
    con.close()
    assert rows["1"] == "./mlruns/1"
    assert rows["2"] == "./mlruns/2"


def test_repair_artifact_locations_skips_existing_kaggle_dir():
    # Point at a kaggle-shaped path that actually exists/writable: no repair.
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        kaggle = Path(td) / "kaggle/working/mlruns"
        (kaggle / "1").mkdir(parents=True)
        uri = f"sqlite:///{td}/mlflow.db"
        import sqlite3

        con = sqlite3.connect(uri[10:])
        con.execute(
            "create table experiments (experiment_id text, artifact_location text)"
        )
        con.execute(
            "insert into experiments values (?, ?)",
            ("1", f"file://{kaggle}/1"),
        )
        con.commit()
        con.close()

        assert _repair_artifact_locations(uri) == 0


def test_repair_artifact_locations_file_store(tmp_path):
    exp = tmp_path / "mlruns"
    (exp / "0").mkdir(parents=True)
    (exp / "0" / "meta.yaml").write_text(
        "name: raay_training\nartifact_location: file:///kaggle/working/mlruns/0\n"
    )

    assert _repair_artifact_locations(f"file:{exp}") == 1

    text = (exp / "0" / "meta.yaml").read_text()
    assert "artifact_location: ./mlruns/0" in text


def test_export_and_ort_parity_smoke(tmp_path):
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
    enc = {"input_ids": input_ids, "attention_mask": attention_mask}

    onnx_path = str(tmp_path / "tiny.onnx")
    export_to_onnx(model, enc, onnx_path, opset=17)

    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    feed = {
        "input_ids": input_ids.numpy(),
        "attention_mask": attention_mask.numpy(),
    }
    with torch.no_grad():
        pt_logits = model(**enc).logits.numpy()
    ort_logits = session.run(["logits"], feed)[0]

    assert verify_parity(pt_logits, ort_logits, tolerance=1e-4)["passed"] is True
