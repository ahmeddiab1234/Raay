"""Export the fine-tuned AraBERT checkpoints to ONNX and validate parity.

Phase 3 (compression/optimization) step 3: freeze each checkpoints' BERT
backbone into an ``onnxruntime``-runnable graph via ``torch.onnx.export`` and
confirm the exported logits match PyTorch within a tolerance.

Run from the repo root:

    uv run python -m raay.inference.export_onnx

By default it exports BOTH local checkpoints:
    - models/baseline/final   -> models/onnx/model.onnx
    - models/distilled/final  -> models/onnx/distilled.onnx

Use ``--model-dir`` (repeatable) to override and ``--name`` to pick the output
basename (fallback: parent basename, e.g. ``models/onnx/distilled.onnx``).

The graph only contains the BERT classification head with inputs
``{input_ids, attention_mask}`` (``token_type_ids`` is folded as a constant
zero tensor at trace time — single-sentence classification). The
``ArabertPreprocessor`` text normalization and the HF fast tokenizer stay in
Python, exactly as in the training/eval pipeline, so every row fed to ONNX
uses the same preprocessed text.

Validates parity by running a fixed set of Arabic sample reviews through both
PyTorch and ONNX Runtime and comparing logits (max abs diff vs tolerance, plus
argmax agreement), writes ``reports/onnx_parity.json`` and logs the ONNX
artifacts + parity metrics to the MLflow experiment.

A local ``mlflow.db`` copied from a Kaggle GPU session records ``/kaggle``
artifact roots for its experiments; before exporting, the script idempotently
repoints those dead roots to the local ``./mlruns`` store so artifacts land on
disk (skipped when a real writable Kaggle workspace is present).

Requires deps: onnx, onnxruntime, onnxscript.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
import yaml
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import mlflow
from raay.config.env import load_environment, mlflow_tracking_uri
from raay.enums.constants import DefaultPaths, Experiments, Models

warnings.filterwarnings("ignore", category=SyntaxWarning)

try:
    from arabert.preprocess import ArabertPreprocessor
except ImportError:  # pragma: no cover - import path guard
    ArabertPreprocessor = None

_SAMPLE_TEXTS: tuple[str, ...] = (
    "هذا المنتج ممتاز والجودة عالية جدا",
    "المنتج وصل متأخر والجودة رديئة",
    "الطلبية وصلت بسرعة والحاجة تمام جدا شكرا",
    "حسبي الله ونعم الوكيل ياخي الجودة خايسة",
    "المنتج محايد شكله عادي",
    "الخدمة ممتازة وسعر مناسب لكن التوصيل بطيء",
)

# Well-known checkpoint dir -> onnx output basename (without suffix).
_DEFAULT_OUTPUT_NAMES = {
    DefaultPaths.BASELINE_MODEL.value: "model",
    DefaultPaths.DISTILLED_MODEL.value: "distilled",
}


def _preprocess(text: str) -> str:
    if ArabertPreprocessor is not None:
        return ArabertPreprocessor(model_name=Models.TEACHER.value).preprocess(text)
    return str(text)


def load_model(model_dir: str):
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    id2label = getattr(model.config, "id2label", None)
    return model, tokenizer, id2label


def export_to_onnx(
    model: torch.nn.Module,
    enc: dict[str, torch.Tensor],
    onnx_path: str,
    opset: int,
) -> None:
    """Freeze ``model`` to ``onnx_path`` with dynamic batch/sequence axes."""
    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model,
            (enc["input_ids"], enc["attention_mask"]),
            onnx_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "logits": {0: "batch"},
            },
            opset_version=opset,
            do_constant_folding=True,
        )
    logger.info(f"Exported {onnx_path} (opset={opset})")


def run_ort(session: ort.InferenceSession, enc: dict[str, torch.Tensor]) -> np.ndarray:
    """Run the exported graph and return the raw logits as a numpy array."""
    feed = {
        "input_ids": enc["input_ids"].numpy(),
        "attention_mask": enc["attention_mask"].numpy(),
    }
    return session.run(["logits"], feed)[0]


def torch_logits(model: torch.nn.Module, enc: dict[str, torch.Tensor]) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(**enc).logits.numpy()


def verify_parity(
    pt_logits: np.ndarray, ort_logits: np.ndarray, tolerance: float
) -> dict[str, Any]:
    """Compare PyTorch vs ONNX Runtime logits; return report + pass flag."""
    diff = np.abs(pt_logits - ort_logits)
    argmax_match = bool(
        np.array_equal(np.argmax(pt_logits, axis=-1), np.argmax(ort_logits, axis=-1))
    )
    result = {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "argmax_match": argmax_match,
        "tolerance": tolerance,
        "passed": bool(diff.max() <= tolerance) and argmax_match,
    }
    return result


def _output_name_for(model_dir: str, explicit_name: str | None) -> str:
    if explicit_name:
        return explicit_name
    return _DEFAULT_OUTPUT_NAMES.get(model_dir, Path(model_dir).parent.name)


def _repair_artifact_locations(tracking_uri: str) -> int:
    """Repoint experiments whose artifact root is a dead ``/kaggle`` path.

    The local ``mlflow.db`` is a copy of the Kaggle tracking store, whose
    experiments record ``file:///kaggle/working/mlruns/...`` artifact roots.
    On a local checkout those paths don't exist, so ``log_artifact`` would
    fail with a permission error. When the pointed-at directory is missing we
    idempotently rewrite the root to MLflow's native ``./mlruns/<exp>`` so
    artifacts land in a writable local store. On Kaggle the directory exists
    and nothing is touched. Returns the number of experiments repaired.
    """
    repaired = 0

    def fix(loc: str | None, exp_id: str, update) -> bool:  # type: ignore[no-untyped-def]
        if not loc or "/kaggle/" not in loc:
            return False
        path = loc.removeprefix("file://").split("?", 1)[0]
        if os.path.isdir(path) and os.access(path, os.W_OK):
            # A real (writable) Kaggle workspace is present: leave it alone.
            return False
        update(f"./mlruns/{exp_id}")
        return True

    if tracking_uri.startswith("sqlite:"):
        db_path = tracking_uri.removeprefix("sqlite:///")
        if not db_path or db_path == ":memory:":
            return 0
        con = sqlite3.connect(db_path)
        try:
            rows = con.execute(
                "select experiment_id, artifact_location from experiments"
            ).fetchall()
            for exp_id, loc in rows:
                if fix(
                    loc,
                    exp_id,
                    lambda new, eid=exp_id: con.execute(
                        "update experiments set artifact_location=? "
                        "where experiment_id=?",
                        (new, eid),
                    ),
                ):
                    repaired += 1
            con.commit()
        finally:
            con.close()
        return repaired

    if tracking_uri.startswith("file:"):
        root = Path(tracking_uri.removeprefix("file:"))
        for meta in root.glob("*/meta.yaml"):
            data = yaml.safe_load(meta.read_text()) or {}
            exp_id = meta.parent.name
            loc = data.get("artifact_location")

            def do_update(
                new: str, data: dict[str, Any] = data, meta: Path = meta
            ) -> None:
                data["artifact_location"] = new
                meta.write_text(yaml.safe_dump(data, sort_keys=False))

            if fix(loc, exp_id, do_update):
                repaired += 1
        return repaired

    return 0


def _report_results(report: dict[str, dict[str, Any]], report_path: str) -> None:
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Wrote parity report: {report_path}")


def export_one(
    model_dir: str,
    name: str,
    output_dir: str,
    max_length: int,
    opset: int,
    tolerance: float,
    experiment: str,
) -> dict[str, Any]:
    onnx_path = str(Path(output_dir) / f"{name}.onnx")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading model from {model_dir}")
    model, tokenizer, id2label = load_model(model_dir)

    texts = [_preprocess(t) for t in _SAMPLE_TEXTS]
    enc = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )

    export_to_onnx(model, enc, onnx_path, opset)

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    pt = torch_logits(model, enc)
    ort_out = run_ort(session, enc)
    parity = verify_parity(pt, ort_out, tolerance)
    logger.info(
        f"Parity [{name}]: max_abs_diff={parity['max_abs_diff']:.3e} "
        f"passed={parity['passed']}"
    )
    if not parity["passed"]:
        raise RuntimeError(f"ONNX parity check FAILED for {onnx_path}: {parity}")

    # torch's dynamo exporter writes weights to external `${name}.onnx.data`.
    externals: list[str] = []
    data_file = Path(onnx_path).with_name(f"{name}.onnx.data")
    if data_file.exists():
        externals.append(str(data_file))
    size_bytes = Path(onnx_path).stat().st_size + sum(
        Path(p).stat().st_size for p in externals
    )
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=f"export-onnx-{name}") as run:
        mlflow.log_params(
            {
                "model_dir": model_dir,
                "model_name": str(id2label or name),
                "opset": opset,
                "max_length": max_length,
                "onnx_size_bytes": size_bytes,
                "output_name": name,
                "n_sample_texts": len(_SAMPLE_TEXTS),
            }
        )
        mlflow.log_metrics(
            {
                "max_logits_abs_diff": parity["max_abs_diff"],
                "mean_logits_abs_diff": parity["mean_abs_diff"],
            }
        )
        mlflow.log_artifact(onnx_path, artifact_path="onnx")
        for ext in externals:
            mlflow.log_artifact(ext, artifact_path="onnx")
        logger.info(f"MLflow run {run.info.run_id} logged {onnx_path}")

    return {
        "model_dir": model_dir,
        "onnx_path": onnx_path,
        "opset": opset,
        "max_length": max_length,
        "onnx_size_bytes": size_bytes,
        **parity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        action="append",
        default=None,
        help=(
            "Checkpoint dir to export (repeatable). Defaults to both the "
            "baseline and the distilled checkpoints."
        ),
    )
    parser.add_argument(
        "--name",
        action="append",
        default=None,
        help=(
            "ONNX output basename (repeatable, pairs positionally with "
            "--model-dir). Default: baseline->model, distilled->distilled."
        ),
    )
    parser.add_argument("--output-dir", default="models/onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--report", default="reports/onnx_parity.json")
    parser.add_argument("--experiment", default=Experiments.TRAINING.value)
    parser.add_argument("--tracking-uri", default=None)
    args = parser.parse_args()

    model_dirs = args.model_dir or [
        DefaultPaths.BASELINE_MODEL.value,
        DefaultPaths.DISTILLED_MODEL.value,
    ]
    names = args.name or [None] * len(model_dirs)
    if len(names) != len(model_dirs):
        parser.error("--name must pair 1:1 with --model-dir")
    outputs = [_output_name_for(d, n) for d, n in zip(model_dirs, names)]

    load_environment()
    tracking_uri = (
        args.tracking_uri
        if args.tracking_uri
        else mlflow_tracking_uri(default="file:./mlruns")
    )
    mlflow.set_tracking_uri(tracking_uri)

    repaired = _repair_artifact_locations(tracking_uri)
    if repaired:
        logger.info(
            f"Repointed {repaired} experiment(s) with stale /kaggle artifact "
            f"roots to the local ./mlruns store"
        )

    report: dict[str, dict[str, Any]] = {}
    for model_dir, name in zip(model_dirs, outputs):
        report[name] = export_one(
            model_dir,
            name,
            args.output_dir,
            args.max_length,
            args.opset,
            args.tolerance,
            args.experiment,
        )
    _report_results(report, args.report)


if __name__ == "__main__":
    main()
