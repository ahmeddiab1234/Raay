"""Materialize a self-contained MLflow Model dir for an ONNX variant.

Why this exists instead of ``mlflow.onnx.log_model``: on MLflow 3.15 with a
relative ``artifact_location`` (which is what this project has), logging stages
the model into internal ``mlruns/<exp>/models/m-*`` copies rather than the
active run's artifact dir, so ``register_model(model_uri="runs:/<id>/model")``
resolves to nothing. Assembling the ``onnx``-flavored Model dir by hand lets
registration point at a real directory.
"""

from __future__ import annotations

import platform
import shutil
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import onnx
import yaml

_REQUIREMENTS = "onnxruntime>=1.18.0\nnumpy"


def materialize_model_dir(run_id: str, onnx_path: str, out_dir: Path) -> str:
    """Build a self-contained mlflow Model dir (onnx flavor) and return its path."""
    out_dir = out_dir.resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    shutil.copy(onnx_path, out_dir / "model.onnx")
    (out_dir / "requirements.txt").write_text(
        f"mlflow=={mlflow.__version__}\n{_REQUIREMENTS}"
    )
    (out_dir / "python_env.yaml").write_text(
        "python: 3.12.3\n"
        "build_dependencies:\n"
        "- pip\n"
        "- setuptools==84.0.0\n"
        "- wheel\n"
        "dependencies:\n"
        "- -r requirements.txt\n"
    )
    (out_dir / "conda.yaml").write_text(
        "channels:\n"
        "- conda-forge\n"
        "dependencies:\n"
        "- python=3.12.3\n"
        "- pip\n"
        "- pip:\n"
        f"  - mlflow=={mlflow.__version__}\n"
        "  - onnxruntime>=1.18.0\n"
        "  - numpy\n"
        "name: mlflow-env\n"
    )
    mlmodel = {
        "artifact_path": str(out_dir),
        "flavors": {
            "onnx": {
                "code": None,
                "data": "model.onnx",
                "onnx_session_options": None,
                "onnx_version": onnx.__version__,
                "providers": ["CPUExecutionProvider"],
            },
            "python_function": {
                "data": "model.onnx",
                "env": {"conda": "conda.yaml", "virtualenv": "python_env.yaml"},
                "loader_module": "mlflow.onnx",
                "python_version": platform.python_version(),
            },
        },
        "mlflow_version": mlflow.__version__,
        "run_id": run_id,
        "utc_time_created": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S.%f"),
    }
    (out_dir / "MLmodel").write_text(yaml.safe_dump(mlmodel, sort_keys=False))
    return str(out_dir)
