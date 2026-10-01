"""Build a self-contained mlflow Model dir for the canary graph.

Mirrors ``_materialize_model_dir`` in ``scripts/log_variants_mlflow.py``: mlflow
3.15 with this project's relative ``artifact_location`` stages ``mlflow.onnx``
models into internal ``mlruns/<exp>/models/m-*`` copies that make
``runs:/<run_id>/model`` unresolvable, so the Model dir is assembled by hand and
registered straight from that path. The run<->model backlink is then kept as an
``mlflow.run_id`` version tag.
"""

from __future__ import annotations

import platform
import shutil
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import onnx
import yaml


def _materialize_canary_model_dir(run_id: str, onnx_path: str, out_dir: Path) -> str:
    """Build a self-contained mlflow Model dir (onnx flavor) for the distilled fp32 graph."""
    out_dir = out_dir.resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # The graph uses external weights (distilled.onnx.data) next to it -- copy
    # both so the registered artifact is self-contained and locally loadable.
    shutil.copy(onnx_path, out_dir / "model.onnx")
    external = Path(onnx_path).with_suffix(".onnx.data")
    if external.exists():
        shutil.copy(external, out_dir / "model.onnx.data")

    (out_dir / "requirements.txt").write_text(
        f"mlflow=={mlflow.__version__}\nonnxruntime>=1.18.0\nnumpy\n"
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
