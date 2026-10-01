"""Where the graph comes from: env override first, then the MLflow alias.

Split out of ``runtime`` because the resolution logic is the one part that is
BentoML-*and* MLflow-aware, while the rest of ``runtime`` is pure
"text -> probabilities". Keeping them apart means the hot path has no MLflow
import at module scope (the ``import mlflow`` below is deliberately inside the
function).

``RAAY_ONNX_PATH`` wins when set -- that is what ``benchmark.py`` /
``locust_run.py`` and the released container use, so a self-contained image does
not need a tracking server at boot. Otherwise the named model's ``Production``
alias is resolved, so a worker picks up a promoted model on restart.

The two failure modes raise **loudly on purpose**. An unresolvable alias means
``/health`` still returns 200 while ``/predict`` 500s with
``Registered Model with name=ArabicSentiment not found`` -- a container that
looks healthy and serves nothing. That exact bug is why ``RAAY_ONNX_PATH`` is
baked into the bento image, so this path is not the one production takes.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from raay.config.env import load_environment, mlflow_tracking_uri
from raay.enums.constants import Models

DEFAULT_REGISTERED_MODEL = Models.REGISTERED_BASELINE.value
DEFAULT_ALIAS = "Production"


def download_artifacts(uri: str) -> Any:
    """Resolve a model URI to a local directory via the .env MLflow store."""
    import mlflow

    load_environment()
    mlflow.set_tracking_uri(mlflow_tracking_uri(default="file:./mlruns"))
    return mlflow.artifacts.download_artifacts(uri)


def resolve_onnx_path(
    env_get: Callable[..., Any], *, download: Callable[..., Any] = download_artifacts
) -> tuple[str, str, str]:
    """Return ``(source_label, onnx_path, detail)`` for the graph to serve."""
    explicit = env_get("RAAY_ONNX_PATH")
    if explicit:
        return "RAAY_ONNX_PATH", str(explicit), str(explicit)
    registered_model = env_get("RAAY_REGISTERED_MODEL", DEFAULT_REGISTERED_MODEL)
    alias = env_get("RAAY_ALIAS", DEFAULT_ALIAS)
    uri = f"models:/{registered_model}/{alias}"
    try:
        model_dir = Path(str(download(uri)))
    except Exception as exc:
        raise RuntimeError(
            f"Could not resolve {uri}: {exc}. Register + promote a model first "
            f"(uv run python scripts/log_variants_mlflow.py) or set "
            f"RAAY_ONNX_PATH to a local graph."
        ) from exc
    onnx_path = model_dir / "model.onnx"
    if not onnx_path.exists():
        raise RuntimeError(f"{uri} resolved to {model_dir} but has no model.onnx")
    return uri, str(onnx_path), str(model_dir)
