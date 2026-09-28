"""Register the distilled fp32 graph as the canary variant and manage rollout.

Phase 5 step 4: tie the canary rollout to the MLflow registry, exactly like
the baseline winner was registered (self-contained ONNX Model dir + alias).
The canary is the **distilled fp32** ONNX graph -- a genuinely different set
of weights from the Production int8 graph, so the 95/5 nginx split routes to
real, distinct predictions, not the same model with different labels.

Mode: (run from repo root)

    uv run python scripts/canary_promote.py --mode declare   # idempotent
    uv run python scripts/canary_promote.py --mode promote   # health-gated
    uv run python scripts/canary_promote.py --mode rollback  # health-gated

* ``declare`` -- materialize a self-contained mlflow Model dir for the
  distilled fp32 graph (``models/onnx/distilled.onnx`` + external
  ``distilled.onnx.data`` weights) and register it to ``ArabicSentiment``
  with the ``Canary`` alias -- next to the existing int8 ``Production``
  version. Idempotent (purges the prior canary run unless the registry still
  references it).
* ``promote`` -- health-gate the PROD worker (:8000/health) and the CANARY
  front (:8081/health through nginx), then re-point ``Production`` alias at
  the distilled fp32 ``Canary`` version. ``archive_existing_versions=True``
  sends the old int8 to Archived. The flip is what a freshly-started worker
  resolves via ``RAAY_REGISTERED_MODEL=ArabicSentiment`` +
  ``RAAY_ALIAS=Production``.
* ``rollback`` -- health-gate both again, then flip ``Production`` back to
  the int8 version, archiving the distilled one.

The compose canary worker binds the distilled ONNX + external weights
read-only and loads them via ``RAAY_ONNX_PATH`` explicitly, so the 5% slice
serves genuinely different weights regardless of registry state; the registry
flip is the promotion gate. Widening the nginx split itself is a conf change
(``deploy/nginx_canary.conf``) on top.
"""

from __future__ import annotations

import argparse
import platform
import shutil
import urllib.request
import warnings
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import URLError

import onnx
import yaml
from loguru import logger

import mlflow
from raay.config.env import load_environment, mlflow_tracking_uri
from raay.enums.constants import DefaultPaths, Experiments, Models

warnings.filterwarnings("ignore", category=SyntaxWarning)

_TOOL_TAG = "scripts.canary_promote"
_MODEL = Models.REGISTERED_BASELINE.value  # both variants live under ArabicSentiment
_STAGE_CANARY = "Canary"
_STAGE_PRODUCTION = "Production"

# Health gates. The canary worker has no host port (nginx talks to it by
# compose service name), so the :8081 gate goes through the nginx front --
# which also proves the 95/5 split passes health checks.
_CANARY_FRONT_URL = "http://127.0.0.1:8081/health"
_PROD_URL = "http://127.0.0.1:8000/health"


def _materialize_canary_model_dir(run_id: str, onnx_path: str, out_dir: Path) -> str:
    """Build a self-contained mlflow Model dir (onnx flavor) for the distilled fp32 graph.

    Mirrors ``_materialize_model_dir`` in ``scripts/log_variants_mlflow.py``:
    mlflow 3.15 + the relative artifact_location here stages ``mlflow.onnx``
    models into internal ``mlruns/<exp>/models/m-*`` copies that make
    ``runs:/<run_id>/model`` unresolvable, so we assemble the Model dir
    ourselves and register straight from that path.
    """

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


def _health_gate(url: str) -> None:
    """Fail loudly if a worker healthcheck is not 200."""
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            body = resp.read().decode("utf-8", "replace")
            status = resp.status
    except URLError as exc:
        raise SystemExit(f"Health gate FAILED ({url}): {exc}") from exc
    if status != 200:
        raise SystemExit(f"Health gate FAILED ({url}): HTTP {status}")
    logger.info(f"Health gate OK ({url}): {body.strip()}")


def declare(
    client: mlflow.tracking.MlflowClient, experiment: str
) -> mlflow.entities.ModelVersion:
    """Register the distilled fp32 graph as the ``Canary`` alias. Idempotent."""

    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        raise SystemExit(f"Experiment {experiment!r} not found")
    experiment_id = exp.experiment_id

    referenced = {v.run_id for v in client.search_model_versions() if v.run_id}
    prior = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.tool = '{_TOOL_TAG}'",
    )
    to_delete = [r for r in prior if r.info.run_id not in referenced]
    for run in to_delete:
        client.delete_run(run.info.run_id)
    if to_delete:
        logger.info(f"Deleted {len(to_delete)} previous canary run(s)")
    if len(prior) - len(to_delete):
        logger.info(
            f"Kept {len(prior) - len(to_delete)} canary run(s) referenced by the registry"
        )

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name="register-canary-distilled-fp32",
        tags={
            "tool": _TOOL_TAG,
            "stage": "canary",
            "variant": "distilled-fp32",
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        },
    ) as run:
        run_id = run.info.run_id
        logger.info(f"Logged canary run {run_id}")

    model_uri = _materialize_canary_model_dir(
        run_id=run_id,
        onnx_path=DefaultPaths.ONNX_DISTILLED.value,
        out_dir=Path("reports") / "models" / "canary_distilled_fp32_mlflow",
    )
    logger.info(f"Materialized canary MLmodel dir -> {model_uri}")

    registered = mlflow.register_model(model_uri=model_uri, name=_MODEL)
    client.set_registered_model_alias(_MODEL, _STAGE_CANARY, registered.version)
    client.set_model_version_tag(_MODEL, registered.version, "mlflow.run_id", run_id)
    client.set_model_version_tag(
        _MODEL, registered.version, "variant", "distilled-fp32"
    )
    client.set_model_version_tag(_MODEL, registered.version, "stage", "canary")
    logger.info(f"Registered {_MODEL} v{registered.version} -> '{_STAGE_CANARY}' alias")
    return registered


def _canary_version(
    client: mlflow.tracking.MlflowClient,
) -> mlflow.entities.ModelVersion:
    versions = [
        v
        for v in client.search_model_versions(f"name = '{_MODEL}'")
        if v.tags.get("variant") == "distilled-fp32"
    ]
    if not versions:
        raise SystemExit(
            f"No distilled-fp32 version under {_MODEL}. Run `--mode declare` first."
        )
    return versions[0]


def promote(client: mlflow.tracking.MlflowClient) -> None:
    """Health-gate both workers, then point Production at the canary distilled fp32."""
    _health_gate(_PROD_URL)
    _health_gate(_CANARY_FRONT_URL)

    canary = _canary_version(client)
    client.transition_model_version_stage(
        name=_MODEL,
        version=canary.version,
        stage=_STAGE_PRODUCTION,
        archive_existing_versions=True,
    )
    client.set_registered_model_alias(_MODEL, _STAGE_PRODUCTION, canary.version)
    logger.info(
        f"Promoted {_MODEL} v{canary.version} -> '{_STAGE_PRODUCTION}' alias "
        f"(int8 -> Archived)"
    )


def rollback(client: mlflow.tracking.MlflowClient) -> None:
    """Health-gate both workers, then flip Production back to the archived int8."""
    _health_gate(_PROD_URL)
    _health_gate(_CANARY_FRONT_URL)

    int8_versions = [
        v
        for v in client.search_model_versions(f"name = '{_MODEL}'")
        if v.tags.get("variant") == "onnx-int8"
    ]
    if not int8_versions:
        raise SystemExit(
            f"No onnx-int8 version under {_MODEL} to roll back to. "
            f"Run `scripts/benchmark.py` + `scripts/log_variants_mlflow.py` first."
        )
    int8 = int8_versions[0]
    client.transition_model_version_stage(
        name=_MODEL,
        version=int8.version,
        stage=_STAGE_PRODUCTION,
        archive_existing_versions=True,
    )
    client.set_registered_model_alias(_MODEL, _STAGE_PRODUCTION, int8.version)
    logger.info(
        f"Rolled back {_MODEL} '{_STAGE_PRODUCTION}' alias -> int8 v{int8.version}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["declare", "promote", "rollback"], required=True
    )
    parser.add_argument("--experiment", default=Experiments.TRAINING.value)
    parser.add_argument("--tracking-uri", default=None)
    args = parser.parse_args()

    load_environment()
    tracking_uri = (
        args.tracking_uri
        if args.tracking_uri
        else mlflow_tracking_uri(default="file:./mlruns")
    )
    mlflow.set_tracking_uri(tracking_uri)

    if args.mode == "declare":
        registered = declare(mlflow.tracking.MlflowClient(), args.experiment)
        print(f"Declared canary: {_MODEL} v{registered.version} (alias 'Canary')")
    elif args.mode == "promote":
        promote(mlflow.tracking.MlflowClient())
        print(f"Promoted canary -> {_MODEL} 'Production' alias")
    elif args.mode == "rollback":
        rollback(mlflow.tracking.MlflowClient())
        print(f"Rolled back {_MODEL} 'Production' alias -> int8")


if __name__ == "__main__":
    main()
