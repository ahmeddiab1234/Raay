"""``--mode declare``: register the distilled fp32 graph under the Canary alias.

Idempotent by purging unreferenced prior runs rather than by refusing to run
twice. A run referenced by a registered model version is kept, because deleting
it would leave a live version pointing at a run that no longer exists.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import canary_model
import mlflow
from canary_registry import _MODEL, _STAGE_CANARY
from loguru import logger

from raay.enums.constants import DefaultPaths

_TOOL_TAG = "scripts.canary_promote"


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

    model_uri = canary_model._materialize_canary_model_dir(
        run_id=run_id,
        onnx_path=DefaultPaths.ONNX_DISTILLED.value,
        out_dir=Path("reports") / "models" / "canary_distilled_fp32_mlflow",
    )
    logger.info(f"Materialized canary MLmodel dir -> {model_uri}")

    registered = mlflow.register_model(model_uri=model_uri, name=_MODEL)
    client.set_registered_model_alias(_MODEL, _STAGE_CANARY, str(registered.version))
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
