"""Register the winning ONNX variant and promote it to the serving alias.

Mirrors the baseline's Kaggle register+promote flow, with one addition: the
``Production`` **alias** is set next to the (deprecated) stage, because
``serve.py`` resolves ``models:/ArabicSentiment/Production`` as an alias and would
otherwise keep serving the previous version after a restart.

Note this experiment carries a relative ``artifact_location`` that MLflow 3.x can
no longer repoint, so registration goes through a locally materialized Model dir
(``variants_mlflow_model``) rather than ``runs:/<id>/model``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
from loguru import logger
from variants_mlflow_model import materialize_model_dir


def register_and_promote(
    client: Any,
    winner: str,
    winner_run: str,
    onnx_path: str,
    registered_name: str,
    stage: str,
    stage_by_variant: dict[str, str],
) -> str:
    """Register the winner, promote it, tag it; returns the new version string."""
    model_uri = materialize_model_dir(
        run_id=winner_run,
        onnx_path=onnx_path,
        out_dir=Path("reports") / "models" / f"{winner}_mlflow",
    )
    logger.info(f"Materialized MLmodel dir for {winner} -> {model_uri}")

    registered = mlflow.register_model(model_uri=model_uri, name=registered_name)
    client.transition_model_version_stage(
        name=registered_name,
        version=registered.version,
        stage=stage,
        archive_existing_versions=True,
    )
    client.set_registered_model_alias(registered_name, stage, registered.version)
    client.set_model_version_tag(registered_name, registered.version, "variant", winner)
    client.set_model_version_tag(
        registered_name, registered.version, "mlflow.run_id", winner_run
    )
    client.set_model_version_tag(
        registered_name, registered.version, "stage", stage_by_variant[winner]
    )
    logger.info(
        f"Registered {registered_name} v{registered.version} "
        f"(variant={winner}) -> {stage}"
    )
    return str(registered.version)
