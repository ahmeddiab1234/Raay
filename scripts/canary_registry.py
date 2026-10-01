"""Registry reads and the one Production flip the canary tool performs.

Rollback is the only place this tool touches the Production alias, and it only
ever moves it *back* to the version recorded when shadow started. Advancing to
``full`` does **not** flip it here: that goes through the Phase 6 offline gate
(``canary_offline``), which is the only code allowed to move the alias forward.
"""

from __future__ import annotations

import mlflow
from canary_phases import _EDGES, _PHASES  # noqa: F401 - re-exported for callers
from loguru import logger

from raay.enums.constants import Models

_MODEL = Models.REGISTERED_BASELINE.value  # both variants live under ArabicSentiment
_STAGE_CANARY = "Canary"
_STAGE_PRODUCTION = "Production"


def _production_version(client: mlflow.tracking.MlflowClient) -> int | None:
    """The version the Production alias points at right now (pre-rollout)."""
    try:
        version = client.get_model_version_by_alias(_MODEL, _STAGE_PRODUCTION)
    except Exception:  # noqa: BLE001 - no alias yet means first promotion, nothing to roll back to
        return None
    if not version:
        return None
    try:
        return int(version)
    except (TypeError, ValueError):
        return None


def _int8_version(client: mlflow.tracking.MlflowClient) -> int:
    versions = [
        v
        for v in client.search_model_versions(f"name = '{_MODEL}'")
        if v.tags.get("variant") == "onnx-int8"
    ]
    if not versions:
        raise SystemExit(
            f"No onnx-int8 version under {_MODEL} to roll back to. "
            f"Run `scripts/benchmark.py` + `scripts/log_variants_mlflow.py` first."
        )
    return int(versions[0].version)


def _flip_production(client: mlflow.tracking.MlflowClient, version: int) -> None:
    client.transition_model_version_stage(
        name=_MODEL,
        version=version,
        stage=_STAGE_PRODUCTION,
        archive_existing_versions=True,
    )
    client.set_registered_model_alias(_MODEL, _STAGE_PRODUCTION, str(version))
    logger.info(f"'{_STAGE_PRODUCTION}' alias -> v{version}")
