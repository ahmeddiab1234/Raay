"""One MLflow run per benchmark variant, and idempotent cleanup of prior runs.

Each run carries a ``stage`` tag (baseline/distilled/fp32/int8) plus the decision
metrics read straight from ``reports/benchmark_table.csv`` -- this is the queryable
record behind the promotion decision.

Idempotency has a hard constraint: a previous variant run is purged **unless** a
registered model version still references it. Deleting a referenced run orphans
the version that points at it, which is worse than a duplicated run.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
from loguru import logger

TOOL_TAG = "scripts.log_variants_mlflow"


def purge_prior_runs(client: Any, experiment_id: str, tool_tag: str = TOOL_TAG) -> None:
    """Delete this tool's prior runs, keeping any the registry still references."""
    referenced = {v.run_id for v in client.search_model_versions() if v.run_id}
    prior = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.tool = '{tool_tag}'",
    )
    to_delete = [r for r in prior if r.info.run_id not in referenced]
    for run in to_delete:
        client.delete_run(run.info.run_id)
    if to_delete:
        logger.info(f"Deleted {len(to_delete)} previous variant run(s)")
    if len(prior) - len(to_delete):
        logger.info(
            f"Kept {len(prior) - len(to_delete)} run(s) referenced by the registry"
        )


def log_variant_runs(
    experiment_id: str,
    payloads: dict[str, dict[str, str]],
    stage_by_variant: dict[str, str],
    benchmark_csv: str,
    tool_tag: str = TOOL_TAG,
) -> dict[str, str]:
    """Log one run per variant; returns ``{variant: run_id}``."""
    generated_at = datetime.now(UTC).isoformat(timespec="seconds")
    run_ids: dict[str, str] = {}
    for name, stage in stage_by_variant.items():
        payload = payloads[name]
        tags = {
            "tool": tool_tag,
            "stage": stage,
            "backend": str(payload["backend"]),
            "generated_at": generated_at,
        }
        params = {
            "variant": name,
            "size_vs_baseline": str(payload["size_vs_baseline"]),
            "note": str(payload["note"]),
            "benchmark_source": Path(benchmark_csv).name,
        }
        metrics = {
            "accuracy": float(payload["accuracy"]),
            "f1_macro": float(payload["f1_macro"]),
            "latency_p50": float(payload["batch1_p50_ms"]),
            "latency_p95": float(payload["batch1_p95_ms"]),
            "latency_batch32_p95": float(payload["batch32_p95_ms"]),
            "size_mb": float(payload["size_mb"]),
        }
        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=f"register-variant-{name}",
            tags=tags,
        ) as run:
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            run_ids[name] = run.info.run_id
            logger.info(
                f"Logged {name} (stage={stage}): acc={metrics['accuracy']:.4f} "
                f"f1={metrics['f1_macro']:.4f} latency_p95={metrics['latency_p95']:.1f}ms "
                f"size={metrics['size_mb']:.1f}MB"
            )
    return run_ids
