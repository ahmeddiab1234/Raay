"""Bar chart of ``f1_macro`` per MLflow run in the training experiment."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from loguru import logger


def plot_mlflow_comparison(
    experiment_name: str,
    output_path: str,
    tracking_uri: str | None = None,
) -> None:
    """Bar chart of f1_macro per MLflow run in the training experiment."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        logger.warning(f"Experiment {experiment_name!r} not found; skipping plot.")
        return
    runs = client.search_runs(experiment_ids=[exp.experiment_id])
    if not runs:
        logger.warning(f"No runs found for {experiment_name!r}; skipping plot.")
        return

    labels: list[str] = []
    f1s: list[float] = []
    for run in runs:
        metrics = run.data.metrics
        if "f1_macro" not in metrics and "eval_f1_macro" not in metrics:
            continue
        key = "f1_macro" if "f1_macro" in metrics else "eval_f1_macro"
        f1s.append(float(metrics[key]))
        params = run.data.params
        lr = params.get("learning_rate", "?")
        bs = params.get("batch_size", "?")
        labels.append(
            f"{run.data.tags.get('mlflow.runName', run.info.run_id[:8])}\nlr={lr} bs={bs}"
        )

    if not f1s:
        logger.warning("No runs had f1_macro metrics; skipping plot.")
        return

    plt.figure(figsize=(max(8, 0.9 * len(labels)), 5))
    bars = plt.bar(range(len(f1s)), f1s, color="#4C72B0")
    best = int(np.argmax(f1s))
    bars[best].set_color("#55A868")
    plt.xticks(range(len(f1s)), labels, rotation=0)
    plt.ylabel("f1_macro")
    plt.title(f"MLflow run comparison — {experiment_name}")
    plt.ylim(0, 1.0)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120)
    plt.close()
    logger.info(f"Wrote comparison plot: {output_path}")
