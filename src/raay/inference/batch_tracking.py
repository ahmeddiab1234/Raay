"""MLflow logging for the nightly batch/drift/trigger runs.

One helper, one experiment (``raay_batch``), because the four Airflow tasks
report into a single time series and a reader should be able to compare a
drift PSI against the confidence mean logged by the same night.
"""

from __future__ import annotations

from loguru import logger

from raay.config.env import mlflow_tracking_uri
from raay.enums.constants import Experiments


def _log_run(
    run_name: str,
    metrics: dict[str, float],
    artifacts: list[tuple[str, str]],
    run_type: str,
) -> None:
    """Idempotently log a batch/drift run to the ``raay_batch`` experiment."""
    import mlflow

    tracking_uri = mlflow_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(Experiments.BATCH.value)
    if experiment is None:
        experiment_id = mlflow.create_experiment(Experiments.BATCH.value)
    else:
        experiment_id = experiment.experiment_id
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        mlflow.set_tag("run_type", run_type)
        for key, value in metrics.items():
            mlflow.log_metric(key, float(value))
        for artifact_path, local_path in artifacts:
            mlflow.log_artifact(local_path, artifact_path=artifact_path)
        logger.info(f"Logged raay_batch run {run.info.run_id}")
