"""Nightly batch re-scoring DAG (Phase 5 step 3).

Runs every night at 03:00 on the local Airflow service (LocalExecutor,
SQLite; AIRFLOW_HOME=airflow_runtime). Three BashOperators shell out to the
raay project venv via ``uv run``:

1. ``materialize_daily_input``  -- sample a day's worth of reviews (>=1000) into
   data/scoring/input/{{ ds }}.csv;
2. ``score_daily_batch``        -- score that day in large in-process INT8 ONNX
   batches -> data/scoring/output/{{ ds }}.csv; logs to the ``raay_batch``
   MLflow experiment;
3. ``run_drift_check``          -- Evidently PSI (predicted_label, positive)
   vs data/scoring/reference/reference.csv -> reports/drift/{{ ds }}.json.

The DAG itself is thin and stateless on purpose: Airflow owns retries /
scheduling / logs; the heavy lifting stays in the tested batch_score module.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from airflow.operators.bash import BashOperator

from airflow import DAG

REPO = "/home/diab/Documents/Raay"


def step(mode: str) -> str:
    """Bash snippet: run one batch_score mode against the project venv."""
    return (
        f"cd {REPO} && uv run python -m raay.inference.batch_score "
        f"--mode {mode} --date " + "{{ ds }}"
    )


default_args = {
    "owner": "raay",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "start_date": datetime(2026, 1, 1, tzinfo=UTC),
}


with DAG(
    dag_id="raay_nightly_batch_scoring",
    description="Nightly batch re-scoring of a day's reviews + Evidently PSI drift",
    schedule="0 3 * * *",
    default_args=default_args,
    catchup=False,
    tags=["raay", "batch-scoring"],
    doc_md=__doc__,
) as dag:
    materialize = BashOperator(
        task_id="materialize_daily_input",
        bash_command=step("make-input"),
    )
    score = BashOperator(
        task_id="score_daily_batch",
        bash_command=step("score"),
    )
    drift = BashOperator(
        task_id="run_drift_check",
        bash_command=step("drift"),
    )
    materialize >> score >> drift
