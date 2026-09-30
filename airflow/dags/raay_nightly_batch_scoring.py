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
   vs data/scoring/reference/reference.csv -> reports/drift/{{ ds }}.json;
4. ``run_prediction_drift_check`` -- Phase 6 step 2: the predicted class mix vs
   the training label prior, mean confidence, and a triage verdict pairing this
   output-side signal with task 3 -> reports/prediction_drift/{{ ds }}.json;
5. ``evaluate_retrain_trigger`` -- Phase 6 step 3: turns tasks 3+4 into a
   retrain decision (psi_breach / scheduled / manual / none) and records the
   reason in MLflow -> reports/retrain_trigger/{{ ds }}.json. It does not
   retrain: fine-tuning is scripts/kaggle_train_runs.py on a Kaggle GPU.

The DAG itself is thin and stateless on purpose: Airflow owns retries /
scheduling / logs; the heavy lifting stays in the tested batch_score module.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

REPO = "/home/diab/Documents/Raay"


def step(mode: str) -> str:
    """Bash snippet: run one batch_score mode against the project venv."""
    return (
        f"cd {REPO} && uv run python -m raay.inference.batch_score "
        f"--mode {mode} --date " + "{{ ds }}"
    )


def trigger_step() -> str:
    """Bash snippet: run the Phase 6 step 3 retrain trigger.

    A separate module rather than a sixth ``batch_score`` mode: the trigger
    reads two JSON reports and needs no frame, encoder, or ONNX graph, so
    putting it in ``batch_score`` would drag that machinery's argument parsing
    and lazy-loading decisions into a decision that needs none of it.
    """
    return (
        f"cd {REPO} && uv run python -m raay.inference.retrain_trigger --date "
        + "{{ ds }}"
        + " --token-file airflow_runtime/secrets/github_dispatch_token"
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
    # Phase 6 step 2. Deliberately `>>` the input-drift check: this task reads
    # `reports/drift/{{ ds }}.json` to decide whether a class-mix shift came
    # from the inputs or the model, so running it first would classify the day
    # "indeterminate" on every run.
    predict_drift = BashOperator(
        task_id="run_prediction_drift_check",
        bash_command=step("predict-drift"),
    )
    # Phase 6 step 3. Last in the chain: it reads both of the preceding reports,
    # so running it before them would decide `none` on a night whose signals
    # had not been written yet.
    trigger = BashOperator(
        task_id="evaluate_retrain_trigger",
        bash_command=trigger_step(),
    )
    materialize >> score >> drift >> predict_drift >> trigger
