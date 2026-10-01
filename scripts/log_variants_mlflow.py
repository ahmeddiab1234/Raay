"""Log each model variant as an MLflow run and register the best tradeoff.

Phase 3 step 7: give the variant comparison an MLflow presence. One run per
variant in ``raay_training`` carrying a ``stage`` tag and the decision metrics
accuracy / f1_macro / latency_p95 / size_mb, all sourced from the canonical
``reports/benchmark_table.csv``. Then register the winning variant the same
way the baseline was registered on Kaggle (``mlflow.register_model`` +
``transition_model_version_stage``) and promote it to ``Production``.

Run from the repo root:

    uv run python scripts/log_variants_mlflow.py            # winner = onnx-int8
    uv run python scripts/log_variants_mlflow.py --winner onnx-int8 \
        --registered-name ArabicSentiment --stage Production

Winner selection: the INT8 graph keeps baseline accuracy (0.8503 vs 0.8492,
f1_macro within 0.004) at 3.6x the speed and 25% of the size — the classic
best tradeoff. Pass ``--winner`` to override; only ONNX variants can be
registered here (torch checkpoints should use the train run's transformers
flavor instead). Re-running deletes the previous set of variant runs, so the
script stays idempotent. A TRT engine, once built, is logged the same way
with ``stage=trt``.
"""

from __future__ import annotations

import argparse
import csv
import warnings

import mlflow
from variants_mlflow_registry import register_and_promote
from variants_mlflow_runs import TOOL_TAG as _TOOL_TAG
from variants_mlflow_runs import log_variant_runs, purge_prior_runs

from raay.config.env import load_environment, mlflow_tracking_uri
from raay.enums.constants import DefaultPaths, Experiments

warnings.filterwarnings("ignore", category=SyntaxWarning)

_BENCHMARK_CSV = "reports/benchmark_table.csv"
_ONNX_PATH_BY_VARIANT = {
    "onnx-fp32": DefaultPaths.ONNX_MODEL.value,
    "onnx-int8": DefaultPaths.ONNX_INT8_MODEL.value,
}
_STAGE_BY_VARIANT = {
    "baseline-torch": "baseline",
    "distilled-torch": "distilled",
    "onnx-fp32": "fp32",
    "onnx-int8": "int8",
}


def _load_rows(csv_path: str) -> list[dict[str, float | str]]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--winner",
        default="onnx-int8",
        help="Variant to register + promote; must be an ONNX variant.",
    )
    parser.add_argument("--registered-name", default="ArabicSentiment")
    parser.add_argument("--stage", default="Production")
    parser.add_argument("--benchmark-csv", default=_BENCHMARK_CSV)
    parser.add_argument("--experiment", default=Experiments.TRAINING.value)
    parser.add_argument("--tracking-uri", default=None)
    args = parser.parse_args()

    if args.winner not in _ONNX_PATH_BY_VARIANT:
        raise SystemExit(
            f"--winner {args.winner!r} is not a registerable ONNX variant; "
            f"choose one of {sorted(_ONNX_PATH_BY_VARIANT)}."
        )

    load_environment()
    tracking_uri = (
        args.tracking_uri
        if args.tracking_uri
        else mlflow_tracking_uri(default="file:./mlruns")
    )
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    rows = _load_rows(args.benchmark_csv)
    payloads = {row["variant"]: row for row in rows}
    missing = [name for name in _STAGE_BY_VARIANT if name not in payloads]
    if missing:
        raise SystemExit(
            f"Benchmark table missing variant rows: {sorted(missing)}. "
            f"Run `uv run python scripts/benchmark.py` first."
        )

    exp = client.get_experiment_by_name(args.experiment)
    if exp is None:
        raise SystemExit(f"Experiment {args.experiment!r} not found")
    experiment_id = exp.experiment_id

    # Note: this experiment carries a relative artifact_location that mlflow 3.x
    # can no longer repoint and that leaves ``runs:/<run_id>/model``
    # unresolvable, so registration uses a locally materialized Model dir
    # (see :func:`_materialize_model_dir`). Re-running the script deletes the
    # previous variant runs (keeping any referenced by the registry) and
    # registers a fresh version.

    # This experiment carries a relative artifact_location that mlflow 3.x can
    # no longer repoint and that leaves ``runs:/<run_id>/model`` unresolvable,
    # so registration uses a locally materialized Model dir (variants_mlflow_model).
    purge_prior_runs(client, experiment_id, _TOOL_TAG)
    run_ids = log_variant_runs(
        experiment_id, payloads, _STAGE_BY_VARIANT, args.benchmark_csv, _TOOL_TAG
    )
    version = register_and_promote(
        client,
        winner=args.winner,
        winner_run=run_ids[args.winner],
        onnx_path=_ONNX_PATH_BY_VARIANT[args.winner],
        registered_name=args.registered_name,
        stage=args.stage,
        stage_by_variant=_STAGE_BY_VARIANT,
    )
    winner_row = payloads[args.winner]
    print(
        "\nWinner decision metrics (from reports/benchmark_table.csv):\n"
        f"  {args.winner}: accuracy={winner_row['accuracy']} "
        f"f1_macro={winner_row['f1_macro']} "
        f"latency_p95={winner_row['batch1_p95_ms']}ms "
        f"size_mb={winner_row['size_mb']}"
    )
    print(f"Registered {args.registered_name} v{version} -> {args.stage}")


if __name__ == "__main__":
    main()
