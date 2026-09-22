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
import platform
import shutil
import warnings
from datetime import UTC, datetime
from pathlib import Path

import onnx
import yaml
from loguru import logger

import mlflow
from raay.config.env import load_environment, mlflow_tracking_uri
from raay.enums.constants import DefaultPaths, Experiments

warnings.filterwarnings("ignore", category=SyntaxWarning)

_TOOL_TAG = "scripts.log_variants_mlflow"
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


def _materialize_model_dir(run_id: str, onnx_path: str, out_dir: Path) -> str:
    """Build a self-contained mlflow Model dir (onnx flavor) and return its path.

    mlflow.onnx.log_model stages into internal ``mlruns/<exp>/models/m-*``
    copies on this 3.15 + relative-artifact-location setup (never the active
    run's artifact dir), which leaves ``runs:/<run_id>/model`` unresolvable.
    Instead, assemble the Model dir ourselves from the already-validated graph
    and register straight from that path.
    """

    out_dir = out_dir.resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    shutil.copy(onnx_path, out_dir / "model.onnx")
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

    # Idempotency: purge previous variant runs UNLESS a registered model
    # version still references them (deleting those would orphan the version).
    referenced = {v.run_id for v in client.search_model_versions() if v.run_id}
    prior = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.tool = '{_TOOL_TAG}'",
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

    generated_at = datetime.now(UTC).isoformat(timespec="seconds")
    run_ids: dict[str, str] = {}
    for name, stage in _STAGE_BY_VARIANT.items():
        payload = payloads[name]
        tags = {
            "tool": _TOOL_TAG,
            "stage": stage,
            "backend": str(payload["backend"]),
            "generated_at": generated_at,
        }
        params = {
            "variant": name,
            "size_vs_baseline": str(payload["size_vs_baseline"]),
            "note": str(payload["note"]),
            "benchmark_source": Path(args.benchmark_csv).name,
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

    # Materialize a self-contained MLmodel dir for the winning variant (the
    # int8 graph = the baseline's tokenizer+weights compressed) and register
    # straight from it, mirroring the baseline's register + promote flow.
    winner_row = payloads[args.winner]
    winner_run = run_ids[args.winner]
    winner_onnx = _ONNX_PATH_BY_VARIANT[args.winner]
    model_uri = _materialize_model_dir(
        run_id=winner_run,
        onnx_path=winner_onnx,
        out_dir=Path("reports") / "models" / f"{args.winner}_mlflow",
    )
    logger.info(f"Materialized MLmodel dir for {args.winner} -> {model_uri}")

    registered = mlflow.register_model(model_uri=model_uri, name=args.registered_name)
    client.transition_model_version_stage(
        name=args.registered_name,
        version=registered.version,
        stage=args.stage,
        archive_existing_versions=True,
    )
    client.set_model_version_tag(
        args.registered_name, registered.version, "variant", args.winner
    )
    client.set_model_version_tag(
        args.registered_name, registered.version, "mlflow.run_id", winner_run
    )
    client.set_model_version_tag(
        args.registered_name,
        registered.version,
        "stage",
        _STAGE_BY_VARIANT[args.winner],
    )
    logger.info(
        f"Registered {args.registered_name} v{registered.version} "
        f"(variant={args.winner}) -> {args.stage}"
    )

    print(
        "\nWinner decision metrics (from reports/benchmark_table.csv):\n"
        f"  {args.winner}: accuracy={winner_row['accuracy']} "
        f"f1_macro={winner_row['f1_macro']} "
        f"latency_p95={winner_row['batch1_p95_ms']}ms "
        f"size_mb={winner_row['size_mb']}"
    )
    print(f"Registered {args.registered_name} v{registered.version} -> {args.stage}")


if __name__ == "__main__":
    main()
