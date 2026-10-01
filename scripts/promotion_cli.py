"""Argument parsing and the exit-code contract for the promotion gate.

Exit codes are load-bearing for the workflow: ``0`` gates passed, ``1`` a gate
failed, ``2`` the gate could not run at all. ``promote.yml`` reads all three, and
the third is deliberately distinct -- a runner that cannot reach the registry
should not look like a candidate that failed its metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from promotion_flow import promote
from promotion_report import format_table, report_path, write_report
from promotion_types import Config, PromotionError

from raay.enums.constants import DefaultPaths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gate a candidate model")
    parser.add_argument("--candidate-version", required=True)
    parser.add_argument(
        "--candidate-onnx",
        default="models/onnx/model_int8.onnx",
        help="graph to gate (measured, not trusted)",
    )
    parser.add_argument(
        "--production-onnx",
        default="models/onnx/model_int8.onnx",
        help="the graph currently in Production, for the comparison",
    )
    parser.add_argument("--tokenizer-dir", default=DefaultPaths.BASELINE_MODEL.value)
    parser.add_argument("--floor-report", default=DefaultPaths.EVAL_BASELINE.value)
    parser.add_argument("--parity-report", default="reports/onnx_int8_parity.json")
    parser.add_argument("--test-split", default=DefaultPaths.TEST_SPLIT.value)
    parser.add_argument("--report-dir", default="reports")
    # Kept as a flag rather than hard-coded so the lock the split is checked
    # against is always explicit at the call site.
    parser.add_argument("--dvc-lock", default="dvc.lock")
    parser.add_argument("--max-size-mb", type=float, default=None)
    parser.add_argument("--f1-tolerance", type=float, default=0.005)
    parser.add_argument("--floor-tolerance", type=float, default=0.01)
    parser.add_argument("--recall-tolerance", type=float, default=0.02)
    parser.add_argument("--latency-regression", type=float, default=0.10)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument(
        "--dialect-breakdown",
        action="store_true",
        help="also report per-dialect metrics (roughly doubles the runtime)",
    )
    parser.add_argument(
        "--skip-registry",
        action="store_true",
        help="evaluate and gate only, touching no alias (local rehearsal)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="evaluate and gate, then stop before moving Production",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = Config(
        test_split=Path(args.test_split),
        tokenizer_dir=args.tokenizer_dir,
        floor_report=Path(args.floor_report),
        parity_report=Path(args.parity_report),
        max_size_mb=args.max_size_mb,
        f1_tolerance=args.f1_tolerance,
        floor_tolerance=args.floor_tolerance,
        recall_tolerance=args.recall_tolerance,
        latency_regression=args.latency_regression,
        eval_limit=args.eval_limit,
        dialect_breakdown=args.dialect_breakdown,
        report_dir=Path(args.report_dir),
        dvc_lock=Path(args.dvc_lock),
    )

    client = None
    if not args.skip_registry:
        try:
            import mlflow

            from raay.config.env import load_environment, mlflow_tracking_uri

            load_environment()
            if mlflow_tracking_uri().startswith("file:"):
                import os

                os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
            client = mlflow.tracking.MlflowClient()
        except Exception as exc:  # pragma: no cover - environment dependent
            raise PromotionError(f"could not reach the MLflow registry: {exc}") from exc

    try:
        decision = promote(
            cfg,
            args.candidate_version,
            args.candidate_onnx,
            args.production_onnx,
            client=client,
            dry_run=args.dry_run,
        )
    except PromotionError as exc:
        print(f"promotion gate could not run: {exc}")
        return 2

    path = report_path(cfg, args.candidate_version)
    write_report(path, decision.payload)

    print(format_table(decision.gates))
    print()
    if decision.promoted:
        verdict = "PROMOTED"
    elif decision.payload["decision"] == "passed_not_promoted":
        verdict = "PASSED (not promoted)"
    else:
        verdict = "REJECTED"
    print(f"candidate v{args.candidate_version}: {verdict}")
    print(f"report: {path}")
    if decision.failed:
        for gate in decision.failed:
            print(
                f"  FAIL {gate.name}: observed={gate.observed} "
                f"threshold={gate.threshold} ({gate.detail})"
            )
        print("the Production alias was not touched")
    return 0 if not decision.failed else 1
