"""CLI for the ONNX export job."""

from __future__ import annotations

import argparse
from typing import Any

import mlflow
from loguru import logger

from raay.config.env import load_environment, mlflow_tracking_uri
from raay.enums.constants import DefaultPaths, Experiments
from raay.inference.export_artifacts import (
    _output_name_for,
    _repair_artifact_locations,
    _report_results,
)
from raay.inference.export_one import export_one

_DESCRIPTION = "Export the fine-tuned AraBERT checkpoints to ONNX and validate parity."


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=_DESCRIPTION)
    parser.add_argument(
        "--model-dir",
        action="append",
        default=None,
        help=(
            "Checkpoint dir to export (repeatable). Defaults to both the "
            "baseline and the distilled checkpoints."
        ),
    )
    parser.add_argument(
        "--name",
        action="append",
        default=None,
        help=(
            "ONNX output basename (repeatable, pairs positionally with "
            "--model-dir). Default: baseline->model, distilled->distilled."
        ),
    )
    parser.add_argument("--output-dir", default="models/onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--report", default="reports/onnx_parity.json")
    parser.add_argument("--experiment", default=Experiments.TRAINING.value)
    parser.add_argument("--tracking-uri", default=None)
    return parser


def _resolve_targets(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> tuple[list[str], list[str]]:
    model_dirs = args.model_dir or [
        DefaultPaths.BASELINE_MODEL.value,
        DefaultPaths.DISTILLED_MODEL.value,
    ]
    names = args.name or [None] * len(model_dirs)
    if len(names) != len(model_dirs):
        parser.error("--name must pair 1:1 with --model-dir")
    outputs = [_output_name_for(d, n) for d, n in zip(model_dirs, names)]
    return model_dirs, outputs


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    model_dirs, outputs = _resolve_targets(parser, args)

    load_environment()
    tracking_uri = (
        args.tracking_uri
        if args.tracking_uri
        else mlflow_tracking_uri(default="file:./mlruns")
    )
    mlflow.set_tracking_uri(tracking_uri)

    repaired = _repair_artifact_locations(tracking_uri)
    if repaired:
        logger.info(
            f"Repointed {repaired} experiment(s) with stale /kaggle artifact "
            f"roots to the local ./mlruns store"
        )

    report: dict[str, dict[str, Any]] = {}
    for model_dir, name in zip(model_dirs, outputs):
        report[name] = export_one(
            model_dir,
            name,
            args.output_dir,
            args.max_length,
            args.opset,
            args.tolerance,
            args.experiment,
        )
    _report_results(report, args.report)
