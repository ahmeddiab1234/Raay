"""Merge Kaggle-session MLflow runs into the local tracking store.

Kaggle training writes runs to a session-local file store (e.g.
``~/.mlflow`` or ``mlruns`` copied from the notebook working dir). Copy that
directory into this repo and this script will merge every experiment/run/metri
 into the local tracking store so that ``mlflow ui`` shows runs from both the
local pipeline and the GPU sessions.

Usage:
    uv run python scripts/merge_mlflow.py --source path/to/session/mlruns
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from loguru import logger

LOCAL_ROOT = Path("mlruns")
EXPERIMENT_ID_PREFIX = "experiments"


def merge_experiment(source_exp: Path) -> None:
    exp_id = source_exp.name
    dest = LOCAL_ROOT / "experiments" / exp_id
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.mkdir(parents=True, exist_ok=True)

    meta = source_exp / "meta.yaml"
    if meta.exists():
        shutil.copy(meta, dest / "meta.yaml")

    for run_dir in source_exp.iterdir():
        if not run_dir.is_dir():
            continue
        run_dest = dest / run_dir.name
        if run_dest.exists():
            logger.warning(
                f"Run {run_dir.name} already present; skipping (use --overwrite)."
            )
            continue
        logger.info(f"Merging run {run_dir.name} from {run_dir}")
        shutil.copytree(run_dir, run_dest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Path to Kaggle mlruns dir")
    parser.add_argument("--local", default=str(LOCAL_ROOT))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.is_dir():
        raise SystemExit(f"Source not found: {source}")

    for dir_entry in source.iterdir():
        # Each top-level dir is an experiment id (e.g. "936361...") with a
        # meta.yaml; the spec-compliant layout nests under experiments/.
        if dir_entry.name == "models":
            continue
        if (dir_entry / "meta.yaml").exists() or dir_entry.name.isdigit():
            merge_experiment(dir_entry)

    logger.info("Done. View with: mlflow ui")


if __name__ == "__main__":
    main()
