"""Merge Kaggle-session MLflow runs into the local tracking store.

Kaggle training writes runs to a session-local file store (e.g. ``mlruns``
zipped from the notebook working dir). Copy that directory into this repo and
this script will merge every experiment/run/metric into the local tracking
store so that ``mlflow ui`` shows runs from both the local pipeline and the GPU
sessions.

Experiments are written at the top level of the local file store
(``<local>/<experiment_id>/``), the layout MLflow reads — not the
``experiments/``-nested variant. The file-store model registry (``models/``)
is copied as well so registered models survive the merge.

Usage:
    uv run python scripts/merge_mlflow.py --source path/to/session/mlruns
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from loguru import logger

from raay.enums.constants import DefaultPaths

LOCAL_ROOT = Path(DefaultPaths.LOCAL_MLRUNS.value)


def _copy_tree(source: Path, dest: Path, overwrite: bool) -> bool:
    if dest.exists():
        if not overwrite:
            logger.warning(
                f"{source.name} already present; skipping (use --overwrite)."
            )
            return False
        shutil.rmtree(dest)
    logger.info(f"Merging {source} -> {dest}")
    shutil.copytree(source, dest)
    return True


def merge_experiment(source_exp: Path, local_root: Path, overwrite: bool) -> None:
    """Merge one experiment dir into the top-level MLflow file-store layout."""
    exp_id = source_exp.name
    dest = local_root / exp_id
    dest.mkdir(parents=True, exist_ok=True)

    meta = source_exp / "meta.yaml"
    if meta.exists():
        meta_dest = dest / "meta.yaml"
        if not meta_dest.exists() or overwrite:
            logger.info(f"Copying experiment meta {meta} -> {meta_dest}")
            shutil.copy(meta, meta_dest)

    for run_dir in source_exp.iterdir():
        if run_dir.is_dir():
            _copy_tree(run_dir, dest / run_dir.name, overwrite)


def _merge_models(source: Path, local_root: Path, overwrite: bool) -> None:
    """Copy the file-store model registry (registered models) if present."""
    models_src = source / "models"
    if models_src.is_dir():
        _copy_tree(models_src, local_root / "models", overwrite)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Path to Kaggle mlruns dir")
    parser.add_argument("--local", default=str(LOCAL_ROOT))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.is_dir():
        raise SystemExit(f"Source not found: {source}")
    local_root = Path(args.local)
    local_root.mkdir(parents=True, exist_ok=True)

    _merge_models(source, local_root, args.overwrite)

    for dir_entry in source.iterdir():
        # Each top-level dir is an experiment id (e.g. "703216...") with a
        # meta.yaml; `models` is the registry dir, handled above.
        if dir_entry.name == "models":
            continue
        if (dir_entry / "meta.yaml").exists() or dir_entry.name.isdigit():
            merge_experiment(dir_entry, local_root, args.overwrite)

    logger.info(f"Done. View with: mlflow ui --backend-store-uri {local_root}")


if __name__ == "__main__":
    main()
