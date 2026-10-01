"""MLflow/artifact helpers for ONNX export."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from raay.inference.export_config import _DEFAULT_OUTPUT_NAMES


def output_name_for(model_dir: str, explicit_name: str | None) -> str:
    if explicit_name:
        return explicit_name
    return _DEFAULT_OUTPUT_NAMES.get(model_dir, Path(model_dir).parent.name)


def _output_name_for(model_dir: str, explicit_name: str | None) -> str:
    return output_name_for(model_dir, explicit_name)


def _repair_artifact_locations(tracking_uri: str) -> int:
    """Repoint experiments whose artifact root is a dead ``/kaggle`` path."""
    repaired = 0

    def fix(loc: str | None, exp_id: str, update) -> bool:  # type: ignore[no-untyped-def]
        if not loc or "/kaggle/" not in loc:
            return False
        path = loc.removeprefix("file://").split("?", 1)[0]
        if os.path.isdir(path) and os.access(path, os.W_OK):
            # A real (writable) Kaggle workspace is present: leave it alone.
            return False
        update(f"./mlruns/{exp_id}")
        return True

    if tracking_uri.startswith("sqlite:"):
        db_path = tracking_uri.removeprefix("sqlite:///")
        if not db_path or db_path == ":memory:":
            return 0
        con = sqlite3.connect(db_path)
        try:
            rows = con.execute(
                "select experiment_id, artifact_location from experiments"
            ).fetchall()
            for exp_id, loc in rows:
                if fix(
                    loc,
                    exp_id,
                    lambda new, eid=exp_id: con.execute(
                        "update experiments set artifact_location=? "
                        "where experiment_id=?",
                        (new, eid),
                    ),
                ):
                    repaired += 1
            con.commit()
        finally:
            con.close()
        return repaired

    if tracking_uri.startswith("file:"):
        root = Path(tracking_uri.removeprefix("file:"))
        for meta in root.glob("*/meta.yaml"):
            data = yaml.safe_load(meta.read_text()) or {}
            exp_id = meta.parent.name
            loc = data.get("artifact_location")

            def do_update(
                new: str, data: dict[str, Any] = data, meta: Path = meta
            ) -> None:
                data["artifact_location"] = new
                meta.write_text(yaml.safe_dump(data, sort_keys=False))

            if fix(loc, exp_id, do_update):
                repaired += 1
        return repaired

    return 0


def _report_results(report: dict[str, dict[str, Any]], report_path: str) -> None:
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Wrote parity report: {report_path}")
