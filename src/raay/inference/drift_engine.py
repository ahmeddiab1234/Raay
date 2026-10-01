"""Run the engineered input-drift check and write its JSON report.

This module wires together ``batch_scoring`` (to read frames), the engineered
feature stack (to compute the columns), and ``drift_psi`` (to run per-column
PSI). The output side (class mix, confidence series, triage) is separate (see
:mod:`raay.inference.prediction_drift`): the useful question when something
moves is *which half moved*, not "did anything move at all".

The gate reads a cached ``reference_engineered.csv`` when present -- that is the
frozen side -- and always writes the current day's engineered panel beside it so
a human can open the exact comparison that failed. ``--no-engineer`` falls back
to the two original output columns, in which case the entire encoder path is
never touched.

The frame-shaping helpers live in ``drift_frames``; this module is the check
itself and the report.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from raay.enums.constants import DataColumns
from raay.inference.drift_columns import (
    COL_CONFIDENCE,
    COL_OOV_RATE,
    COL_TEXT_LENGTH,
)
from raay.inference.drift_frames import (
    OUTPUT_ONLY_COLUMNS,
    EngineeredDriftSpec,
    engineered_block,
    engineered_frames,
    resolve_columns,
)
from raay.inference.drift_psi import (
    DEFAULT_THRESHOLDS,
    _psi_per_column,
    drift_share,
    overall_verdict,
)

#: Feature means reported in the verdict so a rising OOV rate or a shortening
#: review is a visible series rather than something you have to open a CSV for.
REPORTED_MEANS = (COL_TEXT_LENGTH, COL_CONFIDENCE, COL_OOV_RATE)

__all__ = [
    "OUTPUT_ONLY_COLUMNS",
    "REPORTED_MEANS",
    "EngineeredDriftSpec",
    "drift_check",
]


def drift_check(
    reference_csv: str,
    current_csv: str,
    out_json: str,
    date: str,
    thresholds: tuple[float, float] = DEFAULT_THRESHOLDS,
    drift_columns: tuple[str, ...] | None = None,
    engineered: EngineeredDriftSpec | None = None,
) -> dict[str, Any]:
    """Evidently PSI drift between reference and a scored day; write verdicts.

    With ``engineered`` the comparison widens from the model's outputs to the
    inputs as well (embedding PCs, ``[UNK]`` rate, dialect mix, length,
    confidence).
    """
    ref = pd.read_csv(reference_csv)
    cur = pd.read_csv(current_csv)
    reference_label, current_label = reference_csv, current_csv
    basis = None
    reference_from_cache = False
    features_applied = False

    if engineered is not None:
        if DataColumns.TEXT.value not in ref.columns:
            # No text means no embeddings, no OOV, no recomputed dialect. Say so
            # and fall through to the output-only comparison rather than
            # reporting engineered columns that were never computed.
            logger.warning(
                f"{reference_csv} has no 'text' column, so the engineered input "
                f"features are unavailable; falling back to the output-only "
                f"drift columns"
            )
        else:
            ref, cur, basis, reference_from_cache = engineered_frames(
                reference_csv, current_csv, engineered
            )
            reference_label = (
                engineered.reference_engineered
                if reference_from_cache
                else reference_csv
            )
            current_label = engineered.current_engineered
            features_applied = True

    present, skipped = resolve_columns(
        ref,
        cur,
        (reference_label, current_label),
        drift_columns,
        engineered,
        features_applied,
    )
    lo, hi = thresholds
    columns, errored, uncomparable = _psi_per_column(present, ref, cur, thresholds)
    verdict: dict[str, Any] = {
        "date": date,
        "thresholds": {"warn": lo, "fail": hi},
        "reference": reference_label,
        "current": current_label,
        "n_reference": len(ref),
        "n_current": len(cur),
        "columns": columns,
        "n_columns_checked": sum(
            1 for c in columns.values() if c["drift_score"] is not None
        ),
        "skipped_columns": skipped,
        "uncomparable_columns": uncomparable,
        "errored_columns": errored,
        "drift_share": drift_share(columns),
        "overall": overall_verdict(columns),
    }
    if features_applied:
        verdict |= engineered_block(
            ref, cur, basis, reference_from_cache, REPORTED_MEANS
        )
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(verdict, f, indent=2, ensure_ascii=False)
    logger.info(f"Drift check {verdict['overall']}: {columns}")
    return verdict
