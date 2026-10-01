"""One PSI implementation, shared by the input-drift and output-drift checks.

Both directions of Phase 6 step 1+2 must agree numerically -- a triage that
compared an input PSI from one implementation against an output PSI from
another would be comparing two different quantities under one name -- so
``batch_score`` and ``prediction_drift`` both import from here.

Two failure modes, deliberately reported differently:

**SKIPPED (uncomparable).** The reference column has no usable spread, so
there is no distribution to compare. This is not hypothetical: the tail
principal components of a PCA basis routinely land at ``~1e-16`` std, i.e.
floating-point noise around the component mean, and Evidently's
``numpy.histogram_bin_edges(bins="sturges")`` then raises ``Too many bins for
data range``. Gating those would fail the report for a component that by
construction cannot move, so they are recorded with their observed std and left
out of the severity calculation -- with the number visible, so a reader can see
*why* the column was not checked.

**ERROR (escalates).** Anything else, e.g. a binning failure on a column that
does have spread. That is a bug or an unanticipated input shape, and a
monitoring gate that silently stops checking a column is worse than one that
pages someone, so it is recorded with ``decision: "ERROR"`` and lifts the
overall verdict.

Running per column rather than as one ``Report`` is what makes either outcome
survivable: a single exception from the shared report would take down every
column, not just the broken one.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from loguru import logger

# Below this reference std a numeric column is treated as having no spread at
# all; see ``_uncomparable_reason``.
_MIN_COMPARABLE_STD = 1e-8

#: PASS / WARN / FAIL cut points, defaulting to the agreed bands.
DEFAULT_THRESHOLDS: tuple[float, float] = (0.1, 0.2)

#: Severity order for rolling column verdicts up into one overall verdict.
#: SKIPPED is deliberately absent: a column with no reference spread cannot
#: drift, so letting it into the maximum could only ever downgrade a real
#: verdict, never upgrade one.
SEVERITY: dict[str, int] = {"PASS": 0, "WARN": 1, "FAIL": 2, "ERROR": 3}


def _uncomparable_reason(reference: pd.Series) -> dict[str, Any] | None:
    """Why this reference column has no distribution to compare, or ``None``.

    A numeric column whose reference std is below ``_MIN_COMPARABLE_STD`` cannot
    be binned by Evidently and cannot drift. The threshold is absolute
    because PCA components of mean-pooled AraBERT embeddings are O(0.1-10), so
    anything under ``1e-8`` is seven-plus orders of magnitude below the leading
    components -- numerically indistinguishable from the mean, not a small but
    real signal.

    Categorical columns are never skipped here: a constant *categorical* column
    (every row the same dialect) is still a comparable distribution, because
    Evidently bins categories rather than a value range.
    """
    if not pd.api.types.is_numeric_dtype(reference):
        return None
    values = reference.dropna()
    if values.empty:
        return {"reason": "reference column is entirely NaN", "reference_std": 0.0}
    std = float(values.std())
    if std >= _MIN_COMPARABLE_STD:
        return None
    return {
        "reason": (
            f"reference std {std:.3e} is below {_MIN_COMPARABLE_STD:g}, so the "
            f"column has no spread to bin or to drift in"
        ),
        "reference_std": std,
    }


def psi_result(reference: pd.Series, current: pd.Series) -> dict[str, Any]:
    """Evidently's own verdict dict for a single column.

    Raises whatever Evidently raises; the caller decides whether that is a
    SKIPPED or an ERROR (see the module docstring). The dict is returned whole
    rather than just the score because ``drift_detected`` comes from Evidently's
    own (much tighter, 0.05) threshold, and re-deriving it from the agreed
    FAIL band would silently change what the report says.
    """
    from evidently.legacy.calculations.stattests.psi import psi_stat_test
    from evidently.legacy.metrics import ColumnDriftMetric
    from evidently.legacy.report import Report

    report = Report(
        metrics=[ColumnDriftMetric(column_name=reference.name, stattest=psi_stat_test)]
    )
    report.run(
        reference_data=reference.to_frame(),
        current_data=current.to_frame(),
    )
    return dict(report.as_dict()["metrics"][0]["result"])


def _psi_per_column(
    columns: list[str],
    ref: pd.DataFrame,
    cur: pd.DataFrame,
    thresholds: tuple[float, float],
) -> tuple[dict[str, dict[str, Any]], list[str], list[str]]:
    """Run PSI one column at a time; a column that cannot be binned is ERRORed.

    Returns ``(verdicts, errored, uncomparable)``.
    """
    lo, hi = thresholds
    verdicts: dict[str, dict[str, Any]] = {}
    errored: list[str] = []
    uncomparable: list[str] = []
    for col in columns:
        degenerate = _uncomparable_reason(ref[col])
        if degenerate is not None:
            logger.warning(
                f"Skipping drift column {col!r}: {degenerate['reason']} "
                f"(observed std={degenerate['reference_std']:.3e})"
            )
            verdicts[col] = _no_score_verdict("SKIPPED", **degenerate)
            uncomparable.append(col)
            continue
        try:
            result = psi_result(ref[col], cur[col])
            score = float(result["drift_score"])
            verdicts[col] = {
                "drift_score": round(score, 4),
                "stattest": result["stattest_name"],
                "stattest_threshold": float(result["stattest_threshold"]),
                "drift_detected": bool(result["drift_detected"]),
                "decision": "PASS"
                if score < lo
                else ("WARN" if score < hi else "FAIL"),
            }
        except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
            logger.warning(f"PSI failed for column {col!r}: {exc!r}")
            verdicts[col] = _no_score_verdict(
                "ERROR", error=f"{type(exc).__name__}: {exc}"
            )
            errored.append(col)
    return verdicts, errored, uncomparable


def _no_score_verdict(decision: str, **extra: Any) -> dict[str, Any]:
    """A verdict row for a column that produced no number, and why."""
    return {
        "drift_score": None,
        "stattest": "PSI",
        "stattest_threshold": None,
        "drift_detected": None,
        "decision": decision,
        **extra,
    }


def overall_verdict(columns: dict[str, dict[str, Any]]) -> str:
    """Roll per-column verdicts up to one, worst comparable column wins.

    SKIPPED columns are excluded from the rollup. If *nothing* was checkable
    the rollup is SKIPPED rather than PASS -- a gate that ran no checks must not
    report itself healthy.
    """
    decisions = [
        row["decision"] for row in columns.values() if row["decision"] in SEVERITY
    ]
    if not decisions:
        return "SKIPPED"
    return max(decisions, key=SEVERITY.__getitem__)


def drift_share(columns: dict[str, dict[str, Any]]) -> float:
    """Share of *scored* columns Evidently flagged as drifted."""
    checked = [c for c in columns.values() if c["drift_score"] is not None]
    if not checked:
        return 0.0
    return round(sum(1 for c in checked if c["drift_detected"]) / len(checked), 4)
