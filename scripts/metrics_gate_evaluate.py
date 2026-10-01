"""The gate's decision: classify each diff entry, then decide pass/fail.

Two rules here are not obvious and were both found the hard way:

* **A per-revision ``errors`` entry always fails.** DVC 3.67 has no
  ``metrics diff -t`` / ``--fail-on-diff``, and its CLI *swallows load errors
  into an empty diff that looks like a pass* -- so the API is used instead, where
  a per-revision ``errors`` mapping can never be mistaken for "no change".
* **A whole-file addition is not drift.** When a metrics file has no baseline at
  all -- every key arrives with ``old=None`` -- DVC reports the file as added and
  nothing is comparable. Those rows become ``NEW`` and pass unless the caller
  asks for ``--fail-on-new``; a key added to a file that *does* have a baseline
  stays a hard failure.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from metrics_gate_types import (
    DEFAULT_THRESHOLD,
    FAIL,
    NEW,
    PASS,
    REASON_ADDED,
    REASON_DRIFT,
    REASON_MALFORMED,
    REASON_NEW_FILE,
    REASON_REMOVED,
    GateResult,
    Row,
)


def _is_number(value: Any) -> bool:
    # bool is an int subclass; a boolean metric is not comparable numerically.
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def classify(
    path: str, metric: str, change: Any, threshold: float = DEFAULT_THRESHOLD
) -> Row:
    """Turn one raw diff entry into a :class:`Row` with a pass/fail verdict."""
    if not isinstance(change, dict):
        return Row(path, metric, None, None, None, FAIL, REASON_MALFORMED)

    old = change.get("old")
    new = change.get("new")
    delta = change.get("diff")

    # No numeric delta: the key is new, gone, or not numeric at all.
    if not _is_number(delta):
        if old is None:
            reason = REASON_ADDED
        elif new is None:
            reason = REASON_REMOVED
        else:
            reason = REASON_MALFORMED
        return Row(path, metric, old, new, None, FAIL, reason)

    status = FAIL if abs(delta) > threshold else PASS
    return Row(
        path,
        metric,
        old,
        new,
        float(delta),
        status,
        REASON_DRIFT if status == FAIL else "",
    )


def _mark_new_files(rows: list[Row], fail_on_new: bool) -> list[Row]:
    """Downgrade whole-file additions to ``NEW`` unless told otherwise."""
    by_path: dict[str, list[Row]] = {}
    for row in rows:
        by_path.setdefault(row.path, []).append(row)

    status = FAIL if fail_on_new else NEW
    for path, group in by_path.items():
        if not group or any(row.old is not None for row in group):
            continue
        if any(row.reason != REASON_ADDED for row in group):
            continue
        for index, row in enumerate(group):
            by_path[path][index] = replace(row, status=status, reason=REASON_NEW_FILE)
    return [row for path in sorted(by_path) for row in by_path[path]]


def format_errors(errors: Any) -> list[str]:
    """Flatten the API's ``{revision: {path: error}}`` mapping to strings."""
    if not errors:
        return []
    if not isinstance(errors, dict):
        return [str(errors)]
    out: list[str] = []
    for revision, detail in errors.items():
        if isinstance(detail, dict):
            for path, err in detail.items():
                out.append(f"{revision}: {path}: {_err_text(err)}")
        else:
            out.append(f"{revision}: {_err_text(detail)}")
    return out


def _err_text(err: Any) -> str:
    if isinstance(err, BaseException):
        return f"{type(err).__name__}: {err}"
    return str(err)


def evaluate(
    diff: dict[str, Any],
    threshold: float = DEFAULT_THRESHOLD,
    errors: Any = None,
    fail_on_new: bool = False,
    ignore: re.Pattern[str] | str | None = None,
) -> GateResult:
    """Classify every metric in a ``dvc metrics diff --json`` payload.

    ``errors`` is the API's per-revision load-failure mapping; any entry at
    all fails the gate, because a revision DVC could not read cannot be a
    revision we verified against.

    ``fail_on_new`` controls the bootstrap case. When a metrics file has no
    baseline at all -- every one of its keys arrives with ``old=None`` -- DVC
    reports the whole file as added and nothing can be compared. That is not
    drift, it is a missing baseline, so those rows are reported as ``NEW`` and
    pass by default; pass ``fail_on_new=True`` to require an explicit baseline
    instead. A key added to a file that *does* have a baseline stays a failure.

    ``ignore`` excludes matching keys from comparison entirely -- they can
    never fail the gate, however far they moved. A key matches when the regex
    hits the metrics path or the flattened metric name (or both); a compiled
    pattern or a regex string are both accepted. This is the refresh-mode
    escape hatch: the scheduled retrain gate wants the *proportions* strict at
    +/-0.005 while letting integer sizes/counts move (the point of new data),
    so it passes something like ``.*_size$`` -- without which any real data
    change would always fail, because at +/-0.005 counts must match exactly.
    CI's own PR gate keeps calling ``evaluate`` without ``ignore`` and is
    therefore unchanged.
    """
    pattern = re.compile(ignore) if isinstance(ignore, str) else ignore
    ignored = 0
    rows: list[Row] = []
    for path in sorted(diff):
        metrics = diff[path]
        if not isinstance(metrics, dict):
            rows.append(Row(path, "", None, None, None, FAIL, REASON_MALFORMED))
            continue
        for metric in sorted(metrics):
            if pattern is not None and (
                pattern.search(path) is not None or pattern.search(metric) is not None
            ):
                ignored += 1
                continue
            rows.append(classify(path, metric, metrics[metric], threshold))

    rows = _mark_new_files(rows, fail_on_new=fail_on_new)
    return GateResult(rows=rows, errors=format_errors(errors), ignored=ignored)
