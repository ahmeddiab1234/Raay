"""Fail the PR when the DVC pipeline's metrics drift beyond a tolerance.

DVC 3.67 has no ``-t/--thresholds`` and no ``--fail-on-diff`` on
``dvc metrics diff`` (it always exits 0 and just prints a table), so the
checklist's "+/-0.005" gate lives here instead. The script calls the DVC
API rather than shelling out, for two reasons:

* ``dvc metrics diff`` swallows load failures -- it prints "DVC failed to
  load some metrics for following revisions" to stderr and still exits 0.
  A base revision written before ``.dvc/config`` was migrated to DVC 3 TOML
  therefore produces an *empty* diff, which would look like a clean pass.
  The API exposes the per-revision ``errors`` dict, so a silent pass is
  impossible here.
* The base revision is resolved up front, so a shallow clone or a typo'd
  ``origin/dev`` fails loudly instead of diffing against nothing.

Usage (from the repo root):

    uv run python scripts/ci_metrics_gate.py --base origin/dev
    uv run python scripts/ci_metrics_gate.py --base origin/dev \\
        --threshold 0.005 --markdown reports/metrics_diff.md

Exit code 0 = within tolerance, 1 = drift or unreadable baseline.

Semantics of the diff (``dvc/utils/diff.py``): every changed metric arrives
as ``{"old": ..., "new": ..., "diff": new - old}`` with dotted, flattened
keys. A key that exists on only one side gets ``None`` on the missing side
and therefore *no* ``diff`` entry -- that is an unbounded change and is
always a violation, as is a non-numeric value, since there is nothing to
compare. Counts in ``preprocess_metrics.json`` are integers, so at +/-0.005
they must match exactly; the proportions in both metrics files may move half
a percentage point.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from loguru import logger

DEFAULT_THRESHOLD = 0.005
DEFAULT_BASE = "origin/dev"

PASS = "PASS"
FAIL = "FAIL"
NEW = "NEW"

REASON_DRIFT = "drift beyond threshold"
REASON_ADDED = "metric added"
REASON_REMOVED = "metric removed"
REASON_UNREADABLE = "unreadable baseline"
REASON_MALFORMED = "malformed diff entry"
REASON_NEW_FILE = "no baseline for this metrics file"


@dataclass(frozen=True)
class Row:
    """One compared metric."""

    path: str
    metric: str
    old: Any
    new: Any
    change: float | None
    status: str
    reason: str = ""

    @property
    def failed(self) -> bool:
        return self.status == FAIL


@dataclass(frozen=True)
class GateResult:
    """Outcome of one gate evaluation."""

    rows: list[Row]
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and not any(row.failed for row in self.rows)

    @property
    def violations(self) -> list[Row]:
        return [row for row in self.rows if row.failed]


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


def evaluate(
    diff: dict[str, Any],
    threshold: float = DEFAULT_THRESHOLD,
    errors: Any = None,
    fail_on_new: bool = False,
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
    """
    rows: list[Row] = []
    for path in sorted(diff):
        metrics = diff[path]
        if not isinstance(metrics, dict):
            rows.append(Row(path, "", None, None, None, FAIL, REASON_MALFORMED))
            continue
        for metric in sorted(metrics):
            rows.append(classify(path, metric, metrics[metric], threshold))

    rows = _mark_new_files(rows, fail_on_new=fail_on_new)
    return GateResult(rows=rows, errors=format_errors(errors))


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


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def render_markdown(
    result: GateResult, base: str = DEFAULT_BASE, workspace: str = "workspace"
) -> str:
    """Render the PR-comment body: one table plus a verdict line."""
    lines = [
        "## DVC metrics drift",
        "",
        (
            "Gate: every numeric metric must move by at most the threshold. "
            f"Baseline `{base}` vs `{workspace}`."
        ),
        "",
    ]

    if result.errors:
        lines += [
            "**FAILED** - DVC could not read the baseline, so nothing was compared:",
            "",
        ]
        lines += [f"- `{message}`" for message in result.errors]
        lines.append("")

    if not result.rows:
        if result.ok:
            lines += ["No metric changed.", ""]
        return "\n".join(lines)

    lines += [
        "| Path | Metric | " + f"{base} | {workspace} | Change | Status |",
        "| --- " * 6 + "|",
    ]
    for row in result.rows:
        status = row.status
        if row.reason:
            status = f"{status} ({row.reason})"
        lines.append(
            f"| `{row.path}` | `{row.metric}` | {_fmt(row.old)} | "
            f"{_fmt(row.new)} | {_fmt(row.change)} | {status} |"
        )
    lines.append("")

    if result.ok:
        fresh = sum(1 for row in result.rows if row.status == NEW)
        if fresh:
            lines.append(
                f"**PASSED** - {len(result.rows)} metric(s) within tolerance, "
                f"{fresh} of them new with no baseline yet."
            )
        else:
            lines.append(f"**PASSED** - {len(result.rows)} metric(s) within tolerance.")
    else:
        lines.append(
            f"**FAILED** - {len(result.violations)} of {len(result.rows)} "
            "metric(s) outside tolerance."
        )
    return "\n".join(lines)


def collect_diff(
    base: str, targets: list[str] | None = None
) -> tuple[dict[str, Any], Any, Any]:
    """Return ``(diff, errors, repo)`` for ``base`` vs the workspace."""
    from dvc.repo import Repo
    from dvc.scm import resolve_rev

    repo = Repo()

    # Resolve first: an unknown/shallow base rev raises, which must not be
    # mistaken for "no changes".
    try:
        resolve_rev(repo.scm, base)
    except Exception as exc:
        raise SystemExit(
            f"ci-metrics-gate: cannot resolve base revision {base!r}: {exc}. "
            "Check out the full history (fetch-depth: 0) and the branch name."
        ) from exc

    result = repo.metrics.diff(a_rev=base, b_rev="workspace", targets=targets)
    return result.get("diff", {}), result.get("errors", {}), repo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", default=DEFAULT_BASE, help="revision holding the baseline metrics"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="maximum tolerated absolute change per metric",
    )
    parser.add_argument(
        "--targets", nargs="*", default=None, help="limit to these metrics files"
    )
    parser.add_argument(
        "--fail-on-new",
        action="store_true",
        help="also fail when a metrics file has no baseline at the base revision",
    )
    parser.add_argument(
        "--markdown", default=None, help="also write the PR comment body here"
    )
    parser.add_argument("--json", action="store_true", help="print rows as JSON")
    args = parser.parse_args()

    diff, errors, repo = collect_diff(args.base, args.targets)
    try:
        result = evaluate(
            diff, threshold=args.threshold, errors=errors, fail_on_new=args.fail_on_new
        )
    finally:
        # Drop the DVC repo before interpreter shutdown: dulwich's pack file
        # __del__ emits ImportError noise during teardown.
        del repo
        gc.collect()

    markdown = render_markdown(result, base=args.base)
    if args.markdown:
        out = Path(args.markdown)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown + "\n")
        logger.info(f"Wrote metrics report to {out}")

    if args.json:
        print(
            json.dumps(
                {
                    "ok": result.ok,
                    "errors": result.errors,
                    "rows": [vars(row) for row in result.rows],
                },
                indent=2,
            )
        )
    else:
        print(markdown)

    sys.exit(0 if result.ok else 1)


if __name__ == "__main__":
    main()
