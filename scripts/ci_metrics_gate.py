"""Gate entry point: collect the DVC diff, evaluate it, report, exit.

Usage (both CI and the scheduled retrain job call this file by path):

    uv run python scripts/ci_metrics_gate.py --base origin/dev --threshold 0.005

Exit codes: 0 = gate passed, 1 = a gate failed (or the baseline was
unreadable -- deliberately the same code, because both mean "do not merge").
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from pathlib import Path
from typing import Any

from loguru import logger
from metrics_gate_evaluate import classify, evaluate, format_errors
from metrics_gate_render import render_markdown
from metrics_gate_types import (
    DEFAULT_BASE,
    DEFAULT_THRESHOLD,
    FAIL,
    NEW,
    PASS,
    REASON_ADDED,
    REASON_DRIFT,
    REASON_MALFORMED,
    REASON_NEW_FILE,
    REASON_REMOVED,
    REASON_UNREADABLE,
    GateResult,
    Row,
)

__all__ = [
    "DEFAULT_BASE",
    "DEFAULT_THRESHOLD",
    "FAIL",
    "NEW",
    "PASS",
    "REASON_ADDED",
    "REASON_DRIFT",
    "REASON_MALFORMED",
    "REASON_NEW_FILE",
    "REASON_REMOVED",
    "REASON_UNREADABLE",
    "GateResult",
    "Row",
    "classify",
    "collect_diff",
    "evaluate",
    "format_errors",
    "main",
    "render_markdown",
]


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
    parser = argparse.ArgumentParser(
        description="Fail when a DVC-tracked metric moves beyond the threshold."
    )
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
        "--ignore",
        action="append",
        default=None,
        help=(
            "regex of keys to exclude from comparison entirely (repeatable). "
            "Refresh runs pass e.g. '.*_size$' so integer sizes/counts may move "
            "while proportions stay strict."
        ),
    )
    parser.add_argument(
        "--markdown", default=None, help="also write the PR comment body here"
    )
    parser.add_argument("--json", action="store_true", help="print rows as JSON")
    args = parser.parse_args()

    ignore = "|".join(args.ignore) if args.ignore else None
    patterns = re.compile(ignore) if ignore else None

    diff, errors, repo = collect_diff(args.base, args.targets)
    try:
        result = evaluate(
            diff,
            threshold=args.threshold,
            errors=errors,
            fail_on_new=args.fail_on_new,
            ignore=patterns,
        )
    finally:
        # Drop the DVC repo before interpreter shutdown: dulwich's pack file
        # __del__ emits ImportError noise during teardown.
        del repo
        gc.collect()

    markdown = render_markdown(
        result, base=args.base, ignore_desc=ignore if ignore else None
    )
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
                    "ignored": result.ignored,
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
