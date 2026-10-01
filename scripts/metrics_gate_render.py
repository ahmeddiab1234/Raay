"""Render the gate's verdict as the PR-comment body (one table + one line)."""

from __future__ import annotations

from typing import Any

from metrics_gate_types import DEFAULT_BASE, NEW, GateResult


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def render_markdown(
    result: GateResult,
    base: str = DEFAULT_BASE,
    workspace: str = "workspace",
    ignore_desc: str | None = None,
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

    if result.ignored:
        lines.append(
            f"Ignored {result.ignored} metric(s)"
            + (f" matching `{ignore_desc}`" if ignore_desc else "")
            + ": they are excluded from this comparison by design."
        )
        lines.append("")

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
