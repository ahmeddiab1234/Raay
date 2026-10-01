"""The promotion report: everything a reader needs to re-derive or challenge it.

Written whenever a decision is produced -- including a rejection, because "the
candidate was rejected and here is every number behind it" is the thing a human
needs. The one case that produces no report is a gate that could not run at all:
a report full of numbers derived from the wrong data is worse than no report, so
that path raises before this module is reached.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from promotion_types import (
    _MODEL,
    _TOOL_TAG,
    Config,
    GateResult,
    _round,
    file_md5,
)

PROVENANCE_DOC = (
    "trigger provenance is read and applied by promotion_trigger, and recorded "
    "here only as `trigger_tags_applied` -- what stuck, not what was attempted"
)


def report_path(cfg: Config, version: str) -> Path:
    return cfg.report_dir / f"promotion_{version}.json"


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )


def format_table(gates: list[GateResult]) -> str:
    header = f"{'gate':<34} {'verdict':<8} {'observed':>12} {'threshold':>12}"
    lines = [header, "-" * len(header)]
    for gate in gates:
        observed = "-" if gate.observed is None else str(gate.observed)
        threshold = "-" if gate.threshold is None else str(gate.threshold)
        lines.append(
            f"{gate.name:<34} {'PASS' if gate.passed else 'FAIL':<8} "
            f"{observed:>12} {threshold:>12}"
        )
    return "\n".join(lines)


def build_payload(
    cfg: Config,
    version: str,
    candidate: dict[str, Any],
    production: dict[str, Any],
    baseline: dict[str, Any],
    gates: list[GateResult],
    promoted: bool,
    previous_production: str | None,
    dry_run: bool = False,
    trigger_tags: list[str] | None = None,
) -> dict[str, Any]:
    """Everything a reader needs to re-derive or challenge the verdict."""
    failed = [gate.name for gate in gates if not gate.passed]
    if promoted:
        decision = "promoted"
    elif failed:
        decision = "rejected"
    else:
        # Every gate passed but nothing was flipped -- a --skip-registry or
        # --dry-run rehearsal. Calling that "rejected" would be a lie about
        # what the measurements said.
        decision = "passed_not_promoted"
    if decision == "passed_not_promoted":
        # Distinguish "measured, nobody could flip it" from "measured, and the
        # Promotion was deliberately withheld", which are very different things
        # to find in a report attached to an approval.
        blocked_reason = "dry_run" if dry_run else "no_registry_client"
    else:
        blocked_reason = None
    return {
        "tool": _TOOL_TAG,
        "registered_model": _MODEL,
        "candidate_version": version,
        "previous_production_version": previous_production,
        "promoted": promoted,
        "decision": decision,
        "promotion_blocked_reason": blocked_reason,
        # Which Phase 6 step 3 trigger tags actually landed on the version. Empty
        # is normal (no trigger report, or a dry run) and is why this records
        # what stuck rather than what was attempted.
        "trigger_tags_applied": trigger_tags or [],
        "dry_run": dry_run,
        "failed_gates": failed,
        "gates": [gate.as_dict() for gate in gates],
        "candidate": candidate,
        "production_baseline": production,
        "floor_baseline": {
            "source": str(cfg.floor_report),
            "f1_macro": baseline["f1_macro"],
            "accuracy": baseline["accuracy"],
            "per_class": {
                name: values.get("recall")
                for name, values in baseline.get("per_class", {}).items()
            },
        },
        "test_split": {
            "path": str(cfg.test_split),
            "md5": file_md5(cfg.test_split) if cfg.test_split.exists() else None,
            "source": "dvc repro output, pinned by dvc.lock",
            "used_for_training": False,
        },
        "thresholds": {
            "f1_tolerance": cfg.f1_tolerance,
            "floor_tolerance": cfg.floor_tolerance,
            "accuracy_tolerance": cfg.accuracy_tolerance,
            "recall_tolerance": cfg.recall_tolerance,
            "min_recall": cfg.min_recall,
            "latency_regression": cfg.latency_regression,
            "max_size_mb": cfg.max_size_mb,
            "min_label_agreement": cfg.min_label_agreement,
            "max_parity_diff": cfg.max_parity_diff,
        },
        "latency_noise": {
            "same_graph_noise_pct": production.get("latency", {}).get("noise_pct"),
            "control_gap_pct": production.get("latency", {}).get("control_gap_pct"),
            "series_p95_ms": {
                "candidate": candidate.get("latency", {}).get("p95_ms"),
                "production": production.get("latency", {}).get("p95_ms"),
                "control": production.get("latency", {}).get("control_p95_ms"),
            },
            "allowed_regression_pct": _round(cfg.latency_regression * 100.0, 2),
            "resolvable_on_this_host": (
                None
                if production.get("latency", {}).get("noise_pct") is None
                else production["latency"]["noise_pct"]
                <= cfg.latency_regression * 100.0
            ),
            "note": (
                "Spread across the three interleaved series (candidate, production, "
                "and a control that re-times production) versus the gap between the "
                "two production series alone. The spread is the bound used for the "
                "diagnosis. "
                "Above allowed_regression_pct the host cannot resolve the band, so "
                "a latency failure is a measurement problem rather than evidence "
                "about the candidate."
            ),
        },
        "caveat": (
            "Latency is measured on the machine that ran this gate and compared "
            "against the production graph measured in the same process; it is not "
            "comparable with reports/benchmark_table.csv, which was measured "
            "elsewhere."
        ),
    }
