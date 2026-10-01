"""nginx reload, the Phase 6 offline gate, and the per-stage gate report.

The offline gate lives in its own module because it is a boundary, not a detail:
``scripts/promote_model.py`` is the only code in the repo allowed to move the
``Production`` alias forward, and ``--to full`` is the one canary path that
reaches it. A non-zero exit from it leaves the nginx conf untouched and the alias
on the old version, which is the whole reason the conf is re-rendered *after* it.
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from canary_gate import GateOutcome
from canary_state import _write_text
from loguru import logger

# The Phase 6 offline gate that ``--to full`` shells out to. It is the only
# code allowed to move the Production alias; a passing single invocation both
# verifies and flips (the 14 offline gates are re-checks of a frozen split, so
# no second pass is needed the way the human-approval workflow requires one).
_PROMOTE_SCRIPT_ARGV = ("uv", "run", "python", "scripts/promote_model.py")


def _reload_nginx(compose_file: str) -> None:
    cmd = [
        "docker",
        "compose",
        "-f",
        compose_file,
        "exec",
        "-T",
        "raay-nginx",
        "nginx",
        "-s",
        "reload",
    ]
    logger.info(f"Reloading nginx: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _run_offline_gate(candidate_version: int) -> None:
    """Run the Phase 6 offline gate; the only code that flips the alias."""
    argv = [*_PROMOTE_SCRIPT_ARGV, "--candidate-version", str(candidate_version)]
    logger.info(f"Running Phase 6 offline gate: {' '.join(argv)}")
    subprocess.run(argv, check=True)


def _gate_report(
    phase: str, target: str, outcome: GateOutcome, thresholds: dict
) -> dict:
    return {
        "decision": "PASS"
        if outcome.passed
        else ("INCONCLUSIVE" if outcome.inconclusive else "FAIL"),
        "phase": phase,
        "target": target,
        "reason": outcome.reason,
        "observed": outcome.observed,
        "thresholds": thresholds,
        "evaluated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def _write_gate_report(report: dict[str, Any], phase: str) -> Path:
    path = Path("reports") / f"canary_gate_{phase}.json"
    _write_text(path, json.dumps(report, indent=2) + "\n")
    return path
