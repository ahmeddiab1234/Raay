"""Shadow-then-canary rollout for the Raay service (Phase 5 step 5).

The rollout owns the *traffic split*, not just the registry alias. It drives
the standalone ``docker-compose.canary.yml`` project (``name: canary``) where
two workers -- ``raay-sentiment`` (the current Production graph) and
``raay-sentiment-canary`` (the declared candidate) -- sit behind an nginx
front that publishes :8000 (live ingress) and :8081 (operator front). nginx
owns :8000 only while this project is up, so the rollback is a conf change,
not a compose swap.

Stage progression (one conf render + ``nginx -s reload`` per stage):

    shadow       stable 100 / candidate 0; every /predict is mirrored to the
                 candidate (nginx ``mirror /shadow``). Both hops carry
                 ``X-Raay-Shadow: 1`` so the agent expects a pair.
    canary-5     95/5 weighted split, no mirror.
    canary-25    75/25.
    canary-50    50/50.
    full         candidate 100, no mirror, then the Phase 6 offline gate.

A stage only advances on gates fed by the compose-local Prometheus (which
scrapes the canary-agent that the workers POST events to). The shadow stage
gates on label agreement + candidate error rate + p95 ratio vs the stable
worker; the weighted stages gate on error rate + p95 ratio only (agreement
needs the mirror). ``--to full`` *additionally* runs the Phase 6 offline
promotion gate (``scripts/promote_model.py``) -- the only code allowed to
move the ``Production`` alias -- so the alias can never flip without both the
online and offline gates being clean.

Rollback re-renders the stable-only conf and flips the alias back to the
version recorded when the shadow started. The state file
``reports/canary_state.json`` (git-ignored) tracks the phase, the candidate,
the pre-rollout Production version and the last gate result; a promotion and
its rollback are one atomic story rooted in that file.

Modes (run from repo root):

    uv run python scripts/canary_promote.py --mode declare                 # idempotent register of the candidate
    uv run python scripts/canary_promote.py --mode shadow                  # enter shadow (mirror, 100/0)
    uv run python scripts/canary_promote.py --mode advance --to canary-5   # gate + widen; also 25 / 50 / full
    uv run python scripts/canary_promote.py --mode rollback                # stable-only conf + alias back to pre-rollout

Implementation lives in siblings -- ``canary_phases`` (phase table + conf
renderer + state dataclass), ``canary_state`` (atomic state writes),
``canary_probe`` (health + PromQL), ``canary_gate`` (the online stage gate),
``canary_nginx_ops`` (reload, offline gate, gate report), ``canary_registry``
(the one backward alias flip), ``canary_model`` (MLmodel dir assembly),
``canary_declare``, ``canary_modes`` (the three commands) and ``canary_cli`` --
re-exported here so ``python scripts/canary_promote.py`` keeps one entry point.

Two consequences of that split, both deliberate:

* ``scripts/`` is not a package, so this module puts its own directory on
  ``sys.path`` before importing the siblings. ``tests/test_canary_nginx.py``
  loads this file by path with ``importlib.util``, which would otherwise leave
  the siblings unimportable.
* The commands call their helpers through the *defining* module
  (``canary_probe._health_gate(...)``), never as from-imported names. A
  from-import copies the reference at import time, so a test that replaces
  ``canary_probe._health_gate`` would patch a name the command never reads and
  the command would try a real HTTP call instead. ``tests/test_canary_nginx.py``
  patches the defining modules for exactly this reason.
"""

from __future__ import annotations

import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

# Reached through this facade by tests that patch a shared *module object*
# (urllib.request.urlopen, subprocess.run, the mlflow client) rather than one of
# our own functions. Patching a module object is global, so it works from here
# exactly as it does from the sibling that actually calls it.
import mlflow

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:  # pragma: no cover - import-time shim
    sys.path.append(_HERE)

from canary_cli import main
from canary_declare import _TOOL_TAG, _canary_version, declare
from canary_gate import GateOutcome, evaluate_gate
from canary_model import _materialize_canary_model_dir
from canary_modes import cmd_advance, cmd_rollback, cmd_shadow
from canary_nginx_ops import (
    _PROMOTE_SCRIPT_ARGV,
    _gate_report,
    _reload_nginx,
    _run_offline_gate,
    _write_gate_report,
)
from canary_phases import (
    _CANDIDATE_UPSTREAM,
    _EDGES,
    _PHASES,
    _STABLE_UPSTREAM,
    CanaryState,
    _weights_for,
    render_canary_conf,
)
from canary_probe import (
    _AGENT_URL,
    _CANDIDATE_URL,
    _FRONT_URL,
    _INGRESS_URL,
    _PROMETHEUS_HEALTH_URL,
    _STABLE_URL,
    _health_gate,
    _health_gates,
    _promql_query,
)
from canary_registry import (
    _MODEL,
    _STAGE_CANARY,
    _STAGE_PRODUCTION,
    _flip_production,
    _int8_version,
    _production_version,
)
from canary_state import _load_state, _save_state, _write_text

__all__ = [
    "CanaryState",
    "GateOutcome",
    "cmd_advance",
    "cmd_rollback",
    "cmd_shadow",
    "declare",
    "evaluate_gate",
    "main",
    "mlflow",
    "render_canary_conf",
    "subprocess",
    "urllib",
]

# Names the original module exposed that the list above does not name.
_PRIVATE_REEXPORTS = (
    _AGENT_URL,
    _CANDIDATE_URL,
    _CANDIDATE_UPSTREAM,
    _EDGES,
    _FRONT_URL,
    _INGRESS_URL,
    _MODEL,
    _PHASES,
    _PROMETHEUS_HEALTH_URL,
    _PROMOTE_SCRIPT_ARGV,
    _STABLE_URL,
    _STABLE_UPSTREAM,
    _STAGE_CANARY,
    _STAGE_PRODUCTION,
    _TOOL_TAG,
    _canary_version,
    _flip_production,
    _gate_report,
    _health_gate,
    _health_gates,
    _int8_version,
    _load_state,
    _materialize_canary_model_dir,
    _production_version,
    _promql_query,
    _reload_nginx,
    _run_offline_gate,
    _save_state,
    _weights_for,
    _write_gate_report,
    _write_text,
)


if __name__ == "__main__":
    main()
