"""The three rollout commands: shadow, advance, rollback.

Each one is ordered so that a failure leaves a *working* configuration rather
than an ambiguous one: rollback moves traffic back before it touches the
registry, and ``advance --to full`` runs the offline gate before it re-renders
the conf, so neither can leave the alias and the routing disagreeing.

Helpers are reached through their module attributes, not from-imported names, so
that replacing e.g. ``canary_probe._health_gate`` in a test actually reaches the
command. See the note in ``canary_promote.py``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import canary_declare
import canary_gate
import canary_nginx_ops
import canary_probe
import canary_registry
import canary_state
import mlflow
from canary_phases import _EDGES, CanaryState, render_canary_conf
from loguru import logger


def cmd_shadow(
    client: mlflow.tracking.MlflowClient,
    *,
    conf_path: Path,
    compose_file: str,
    state_path: Path,
    prometheus_url: str,
) -> None:
    """Health-gate everything, then start mirroring every /predict at the candidate."""
    canary_probe._health_gates(
        (
            canary_probe._STABLE_URL,
            canary_probe._CANDIDATE_URL,
            canary_probe._FRONT_URL,
            canary_probe._INGRESS_URL,
            canary_probe._AGENT_URL,
            canary_probe._PROMETHEUS_HEALTH_URL,
        )
    )
    candidate = canary_declare._canary_version(client)
    canary_state._write_text(conf_path, render_canary_conf("shadow"))
    canary_nginx_ops._reload_nginx(compose_file)
    state = CanaryState(
        phase="shadow",
        candidate_version=int(candidate.version),
        previous_production_version=canary_registry._production_version(client),
        entered_at=datetime.now(UTC).isoformat(timespec="seconds"),
        prometheus_url=prometheus_url,
    )
    canary_state._save_state(state, state_path)
    logger.info(
        f"shadow started: {canary_registry._MODEL} v{state.candidate_version} "
        f"mirrored behind Production v{state.previous_production_version}"
    )


def cmd_advance(
    client: mlflow.tracking.MlflowClient,
    *,
    to: str,
    conf_path: Path,
    compose_file: str,
    state_path: Path,
    window: str,
    min_requests: int,
    hold_seconds: int,
    agreement_min: float,
    error_rate_max: float,
    p95_ratio_max: float,
) -> None:
    """Gate the current stage, then widen the candidate slice (or go full)."""
    state = canary_state._load_state(state_path)
    if _EDGES.get(state.phase) != to:
        raise SystemExit(
            f"cannot advance {state.phase!r} -> {to!r}; "
            f"expected {_EDGES.get(state.phase)!r}"
        )

    canary_probe._health_gates(
        (canary_probe._STABLE_URL, canary_probe._CANDIDATE_URL, canary_probe._AGENT_URL)
    )

    def query(promql: str) -> float | None:
        return canary_probe._promql_query(state.prometheus_url, promql)

    outcome = canary_gate.evaluate_gate(
        state.phase,
        query=query,
        entered_at=state.entered_at,
        window=window,
        min_requests=min_requests,
        hold_seconds=hold_seconds,
        agreement_min=agreement_min,
        error_rate_max=error_rate_max,
        p95_ratio_max=p95_ratio_max,
    )
    thresholds = {
        "window": window,
        "min_requests": min_requests,
        "hold_seconds": hold_seconds,
        "agreement_min": agreement_min,
        "error_rate_max": error_rate_max,
        "p95_ratio_max": p95_ratio_max,
    }
    report = canary_nginx_ops._gate_report(state.phase, to, outcome, thresholds)
    gate_report = canary_nginx_ops._write_gate_report(report, state.phase)
    print(json.dumps(report, indent=2))

    if not outcome.passed:
        print(f"Gate report: {gate_report}")
        raise SystemExit(2 if outcome.inconclusive else 1)

    # ``full`` moves real user traffic to the candidate (they have survived
    # 50/50 already), so the Phase 6 offline gate must be clean first. It is
    # the only code that flips the Production alias; on a non-zero exit the
    # nginx conf is left untouched and the alias is still the old version.
    if to == "full":
        canary_nginx_ops._run_offline_gate(state.candidate_version)

    canary_state._write_text(conf_path, render_canary_conf(to))
    canary_nginx_ops._reload_nginx(compose_file)

    state.last_gate = report
    state.edges.append({"from": state.phase, "to": to, "at": report["evaluated_at"]})
    state.phase = to
    state.entered_at = datetime.now(UTC).isoformat(timespec="seconds")
    canary_state._save_state(state, state_path)
    logger.info(
        f"advanced {canary_registry._MODEL} to {to} "
        f"(roadmap: {[e['to'] for e in state.edges]})"
    )


def cmd_rollback(
    client: mlflow.tracking.MlflowClient,
    *,
    conf_path: Path,
    compose_file: str,
    state_path: Path,
) -> None:
    """Stable-only weights instantly, then return the alias to the pre-rollout version."""
    state = canary_state._load_state(state_path)
    if state.phase == "rolled-back":
        raise SystemExit(f"rollback already completed (state: {state_path})")
    canary_probe._health_gates(
        (
            canary_probe._STABLE_URL,
            canary_probe._CANDIDATE_URL,
            canary_probe._FRONT_URL,
            canary_probe._INGRESS_URL,
            canary_probe._AGENT_URL,
        )
    )

    # Traffic first, registry after: the old graph is the one the service has
    # served all along, so a conf change alone is a working rollback; the alias
    # flip only matters to a freshly-started worker.
    canary_state._write_text(conf_path, render_canary_conf("rollback"))
    canary_nginx_ops._reload_nginx(compose_file)

    target = state.previous_production_version
    if target is None:
        logger.warning(
            "no pre-rollout Production version on record; falling back to the int8 variant"
        )
        target = canary_registry._int8_version(client)
    canary_registry._flip_production(client, target)

    state.phase = "rolled-back"
    canary_state._save_state(state, state_path)
    logger.info(
        f"rolled back: stable-only weights live, "
        f"'{canary_registry._STAGE_PRODUCTION}' alias -> v{target}"
    )
