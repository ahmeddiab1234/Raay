"""Hermetic tests for canary advance behavior."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from canary_helpers import (
    _advance_env,
    _good_query,
    _rolling_client,
    _state,
    canary_nginx_ops,
    canary_probe,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_advance_without_state_fails(canary, monkeypatch, tmp_path):
    client = _rolling_client()
    conf_path, state_path, _ = _advance_env(canary, monkeypatch, tmp_path, None)
    with pytest.raises(SystemExit, match="--mode shadow"):
        canary.cmd_advance(
            client,
            to="canary-5",
            conf_path=conf_path,
            compose_file="f.yml",
            state_path=state_path,
            window="5m",
            min_requests=500,
            hold_seconds=600,
            agreement_min=0.99,
            error_rate_max=0.005,
            p95_ratio_max=1.10,
        )


def test_advance_rejects_illegal_transition(canary, monkeypatch, tmp_path):
    client = _rolling_client()
    conf_path, state_path, _ = _advance_env(
        canary, monkeypatch, tmp_path, _state(phase="shadow")
    )
    with pytest.raises(SystemExit, match="cannot advance"):
        canary.cmd_advance(
            client,
            to="canary-50",
            conf_path=conf_path,
            compose_file="f.yml",
            state_path=state_path,
            window="5m",
            min_requests=500,
            hold_seconds=600,
            agreement_min=0.99,
            error_rate_max=0.005,
            p95_ratio_max=1.10,
        )


def test_advance_shadow_to_canary5_on_green_gate(canary, monkeypatch, tmp_path):
    client = _rolling_client()
    state = _state(phase="shadow")
    conf_path, state_path, reloads = _advance_env(canary, monkeypatch, tmp_path, state)
    q = _good_query()
    monkeypatch.setattr(canary_probe, "_promql_query", lambda _url, promql: q(promql))

    canary.cmd_advance(
        client,
        to="canary-5",
        conf_path=conf_path,
        compose_file="f.yml",
        state_path=state_path,
        window="5m",
        min_requests=500,
        hold_seconds=600,
        agreement_min=0.99,
        error_rate_max=0.005,
        p95_ratio_max=1.10,
    )

    assert conf_path.read_text() == canary.render_canary_conf("canary-5")
    assert reloads == ["f.yml"]
    loaded = json.loads(state_path.read_text())
    assert loaded["phase"] == "canary-5"
    assert loaded["edges"][0]["from"] == "shadow"
    assert loaded["edges"][0]["to"] == "canary-5"


def test_advance_fails_but_leaves_conf_and_state_untouched(
    canary, monkeypatch, tmp_path
):
    client = _rolling_client()
    conf_path, state_path, reloads = _advance_env(
        canary, monkeypatch, tmp_path, _state(phase="shadow")
    )
    conf_path.write_text("UNTOUCHED")
    # Disagree every pair -> agreement 0 -> FAIL with exit 1.
    q = _good_query(agree=0.0, disagree=600.0)
    monkeypatch.setattr(canary_probe, "_promql_query", lambda _url, promql: q(promql))

    with pytest.raises(SystemExit) as exc:
        canary.cmd_advance(
            client,
            to="canary-5",
            conf_path=conf_path,
            compose_file="f.yml",
            state_path=state_path,
            window="5m",
            min_requests=500,
            hold_seconds=600,
            agreement_min=0.99,
            error_rate_max=0.005,
            p95_ratio_max=1.10,
        )
    assert exc.value.code == 1
    assert conf_path.read_text() == "UNTOUCHED"
    assert json.loads(state_path.read_text())["phase"] == "shadow"
    assert reloads == []


def test_advance_inconclusive_never_advances(canary, monkeypatch, tmp_path):
    client = _rolling_client()
    conf_path, state_path, _ = _advance_env(
        canary, monkeypatch, tmp_path, _state(phase="shadow")
    )
    monkeypatch.setattr(
        canary, "_promql_query", lambda _url, promql: None
    )  # scraped nothing

    with pytest.raises(SystemExit) as exc:
        canary.cmd_advance(
            client,
            to="canary-5",
            conf_path=conf_path,
            compose_file="f.yml",
            state_path=state_path,
            window="5m",
            min_requests=500,
            hold_seconds=600,
            agreement_min=0.99,
            error_rate_max=0.005,
            p95_ratio_max=1.10,
        )
    assert exc.value.code == 2
    assert json.loads(state_path.read_text())["phase"] == "shadow"


def test_advance_hold_fail_is_a_failure_not_inconclusive(canary, monkeypatch, tmp_path):
    client = _rolling_client()
    conf_path, state_path, _ = _advance_env(
        canary,
        monkeypatch,
        tmp_path,
        _state(
            phase="shadow",
            entered=(datetime.now(UTC) - timedelta(seconds=10)).isoformat(),
        ),
    )
    q = _good_query()
    monkeypatch.setattr(canary_probe, "_promql_query", lambda _url, promql: q(promql))

    with pytest.raises(SystemExit) as exc:
        canary.cmd_advance(
            client,
            to="canary-5",
            conf_path=conf_path,
            compose_file="f.yml",
            state_path=state_path,
            window="5m",
            min_requests=500,
            hold_seconds=600,
            agreement_min=0.99,
            error_rate_max=0.005,
            p95_ratio_max=1.10,
        )
    assert exc.value.code == 1


def test_advance_to_full_runs_offline_gate_before_moving_traffic(
    canary, monkeypatch, tmp_path
):
    client = _rolling_client()
    conf_path, state_path, reloads = _advance_env(
        canary,
        monkeypatch,
        tmp_path,
        _state(
            phase="canary-50",
        ),
    )
    q = _good_query()
    monkeypatch.setattr(canary_probe, "_promql_query", lambda _url, promql: q(promql))
    offline_calls = []
    monkeypatch.setattr(
        canary_nginx_ops, "_run_offline_gate", lambda v: offline_calls.append(v)
    )

    canary.cmd_advance(
        client,
        to="full",
        conf_path=conf_path,
        compose_file="f.yml",
        state_path=state_path,
        window="5m",
        min_requests=500,
        hold_seconds=600,
        agreement_min=0.99,
        error_rate_max=0.005,
        p95_ratio_max=1.10,
    )

    assert offline_calls == [2]
    assert conf_path.read_text() == canary.render_canary_conf("full")
    assert reloads == ["f.yml"]
    assert json.loads(state_path.read_text())["phase"] == "full"


def test_advance_to_full_is_blocked_if_offline_gate_fails(
    canary, monkeypatch, tmp_path
):
    client = _rolling_client()
    conf_path, state_path, _ = _advance_env(
        canary,
        monkeypatch,
        tmp_path,
        _state(
            phase="canary-50",
        ),
    )
    q = _good_query()
    monkeypatch.setattr(canary_probe, "_promql_query", lambda _url, promql: q(promql))

    def abort(_version):
        raise canary.subprocess.CalledProcessError(1, ["promote_model.py"])

    monkeypatch.setattr(canary_nginx_ops, "_run_offline_gate", abort)

    with pytest.raises(canary.subprocess.CalledProcessError):
        canary.cmd_advance(
            client,
            to="full",
            conf_path=conf_path,
            compose_file="f.yml",
            state_path=state_path,
            window="5m",
            min_requests=500,
            hold_seconds=600,
            agreement_min=0.99,
            error_rate_max=0.005,
            p95_ratio_max=1.10,
        )
    # nginx conf must not have moved to 100% candidate before the offline gate
    # approved the model.
    assert not conf_path.exists()
