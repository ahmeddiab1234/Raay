"""Hermetic tests for canary rollback behavior."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.error import URLError

import pytest
from canary_helpers import _advance_env, _paths, _rolling_client, _state

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_rollback_restores_stable_weights_and_pre_rollout_alias(
    canary, monkeypatch, tmp_path
):
    client = _rolling_client()
    conf_path, state_path, reloads = _advance_env(
        canary,
        monkeypatch,
        tmp_path,
        _state(phase="full", candidate=2, previous=1),
    )

    canary.cmd_rollback(
        client, conf_path=conf_path, compose_file="f.yml", state_path=state_path
    )

    assert conf_path.read_text() == canary.render_canary_conf("rollback")
    assert reloads == ["f.yml"]
    assert ("ArabicSentiment", 1, "Production", True) in client.transitions
    assert client.aliases[("ArabicSentiment", "Production")] == "1"
    assert json.loads(state_path.read_text())["phase"] == "rolled-back"


def test_rollback_without_state_fails(canary, monkeypatch, tmp_path):
    client = _rolling_client()
    conf_path, state_path = _paths(canary, tmp_path)
    with pytest.raises(SystemExit, match="No rollout state"):
        canary.cmd_rollback(
            client, conf_path=conf_path, compose_file="f.yml", state_path=state_path
        )


def test_rollback_only_runs_once(canary, monkeypatch, tmp_path):
    client = _rolling_client()
    conf_path, state_path, _ = _advance_env(
        canary, monkeypatch, tmp_path, _state(phase="rolled-back")
    )
    with pytest.raises(SystemExit, match="already completed"):
        canary.cmd_rollback(
            client, conf_path=conf_path, compose_file="f.yml", state_path=state_path
        )


def test_rollback_falls_back_to_int8_when_no_pre_rollout_version(
    canary, monkeypatch, tmp_path
):
    client = _rolling_client()
    state = _state(phase="canary-50", previous=None)
    state["previous_production_version"] = None
    conf_path, state_path, _ = _advance_env(canary, monkeypatch, tmp_path, state)

    canary.cmd_rollback(
        client, conf_path=conf_path, compose_file="f.yml", state_path=state_path
    )

    assert client.aliases[("ArabicSentiment", "Production")] == "1"


def test_health_gate_fails_loudly_on_connection_error(canary, monkeypatch):
    def boom(url, timeout):
        raise URLError("refused")

    monkeypatch.setattr(canary.urllib.request, "urlopen", boom)
    with pytest.raises(SystemExit, match="Health gate FAILED"):
        canary._health_gate("http://127.0.0.1:9/health")
