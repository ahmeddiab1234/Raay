"""Hermetic tests for canary declare behavior."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from canary_helpers import (
    FakeMlflowClient,
    FakeRun,
    _fake_start_run,
    _patch_rollout,
    _rolling_client,
    _version,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_declare_registers_canary_alias(canary, monkeypatch, tmp_path):
    client = FakeMlflowClient(
        runs=[FakeRun("old-canary")],
        versions=[
            _version("ArabicSentiment", 1, {"variant": "onnx-int8"}),
            _version("ArabicSentiment", 2, {"variant": "distilled-fp32"}),
        ],
    )
    monkeypatch.setattr(canary.mlflow, "start_run", _fake_start_run)

    def fake_register(model_uri, name):
        v = _version(name, 3, {"variant": "distilled-fp32"})
        v.run_id = "new-run"
        return v

    monkeypatch.setattr(canary.mlflow, "register_model", fake_register)
    monkeypatch.setattr(
        canary,
        "_materialize_canary_model_dir",
        lambda run_id, onnx_path, out_dir: str(tmp_path),
    )

    registered = canary.declare(client, "raay_training")

    assert registered.name == "ArabicSentiment"
    assert registered.version == 3
    assert client.aliases[("ArabicSentiment", "Canary")] == "3"
    assert client.deleted == ["old-canary"]


def test_declare_is_idempotent(canary, monkeypatch, tmp_path):
    client = FakeMlflowClient(runs=[], versions=[])
    monkeypatch.setattr(canary.mlflow, "start_run", _fake_start_run)

    def fake_register(model_uri, name):
        return _version(name, 1, {"variant": "distilled-fp32"})

    monkeypatch.setattr(canary.mlflow, "register_model", fake_register)
    monkeypatch.setattr(
        canary,
        "_materialize_canary_model_dir",
        lambda run_id, onnx_path, out_dir: str(tmp_path),
    )

    canary.declare(client, "raay_training")
    second = canary.declare(client, "raay_training")

    assert second.name == "ArabicSentiment"
    assert client.aliases[("ArabicSentiment", "Canary")] == "1"


def test_shadow_renders_mirror_conf_and_writes_state(canary, monkeypatch, tmp_path):
    client = _rolling_client()
    conf_path, state_path, reloads = _patch_rollout(canary, monkeypatch, tmp_path)

    canary.cmd_shadow(
        client,
        conf_path=conf_path,
        compose_file="docker-compose.canary.yml",
        state_path=state_path,
        prometheus_url="http://127.0.0.1:9090",
    )

    assert conf_path.read_text() == canary.render_canary_conf("shadow")
    assert reloads == ["docker-compose.canary.yml"]
    state = json.loads(state_path.read_text())
    assert state["phase"] == "shadow"
    assert state["candidate_version"] == 2
    assert state["previous_production_version"] == 1
    assert state["prometheus_url"] == "http://127.0.0.1:9090"
    assert state["entered_at"]


def test_shadow_refuses_without_declared_candidate(canary, monkeypatch, tmp_path):
    client = FakeMlflowClient(
        versions=[_version("ArabicSentiment", 1, {"variant": "onnx-int8"})]
    )
    conf_path, state_path, _ = _patch_rollout(canary, monkeypatch, tmp_path)
    with pytest.raises(SystemExit, match="--mode declare"):
        canary.cmd_shadow(
            client,
            conf_path=conf_path,
            compose_file="f.yml",
            state_path=state_path,
            prometheus_url="http://x",
        )
