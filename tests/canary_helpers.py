from __future__ import annotations

import importlib.util
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CANARY_SCRIPT = ROOT / "scripts" / "canary_promote.py"

# canary_promote.py is a facade; the names the tests replace live in the sibling
# that defines them. Patching the facade's re-export would rebind a name no
# command reads, and the command would attempt a real HTTP call or a real
# `docker compose exec`.
sys.path.insert(0, str(ROOT / "scripts"))

import canary_nginx_ops
import canary_probe


def _past(hours: float = 2.0) -> str:
    """An entered_at timestamp comfortably inside the hold window."""
    return (datetime.now(UTC) - timedelta(hours=hours)).isoformat()


def _load_canary_script():
    spec = importlib.util.spec_from_file_location("canary_promote", CANARY_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    if spec.name not in sys.modules:
        sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


@pytest.fixture()
def canary():
    return _load_canary_script()


class FakeRegisteredVersion:
    def __init__(self, name, version, tags=None):
        self.name = name
        self.version = version
        self.tags = tags or {}
        self.run_id = None

    def __repr__(self):
        return f"FakeModelVersion({self.name} v{self.version})"


class FakeRun:
    def __init__(self, run_id, experiment_id="1"):
        self.info = FakeRunInfo(run_id, experiment_id)


class FakeRunInfo:
    def __init__(self, run_id, experiment_id):
        self.run_id = run_id
        self.experiment_id = experiment_id


class FakeMlflowClient:
    """Minimal registry double: versions list + stage/alias transitions."""

    def __init__(self, runs=None, versions=None, aliases=None):
        self._runs = runs if runs is not None else []
        self._versions = versions if versions is not None else []
        self.transitions = []
        self.aliases = dict(aliases or {})
        self.deleted = []

    def get_experiment_by_name(self, name):
        if name != "raay_training":
            pytest.fail(f"unexpected experiment name: {name}")
        return type("Exp", (), {"experiment_id": "1"})()

    def search_runs(self, experiment_ids, filter_string):
        return self._runs

    def delete_run(self, run_id):
        self.deleted.append(run_id)

    def get_model_version_by_alias(self, name, alias):
        return self.aliases.get((name, alias))

    def search_model_versions(self, filter_string=None):
        if filter_string:
            name = filter_string.split("'")[1]
            return [v for v in self._versions if v.name == name]
        return list(self._versions)

    def register_model(self, run_id=None):
        raise NotImplementedError

    def transition_model_version_stage(
        self, name, version, stage, archive_existing_versions
    ):
        self.transitions.append((name, version, stage, archive_existing_versions))

    def set_registered_model_alias(self, name, alias, version):
        self.aliases[(name, alias)] = version

    def set_model_version_tag(self, name, version, key, value):
        pass


def _version(name, version, tags):
    v = FakeRegisteredVersion(name, version, tags)
    v.run_id = f"run-{name}-{version}"
    return v


def _fake_start_run(**kwargs):
    class _CM:
        def __init__(self):
            self.info = FakeRunInfo(f"new-{kwargs.get('run_name', 'run')}", "1")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    return _CM()


def _rolling_client():
    return FakeMlflowClient(
        aliases={("ArabicSentiment", "Production"): "1"},
        versions=[
            _version("ArabicSentiment", 1, {"variant": "onnx-int8"}),
            _version("ArabicSentiment", 2, {"variant": "distilled-fp32"}),
        ],
    )


def _paths(canary, tmp_path):
    return (
        tmp_path / "nginx_canary.conf",
        tmp_path / "canary_state.json",
    )


def _patch_rollout(canary, monkeypatch, tmp_path):
    conf_path, state_path = _paths(canary, tmp_path)
    monkeypatch.setattr(canary_probe, "_health_gate", lambda url: None)
    reloads = []
    monkeypatch.setattr(
        canary_nginx_ops, "_reload_nginx", lambda cf: reloads.append(cf)
    )
    monkeypatch.setattr(
        canary_nginx_ops, "_write_gate_report", lambda report, phase: Path("/dev/null")
    )
    return conf_path, state_path, reloads


def _advance_env(canary, monkeypatch, tmp_path, state):
    conf_path, state_path = _paths(canary, tmp_path)
    if state is not None:
        state_path.write_text(json.dumps(state))
    monkeypatch.setattr(canary_probe, "_health_gate", lambda url: None)
    reloads = []
    monkeypatch.setattr(
        canary_nginx_ops, "_reload_nginx", lambda cf: reloads.append(cf)
    )
    monkeypatch.setattr(
        canary_nginx_ops, "_write_gate_report", lambda report, phase: Path("/dev/null")
    )
    return conf_path, state_path, reloads


def _state(phase="shadow", candidate=2, previous=1, entered=None):
    return {
        "phase": phase,
        "candidate_version": candidate,
        "previous_production_version": previous,
        "prometheus_url": "http://127.0.0.1:9090",
        "entered_at": entered if entered is not None else _past(),
    }


def _good_query(requests=600, p95_c=42.0, p95_s=40.0, agree=595.0, disagree=5.0):
    def query(promql):
        if "raay_ingest_total" in promql:
            return requests if '"candidate"' in promql else None
        if "raay_errors_total" in promql:
            return 0.0
        if "raay_latency_seconds_bucket" in promql:
            return p95_c if '"candidate"' in promql else p95_s
        if "raay_agreement_total" in promql:
            return agree if 'status="agree"' in promql else disagree
        return None

    return query
