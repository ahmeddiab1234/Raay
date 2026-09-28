"""Hermetic tests for the Phase 5 step 4 canary rollout.

Covers:

* ``deploy/nginx_canary.conf`` text contract: the 95/5 weighted split, both
  upstream workers, the ``:8081`` listen, and result-passthrough for
  ``/predict`` + ``/health``. No nginx binary is needed -- we assert on the
  conf text exactly as compose mounts it.
* ``docker-compose.yml`` text contract: the canary worker and the nginx front
  service exist, bind the distilled graph read-only, and publish ``:8081``.
* ``scripts/canary_promote.py`` rollout semantics against a **fake** MLflow
  client (no tracking server touched): declare registers a ``Canary`` alias
  idempotently, promote flips ``Production`` to the distilled fp32 version
  (archiving int8), rollback returns ``Production`` to int8.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NGINX_CONF = ROOT / "deploy" / "nginx_canary.conf"
COMPOSE = ROOT / "docker-compose.yml"
CANARY_SCRIPT = ROOT / "scripts" / "canary_promote.py"

_WD_WEIGHTS = (
    "server raay-sentiment:3000 weight=95;",
    "server raay-sentiment-canary:3000 weight=5;",
)


def _load_canary_script():
    spec = importlib.util.spec_from_file_location("canary_promote", CANARY_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


@pytest.fixture()
def canary():
    return _load_canary_script()


# ---------------------------------------------------------------- nginx conf


def test_nginx_conf_weighted_split_has_both_workers(canary):
    text = NGINX_CONF.read_text()
    for line in _WD_WEIGHTS:
        assert line in text


def test_nginx_conf_weights_sum_to_100(canary):
    text = NGINX_CONF.read_text()
    weights = [
        int(line.split("weight=")[1].rstrip(";").strip())
        for line in text.splitlines()
        if "weight=" in line
    ]
    assert sum(weights) == 100
    assert weights == [95, 5]


def test_nginx_conf_listens_on_canary_front_port(canary):
    assert "listen 8081;" in NGINX_CONF.read_text()


def test_nginx_conf_proxies_predict_and_health(canary):
    text = NGINX_CONF.read_text()
    assert "http://raay_sentiment_backend/predict" in text
    assert "http://raay_sentiment_backend/health" in text


def test_nginx_conf_keeps_prod_direct_port_untouched(canary):
    text = NGINX_CONF.read_text()
    assert "listen 8000" not in text
    assert "8000" not in NGINX_CONF.read_text().splitlines()


# ------------------------------------------------------------ docker-compose


def test_compose_adds_canary_worker_and_nginx_services(canary):
    text = COMPOSE.read_text()
    assert "  raay-sentiment-canary:" in text
    assert "  raay-nginx:" in text


def test_compose_canary_worker_binds_distilled_graph_ro(canary):
    text = COMPOSE.read_text()
    assert "./models/onnx/distilled.onnx:" in text
    assert "./models/onnx/distilled.onnx.data:" in text
    assert "RAAY_ALIAS: Canary" in text


def test_compose_nginx_publishes_canary_front_port(canary):
    assert '"8081:8081"' in COMPOSE.read_text()


# ------------------------------------------------- canary_promote semantics


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

    def __init__(self, runs=None, versions=None):
        self._runs = runs if runs is not None else []
        self._versions = versions if versions is not None else []
        self.transitions = []
        self.aliases = {}
        self.deleted = []

    def get_experiment_by_name(self, name):
        if name != "raay_training":
            pytest.fail(f"unexpected experiment name: {name}")
        return type("Exp", (), {"experiment_id": "1"})()

    def search_runs(self, experiment_ids, filter_string):
        return self._runs

    def delete_run(self, run_id):
        self.deleted.append(run_id)

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
    assert client.aliases[("ArabicSentiment", "Canary")] == 3
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
    second = canary.declare(client, "raay_training")  # run again

    assert second.name == "ArabicSentiment"
    assert client.aliases[("ArabicSentiment", "Canary")] == 1


def test_promote_flips_production_to_distilled(canary, monkeypatch):
    client = FakeMlflowClient(
        versions=[
            _version("ArabicSentiment", 1, {"variant": "onnx-int8"}),
            _version("ArabicSentiment", 2, {"variant": "distilled-fp32"}),
        ]
    )
    monkeypatch.setattr(canary, "_health_gate", lambda url: None)

    canary.promote(client)

    assert ("ArabicSentiment", 2, "Production", True) in client.transitions
    assert client.aliases[("ArabicSentiment", "Production")] == 2


def test_promote_refuses_without_canary_declared(canary, monkeypatch):
    client = FakeMlflowClient(
        versions=[_version("ArabicSentiment", 1, {"variant": "onnx-int8"})]
    )
    monkeypatch.setattr(canary, "_health_gate", lambda url: None)

    with pytest.raises(SystemExit, match="--mode declare"):
        canary.promote(client)


def test_rollback_returns_production_to_int8(canary, monkeypatch):
    client = FakeMlflowClient(
        versions=[
            _version("ArabicSentiment", 1, {"variant": "onnx-int8"}),
            _version("ArabicSentiment", 2, {"variant": "distilled-fp32"}),
        ]
    )
    monkeypatch.setattr(canary, "_health_gate", lambda url: None)

    canary.rollback(client)

    assert ("ArabicSentiment", 1, "Production", True) in client.transitions
    assert client.aliases[("ArabicSentiment", "Production")] == 1


def test_health_gate_fails_loudly_on_connection_error(canary, monkeypatch):
    import urllib.error
    import urllib.request

    def boom(url, timeout):
        raise urllib.error.URLError("refused")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(SystemExit, match="Health gate FAILED"):
        canary._health_gate("http://127.0.0.1:9/health")
