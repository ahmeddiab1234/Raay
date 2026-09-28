"""Hermetic tests for the Phase 5 step 5 shadow/canary rollout.

Covers:

* the ``render_canary_conf`` renderer: mirror/headers/weights per stage, no
  ``X-Request-ID`` header (a mirror subrequest cannot share ``$request_id``),
  and the committed ``deploy/nginx_canary.conf`` pinned to the shadow default.
* ``docker-compose.canary.yml`` + ``deploy/prometheus.yml`` text contracts.
* ``scripts/canary_promote.py`` rollout semantics against a **fake** MLflow
  client, fake health checks, a fake nginx reload and a fake Prometheus query
  fn (no registry server, no docker, no nginx): shadow writes the state file
  and mirrors, advance gates then widens (or refuses on FAIL/INCONCLUSIVE),
  full shells out to the Phase 6 offline gate, rollback restores stable-only
  weights and the pre-rollout alias.
* the gate math itself: error rate, p95 ratio, agreement, hold, empty-series
  INCONCLUSIVE handling.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.error import URLError

import pytest

ROOT = Path(__file__).resolve().parents[1]
NGINX_CONF = ROOT / "deploy" / "nginx_canary.conf"
CANARY_COMPOSE = ROOT / "docker-compose.canary.yml"
PROMETHEUS_CONF = ROOT / "deploy" / "prometheus.yml"
CANARY_SCRIPT = ROOT / "scripts" / "canary_promote.py"


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


# ------------------------------------------------------------ conf renderer


def test_render_shadow_conf_mirrors_every_predict(canary):
    text = canary.render_canary_conf("shadow")
    assert "server raay-sentiment:3000 weight=100;" in text
    assert "server raay-sentiment-canary:3000 weight=0;" in text
    assert "listen 8000;" in text
    assert "listen 8081;" in text
    assert "mirror /shadow;" in text
    assert "location /shadow" in text
    assert "proxy_pass http://raay-sentiment-canary:3000/predict;" in text
    # Both the main hop and the mirror hop are marked shadowed.
    assert text.count("proxy_set_header X-Raay-Shadow 1;") == 2


def test_render_shadow_conf_never_injects_request_id(canary):
    text = canary.render_canary_conf("shadow")
    # A mirror subrequest is a separate request object, so $request_id differs
    # between the workers and X-Request-ID would split the pair. Correlation is
    # content-based in the agent instead.
    assert "X-Request-ID" not in text
    assert "$request_id" not in text


@pytest.mark.parametrize(
    "phase,stable,candidate",
    [
        ("canary-5", 95, 5),
        ("canary-25", 75, 25),
        ("canary-50", 50, 50),
        ("full", 0, 100),
    ],
)
def test_render_weighted_phases_have_no_mirror(canary, phase, stable, candidate):
    text = canary.render_canary_conf(phase)
    assert f"server raay-sentiment:3000 weight={stable};" in text
    assert f"server raay-sentiment-canary:3000 weight={candidate};" in text
    assert "mirror /shadow;" not in text
    assert "location /shadow" not in text
    assert "X-Raay-Shadow" not in text


def test_render_rollback_conf_is_stable_only(canary):
    text = canary.render_canary_conf("rollback")
    assert "server raay-sentiment:3000 weight=100;" in text
    assert "server raay-sentiment-canary:3000 weight=0;" in text
    assert "mirror /shadow;" not in text
    assert "X-Raay-Shadow" not in text


def test_all_stage_weights_sum_to_100(canary):
    for phase in ("shadow", "canary-5", "canary-25", "canary-50", "full", "rollback"):
        stable, candidate = canary._weights_for(phase)
        assert stable + candidate == 100


def test_committed_nginx_conf_is_the_shadow_default(canary):
    assert NGINX_CONF.read_text() == canary.render_canary_conf("shadow")


# ------------------------------------------------------------ docker-compose


def test_canary_compose_has_standalone_project(canary):
    text = CANARY_COMPOSE.read_text()
    assert "name: canary" in text
    assert "\n  raay-sentiment:" in text
    assert "\n  raay-sentiment-canary:" in text
    assert "\n  raay-nginx:" in text
    assert "\n  raay-canary-agent:" in text
    assert "\n  prometheus:" in text
    assert "    build:" not in text  # a rollout must never rebuild from the repo


def test_canary_compose_publishes_nginx_ingress_and_front(canary):
    text = CANARY_COMPOSE.read_text()
    assert '"8000:8000"' in text
    assert '"8081:8081"' in text


def test_canary_compose_publishes_direct_worker_health_ports(canary):
    # nginx owns :8000, so each worker gets its own direct port so the rollout
    # can health-gate it without any weighted routing in between.
    text = CANARY_COMPOSE.read_text()
    assert '"8001:3000"' in text  # candidate
    assert '"8002:3000"' in text  # stable


def test_canary_compose_candidate_binds_distilled_graph_read_only(canary):
    text = CANARY_COMPOSE.read_text()
    assert "./models/onnx/distilled.onnx:" in text
    assert "./models/onnx/distilled.onnx.data:" in text
    assert ":ro" in text


def test_canary_compose_sets_worker_and_telemetry_on_both_workers(canary):
    text = CANARY_COMPOSE.read_text()
    assert "RAAY_WORKER: stable" in text
    assert "RAAY_WORKER: candidate" in text
    assert "RAAY_TELEMETRY_URL: http://raay-canary-agent:9100/ingest" in text


def test_canary_compose_nginx_mounts_generated_conf(canary):
    text = CANARY_COMPOSE.read_text()
    assert "./deploy/nginx_canary.conf:/etc/nginx/conf.d/canary.conf:ro" in text


def test_prometheus_conf_scrapes_the_agent(canary):
    text = PROMETHEUS_CONF.read_text()
    assert 'targets: ["raay-canary-agent:9100"]' in text
    assert "scrape_interval: 5s" in text


def test_prod_compose_has_no_canary_front(canary):
    # The rollout owns its own nginx via docker-compose.canary.yml; the prod
    # compose must not double as a shadow front (the same generated conf would
    # be rewritten by a mid-rollout reload).
    text = (ROOT / "docker-compose.yml").read_text()
    assert "raay-nginx" not in text
    assert "raay-sentiment-canary" not in text
    assert "8081" not in text


# ------------------------------------------------ canary_promote semantics


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
    monkeypatch.setattr(canary, "_health_gate", lambda url: None)
    reloads = []
    monkeypatch.setattr(canary, "_reload_nginx", lambda cf: reloads.append(cf))
    monkeypatch.setattr(
        canary, "_write_gate_report", lambda report, phase: Path("/dev/null")
    )
    return conf_path, state_path, reloads


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


# -------------------------------------------------------------------- shadow


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


# ------------------------------------------------------------------- advance


def _advance_env(canary, monkeypatch, tmp_path, state):
    conf_path, state_path = _paths(canary, tmp_path)
    if state is not None:
        state_path.write_text(json.dumps(state))
    monkeypatch.setattr(canary, "_health_gate", lambda url: None)
    reloads = []
    monkeypatch.setattr(canary, "_reload_nginx", lambda cf: reloads.append(cf))
    monkeypatch.setattr(
        canary, "_write_gate_report", lambda report, phase: Path("/dev/null")
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
    monkeypatch.setattr(canary, "_promql_query", lambda _url, promql: q(promql))

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
    monkeypatch.setattr(canary, "_promql_query", lambda _url, promql: q(promql))

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
    monkeypatch.setattr(canary, "_promql_query", lambda _url, promql: q(promql))

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
    monkeypatch.setattr(canary, "_promql_query", lambda _url, promql: q(promql))
    offline_calls = []
    monkeypatch.setattr(canary, "_run_offline_gate", lambda v: offline_calls.append(v))

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
    monkeypatch.setattr(canary, "_promql_query", lambda _url, promql: q(promql))

    def abort(_version):
        raise canary.subprocess.CalledProcessError(1, ["promote_model.py"])

    monkeypatch.setattr(canary, "_run_offline_gate", abort)

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


# ------------------------------------------------------------------ rollback


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


# ----------------------------------------------------------------- gate math


def test_gate_passes_shadow_stage(canary):
    outcome = canary.evaluate_gate(
        "shadow",
        query=_good_query(),
        entered_at=_past(),
        hold_seconds=600,
    )
    assert outcome.passed
    assert not outcome.inconclusive
    assert outcome.observed["agreement_agree"] == 595.0


def test_gate_skips_agreement_outside_shadow(canary):
    outcome = canary.evaluate_gate(
        "canary-5",
        query=_good_query(agree=None, disagree=None),
        entered_at=_past(),
        hold_seconds=600,
    )
    # Weighted canaries have no mirror, so agreement is undefined by design and
    # NOT an inconclusive failure.
    assert outcome.passed


def test_gate_fails_on_error_rate(canary):
    outcome = canary.evaluate_gate(
        "canary-5",
        query=_good_query(requests=600),
        entered_at=_past(),
        hold_seconds=600,
    )
    assert outcome.passed  # 0 errors

    def _q(promql):
        if "raay_errors_total" in promql:
            return 30.0
        return _good_query()(promql)

    outcome_bad = canary.evaluate_gate(
        "canary-5",
        query=_q,
        entered_at=_past(),
        hold_seconds=600,
    )
    assert not outcome_bad.passed
    assert not outcome_bad.inconclusive
    assert "error rate" in outcome_bad.reason


def test_gate_fails_on_p95_ratio(canary):
    def _q(promql):
        if "raay_latency_seconds_bucket" in promql:
            return 220.0 if '"candidate"' in promql else 40.0
        return _good_query()(promql)

    outcome = canary.evaluate_gate(
        "canary-5",
        query=_q,
        entered_at=_past(),
        hold_seconds=600,
    )
    assert not outcome.passed
    assert "p95" in outcome.reason


def test_gate_fails_on_low_agreement(canary):
    outcome = canary.evaluate_gate(
        "shadow",
        query=_good_query(agree=300.0, disagree=300.0),
        entered_at=_past(),
        hold_seconds=600,
    )
    assert not outcome.passed
    assert "agreement" in outcome.reason


def test_gate_inconclusive_with_zero_requests(canary):
    outcome = canary.evaluate_gate(
        "canary-5",
        query=_good_query(requests=0),
        entered_at=_past(),
        hold_seconds=600,
    )
    assert outcome.inconclusive
    assert not outcome.passed


def test_gate_inconclusive_with_missing_p95(canary):
    outcome = canary.evaluate_gate(
        "canary-5",
        query=_good_query(p95_c=None),
        entered_at=_past(),
        hold_seconds=600,
    )
    assert outcome.inconclusive
    assert not outcome.passed


def test_gate_inconclusive_with_zero_paired_traffic(canary):
    outcome = canary.evaluate_gate(
        "shadow",
        query=_good_query(agree=None, disagree=None),
        entered_at=_past(),
        hold_seconds=600,
    )
    assert outcome.inconclusive


def test_gate_fails_below_min_requests(canary):
    outcome = canary.evaluate_gate(
        "canary-5",
        query=_good_query(requests=10),
        entered_at=_past(),
        hold_seconds=600,
        min_requests=500,
    )
    assert not outcome.passed
    assert not outcome.inconclusive
    assert "--min-requests" in outcome.reason


# ------------------------------------------------------------ promql parsing


def test_promql_query_parses_instant_vector_value(canary, monkeypatch):
    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b'{"status":"success","data":{"result":[{"metric":{},"value":[1,"42.5"]}]}}'

    monkeypatch.setattr(canary.urllib.request, "urlopen", lambda url, timeout: _Resp())
    assert canary._promql_query("http://127.0.0.1:9090", "up") == 42.5


def test_promql_query_missing_series_returns_none(canary, monkeypatch):
    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b'{"status":"success","data":{"result":[]}}'

    monkeypatch.setattr(canary.urllib.request, "urlopen", lambda url, timeout: _Resp())
    assert canary._promql_query("http://127.0.0.1:9090", "up") is None


def test_promql_query_network_error_returns_none(canary, monkeypatch):
    def boom(url, timeout):
        raise OSError("refused")

    monkeypatch.setattr(canary.urllib.request, "urlopen", boom)
    assert canary._promql_query("http://127.0.0.1:9090", "up") is None
