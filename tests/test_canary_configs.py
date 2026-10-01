"""Hermetic tests for the Phase 5 step 5 shadow/canary rollout configs.

Covers:
* the ``render_canary_conf`` renderer: mirror/headers/weights per stage, no
  ``X-Request-ID`` header (a mirror subrequest cannot share ``$request_id``),
  and the committed ``deploy/nginx_canary.conf`` pinned to the shadow default.
* ``docker-compose.canary.yml`` + ``deploy/prometheus.yml`` text contracts.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NGINX_CONF = ROOT / "deploy" / "nginx_canary.conf"
CANARY_COMPOSE = ROOT / "docker-compose.canary.yml"
PROMETHEUS_CONF = ROOT / "deploy" / "prometheus.yml"

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
