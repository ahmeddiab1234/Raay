"""Health probes and the Prometheus query used by the stage gates.

nginx owns :8000 through the rollout, so each worker is also published on its
own direct port; the operator can then gate it independently of whatever the
weighted routing happens to be doing.

A missing Prometheus series is ``None``, never zero. The caller treats ``None`` as
INCONCLUSIVE, so an empty canary cannot read as a pass.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from collections.abc import Iterable
from urllib.error import URLError

from loguru import logger

# Health gates. nginx owns :8000 through the rollout, so each worker is also
# published on its own direct port so the operator can gate it independently of
# any weighted routing.
_STABLE_URL = "http://127.0.0.1:8002/health"
_CANDIDATE_URL = "http://127.0.0.1:8001/health"
_FRONT_URL = "http://127.0.0.1:8081/health"
_INGRESS_URL = "http://127.0.0.1:8000/health"
_AGENT_URL = "http://127.0.0.1:9100/health"
_PROMETHEUS_HEALTH_URL = "http://127.0.0.1:9090/-/healthy"


def _health_gate(url: str) -> None:
    """Fail loudly if a worker healthcheck is not 200."""
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            body = resp.read().decode("utf-8", "replace")
            status = resp.status
    except URLError as exc:
        raise SystemExit(f"Health gate FAILED ({url}): {exc}") from exc
    if status != 200:
        raise SystemExit(f"Health gate FAILED ({url}): HTTP {status}")
    logger.info(f"Health gate OK ({url}): {body.strip()}")


def _health_gates(urls: Iterable[str]) -> None:
    for url in urls:
        _health_gate(url)


def _promql_query(base_url: str, promql: str) -> float | None:
    """Instant query against the Prometheus HTTP API; None = no samples/error.

    A missing series must be distinguishable from a failed scrape: the caller
    treats ``None`` as INCONCLUSIVE so an empty canary never reads as a pass.
    """
    url = (
        f"{base_url.rstrip('/')}/api/v1/query?"
        f"{urllib.parse.urlencode({'query': promql})}"
    )
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            payload = json.loads(resp.read().decode("utf-8", "replace"))
    except Exception as exc:  # noqa: BLE001 - a dead Prometheus is INCONCLUSIVE, not a crash
        logger.warning(f"Prometheus query not answered: {exc}")
        return None
    if payload.get("status") != "success":
        logger.warning(f"Prometheus query error: {payload.get('error')}")
        return None
    result = (payload.get("data") or {}).get("result") or []
    if not result:
        return None
    try:
        return float(result[0]["value"][1])
    except (KeyError, IndexError, TypeError, ValueError):
        return None
