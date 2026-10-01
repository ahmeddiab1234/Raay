"""Waiting for the service to come up on a staging host.

The container-died check is the part that matters: without it a crash-looping
release burns the whole ``--health-timeout`` before anyone learns why.
"""

from __future__ import annotations

import time
import urllib.error
from collections.abc import Callable
from typing import Any

from staging_types import SERVICE, Config, DeployError, Report


def wait_for_health(
    cfg: Config,
    image: str,
    report: Report,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Poll ``/health`` until it answers, the container dies, or time runs out.

    The container-died check matters: without it a crash-looping release burns
    the whole timeout before anyone learns why.
    """
    deadline = monotonic() + cfg.health_timeout_s
    last = "no response yet"
    while monotonic() < deadline:
        try:
            status, body = cfg.http(f"{cfg.base_url}/health", "GET", None, 5)
            if status == 200 and isinstance(body, dict):
                report.checks.append("/health answered 200")
                return body
            last = f"status={status} body={body!r}"
        except (urllib.error.URLError, OSError, ValueError) as exc:
            last = f"{type(exc).__name__}: {exc}"
        if cfg.container_id(image) is None:
            raise DeployError(f"the {SERVICE} container is gone; last probe: {last}")
        running = cfg.runner(
            [
                "docker",
                "inspect",
                "--format",
                "{{.State.Running}}",
                cfg.container_id(image),
            ],
            check=False,
        )
        if running.stdout.strip() != "true":
            raise DeployError(
                f"the {SERVICE} container is not running; last probe: {last}"
            )
        sleep(2.0)
    raise DeployError(
        f"{SERVICE} was not healthy within {cfg.health_timeout_s}s; last probe: {last}"
    )
