from __future__ import annotations

from typing import Any

"""Pull the target image, wait for health, run smoke tests, and roll back on failure.

Rollback is the most important path here. The label cross-check (the running
container's ``org.opencontainers.image.revision`` must equal the target SHA, and
``/health``+``/predict``'s ``model_version`` must equal the image's
``org.opencontainers.image.version``) is what catches a stale-but-healthy image.
A failed deploy always returns staging to the last smoke-tested SHA, unless
there has never been one -- in which case it reports a failed rollback and leaves
the box in the failed state, instead of promoting ``latest`` by accident.

The state file is written atomically and lives outside the checkout so a re-copy
of the scripts cannot wipe it.
"""


import time
import urllib.error
from collections.abc import Callable
from pathlib import Path

from staging_smoke import (
    run_smoke,
    wait_for_health,
)
from staging_state import _record, read_state, write_state
from staging_types import (
    REVISION_LABEL,
    VERSION_LABEL,
    Config,
    DeployError,
    Report,
)


def login(cfg: Config, user: str, token: str) -> None:
    """``docker login`` over stdin, so the token never reaches argv or a log."""
    if not token:
        raise DeployError(
            "no registry token: set RAAY_REGISTRY_TOKEN or pass --registry-token-file"
        )
    cfg.runner(
        ["docker", "login", cfg.registry, "-u", user, "--password-stdin"], stdin=token
    )


def pull_and_up(cfg: Config, image: str, report: Report) -> None:
    env = cfg.compose_env(image)
    report.checks.append(f"docker compose pull {image}")
    cfg.runner(cfg.compose("pull"), env=env)
    cfg.runner(cfg.compose("up", "-d", "--remove-orphans"), env=env)


def rollback(
    cfg: Config,
    report: Report,
    state: dict[str, Any],
    state_file: Path,
    sleep: Callable[[float], None],
    monotonic: Callable[[], float],
) -> Report:
    """Put the last smoke-tested SHA back. Never promotes anything new."""
    target = report.previous_sha
    if not target or target == report.target_sha:
        report.outcome = "failed"
        report.errors.append(
            "no rollback target: nothing has ever passed the smoke test on this "
            "host, so there is no previous good version to return to"
        )
        return report

    image = cfg.image_for(target)
    report.checks.append(f"rolling back to {target}")
    try:
        pull_and_up(cfg, image, report)
        if cfg.label(image, REVISION_LABEL) != target:
            raise DeployError(
                f"the rolled-back container reports revision "
                f"{cfg.label(image, REVISION_LABEL)!r}, expected {target!r}"
            )
        wait_for_health(cfg, image, report, sleep, monotonic)
    except (DeployError, urllib.error.URLError, OSError, ValueError) as exc:
        report.outcome = "failed"
        report.errors.append(f"ROLLBACK FAILED: {exc}")
        return report

    report.outcome = "rolled_back"
    report.rolled_back_to = target
    report.errors.append(f"smoke test failed; staging is back on {target}")
    # last_known_good stays exactly as it was: the previous release is still the
    # newest thing known to work, and therefore still the rollback target.
    _record(state, report.target_sha, "rolled_back")
    write_state(state_file, state)
    return report


def deploy(
    cfg: Config,
    target_sha: str,
    *,
    registry_user: str = "github-actions",
    registry_token: str = "",
    skip_login: bool = False,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> Report:
    image = cfg.image_for(target_sha)
    report = Report(target_sha=target_sha, image=image)
    state = read_state(cfg.state_file)
    remembered = state.get("last_known_good")
    report.previous_sha = remembered if isinstance(remembered, str) else None

    if not skip_login:
        login(cfg, registry_user, registry_token)
        report.checks.append(f"docker login {cfg.registry}")

    running = cfg.label(image, REVISION_LABEL)
    if report.previous_sha is None and running and running != target_sha:
        # Bootstrap the state file from the container that is already serving.
        report.previous_sha = running
    report.checks.append(f"previously running revision: {running or 'none'}")

    if not skip_login:
        state["last_attempt"] = target_sha
        _record(state, target_sha, "attempted")
        write_state(cfg.state_file, state)

    try:
        if running == target_sha:
            report.checks.append("already on the target SHA; re-verifying in place")
        else:
            pull_and_up(cfg, image, report)
        revision = cfg.label(image, REVISION_LABEL)
        if revision != target_sha:
            raise DeployError(
                f"the deployed container reports revision {revision!r}, "
                f"expected {target_sha!r}"
            )
        report.checks.append("deployed image label matches the target SHA")
        if cfg.expect_model_version is None:
            cfg.expect_model_version = cfg.label(image, VERSION_LABEL)
        run_smoke(cfg, image, report, sleep, monotonic)
    except (DeployError, urllib.error.URLError, OSError, ValueError) as exc:
        report.errors.append(str(exc))
        report.logs = cfg.logs(image)
        return rollback(cfg, report, state, cfg.state_file, sleep, monotonic)

    state["last_known_good"] = target_sha
    state["last_attempt"] = target_sha
    _record(state, target_sha, "deployed")
    write_state(cfg.state_file, state)
    report.outcome = (
        "deployed" if report.previous_sha != target_sha else "already_deployed"
    )
    return report
