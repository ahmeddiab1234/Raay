"""Config, the report, and the two host adapters (docker runner, HTTP client).

Both adapters are injectable and both default to something real, which is what
lets ``tests/test_deploy_staging.py`` exercise the whole deploy -- including the
rollback paths -- with a fake runner and no Docker daemon on the box.
"""

from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_COMPOSE_FILE = "/opt/raay-staging/docker-compose.staging.yml"
DEFAULT_STATE_FILE = "/var/lib/raay-staging/deploy-state.json"
DEFAULT_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_HEALTH_TIMEOUT_S = 180
SERVICE = "raay-sentiment"
VALID_LABELS = ("positive", "negative", "neutral")
RESPONSE_KEYS = {"predictions", "model_version"}
PREDICTION_KEYS = {"label", "score"}
REVISION_LABEL = "org.opencontainers.image.revision"
VERSION_LABEL = "org.opencontainers.image.version"
HISTORY_LIMIT = 10
LOG_TAIL_LINES = 100
HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class SmokeCase:
    """One fixed review and the label it must produce."""

    text: str
    label: str
    min_score: float


Runner = Callable[..., subprocess.CompletedProcess[str]]
HttpCall = Callable[[str, str, dict[str, Any] | None, int], tuple[int, Any]]


class DeployError(RuntimeError):
    """A deploy step failed. The message is meant to be readable in a job log."""


def default_runner(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    stdin: str | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        cmd, env=env, input=stdin, capture_output=True, text=True, check=False
    )
    if check and proc.returncode != 0:
        raise DeployError(
            f"`{' '.join(cmd)}` exited {proc.returncode}\n"
            f"stdout: {proc.stdout.strip()}\nstderr: {proc.stderr.strip()}"
        )
    return proc


def default_http(
    url: str, method: str, payload: dict[str, Any] | None, timeout: int
) -> tuple[int, Any]:
    """Return ``(status, parsed_body)``. A non-2xx is a value, not an exception."""
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read() or b"null")
    except urllib.error.HTTPError as exc:
        body = exc.read()
        try:
            return exc.code, json.loads(body or b"null")
        except ValueError:
            return exc.code, None


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class Config:
    """Everything a deploy needs to talk to one staging host."""

    registry: str
    repository: str
    compose_file: str = DEFAULT_COMPOSE_FILE
    state_file: Path = Path(DEFAULT_STATE_FILE)
    base_url: str = DEFAULT_BASE_URL
    health_timeout_s: int = DEFAULT_HEALTH_TIMEOUT_S
    expect_model_version: str | None = None
    runner: Runner = default_runner
    http: HttpCall = default_http

    def image_for(self, sha: str) -> str:
        """Build the immutable ``registry/repository:sha`` reference.

        Refusing anything that is not a lowercase hex SHA is deliberate: the
        whole point of pinning is that staging cannot be retargeted by a moving
        tag, and ``latest`` would quietly break that. The length check matters
        as much as the alphabet -- ``abc`` is hex but is not a git SHA.
        """
        if len(sha) < 7 or not HEX.issuperset(sha.lower()):
            raise DeployError(
                f"target {sha!r} is not a git SHA; staging deploys immutable tags only"
            )
        if not self.registry or not self.repository:
            raise DeployError("a registry and repository are required")
        return f"{self.registry.rstrip('/')}/{self.repository.strip('/')}:{sha}"

    def compose_env(self, image: str) -> dict[str, str]:
        """The compose file refuses to interpolate an empty ``RAAY_IMAGE``."""
        return {**os.environ, "RAAY_IMAGE": image}

    def compose(self, *args: str) -> list[str]:
        return ["docker", "compose", "-f", self.compose_file, *args]

    def container_id(self, image: str) -> str | None:
        proc = self.runner(
            self.compose("ps", "-q", SERVICE), env=self.compose_env(image)
        )
        return proc.stdout.strip() or None

    def label(self, image: str, label: str) -> str | None:
        """Read an OCI label off the *running container*.

        Reading the container rather than the tag answers the only question a
        rollback cares about: what is actually running right now.
        """
        cid = self.container_id(image)
        if cid is None:
            return None
        proc = self.runner(
            [
                "docker",
                "inspect",
                "--format",
                f'{{{{ index .Config.Labels "{label}" }}}}',
                cid,
            ],
            check=False,
        )
        if proc.returncode != 0:
            return None
        value = proc.stdout.strip()
        return value if value and value != "<no value>" else None

    def logs(self, image: str) -> str:
        proc = self.runner(
            self.compose("logs", "--no-color", "--tail", str(LOG_TAIL_LINES)),
            env=self.compose_env(image),
            check=False,
        )
        return (proc.stdout + proc.stderr).strip()


@dataclass
class Report:
    """Machine-readable outcome: printed as JSON and folded into the summary."""

    target_sha: str
    image: str = ""
    previous_sha: str | None = None
    outcome: str = "failed"
    rolled_back_to: str | None = None
    checks: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    logs: str = ""

    @property
    def ok(self) -> bool:
        return self.outcome in {"deployed", "already_deployed"}

    def as_dict(self) -> dict[str, Any]:
        return {
            "target_sha": self.target_sha,
            "image": self.image,
            "previous_sha": self.previous_sha,
            "outcome": self.outcome,
            "rolled_back_to": self.rolled_back_to,
            "checks": self.checks,
            "errors": self.errors,
        }
