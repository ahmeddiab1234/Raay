import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from deploy_staging import (
    SMOKE_CASES,
    Config,
    DeployError,
)

GOOD_SHA = "a" * 40
NEXT_SHA = "b" * 40
REPO = "ahmeddiab1234/arabic-sentiment"
REGISTRY = "ghcr.io"
VERSION = "int8-687d587004c6"


def ok_prediction(label: str, score: float = 0.97) -> dict:
    return {"label": label, "score": score}


def good_predict_body(version: str = VERSION) -> dict:
    """A body that passes every assertion, with the measured real scores.

    The scores are a tuple, positionally paired with SMOKE_CASES on purpose: an
    earlier version of this helper used a set literal here, and set iteration
    order for those floats is not stable across import orders, so the first case
    intermittently received the third case's score and the suite failed for
    reasons that had nothing to do with the code under test.
    """
    scores = (0.97, 0.96, 0.66)
    return {
        "predictions": [
            ok_prediction(case.label, score)
            for case, score in zip(SMOKE_CASES, scores, strict=True)
        ],
        "model_version": version,
    }


class FakeDocker:
    """Records the compose/image commands and answers inspect queries.

    ``running`` is the revision the "deployed" container reports, and
    ``pull_fails`` lets a test simulate a tag that is not in the registry.
    """

    def __init__(self, running: str | None = None, running_state: str = "true"):
        self.running = running
        self.running_state = running_state
        self.calls: list[list[str]] = []
        self.pulled: list[str] = []
        self.logs = "staging log line"
        self.pull_fails = False
        self.version = VERSION
        self.label_revision = None  # override to lie about the deployed revision
        self.dies_on_start = False  # container exists but is not running
        # Die after N refused health probes, i.e. while we are polling: the
        # only way to reach the "container is gone" branch inside the wait loop.
        self.vanish_after_refusals = None

    def __call__(self, cmd, *, env=None, stdin=None, check=True):
        self.calls.append(list(cmd))
        out = ""
        code = 0
        if cmd[:2] == ["docker", "compose"]:
            action = cmd[4] if len(cmd) > 4 else ""
            if action == "pull":
                image = (env or {}).get("RAAY_IMAGE", "")
                if self.pull_fails:
                    return subprocess.CompletedProcess(cmd, 1, "", "manifest unknown")
                self.pulled.append(image)
                self.running = image.rsplit(":", 1)[-1]
                out = ""
            elif action == "up":
                pass
            elif action == "ps":
                out = "cid-123\n" if self.running else ""
            elif action == "logs":
                out = self.logs
        elif cmd[1] == "inspect":
            if "--format" in cmd and "Labels" in cmd[cmd.index("--format") + 1]:
                label = cmd[cmd.index("--format") + 1]
                if "revision" in label:
                    value = self.label_revision or self.running
                elif "version" in label:
                    value = self.version
                else:
                    value = None
                out = f"{value or '<no value>'}\n"
            elif (
                "--format" in cmd and "State.Running" in cmd[cmd.index("--format") + 1]
            ):
                out = f"{'false' if self.dies_on_start else self.running_state}\n"
            else:
                out = f"{self.running_state}\n"
        elif cmd[1] == "login":
            out = "Login Succeeded\n"
        if check and code:
            raise DeployError(f"`{' '.join(cmd)}` exited {code}")
        return subprocess.CompletedProcess(cmd, code, out, "")


class FakeHTTP:
    """Serves canned bodies per endpoint, with per-call overrides.

    ``healthy_revisions`` models the realistic case: the *new* image is what
    fails to come up, and once staging is rolled back the previous image is
    healthy again. Without that distinction a "health never arrives" test would
    also break the rollback and could never observe a successful rollback.
    """

    def __init__(self):
        self.health = {"status": "healthy", "model_version": VERSION}
        self.predict = good_predict_body()
        self.post_status = 200
        self.validation_status = 422
        self.calls: list[tuple[str, str, object]] = []
        self.docker: FakeDocker | None = None
        self.healthy_revisions: set[str] | None = None
        self.refusals = 0

    def __call__(self, url, method, payload, timeout):
        self.calls.append((url, method, payload))
        if url.endswith("/health"):
            if self.health is None:
                self.refusals += 1
                limit = getattr(self.docker, "vanish_after_refusals", None)
                if (
                    self.docker is not None
                    and limit is not None
                    and self.refusals > limit
                ):
                    self.docker.running = None
                raise OSError("connection refused")
            if (
                self.healthy_revisions is not None
                and self.docker is not None
                and self.docker.running not in self.healthy_revisions
            ):
                return 200, {"status": "starting", "model_version": VERSION}
            return 200, self.health
        if url.endswith("/predict"):
            if payload == {"texts": [1]}:
                return self.validation_status, {"detail": "validation error for texts"}
            return self.post_status, self.predict
        raise AssertionError(f"unexpected URL {url}")


@pytest.fixture
def tmp_state(tmp_path):
    return tmp_path / "state" / "deploy-state.json"


def make_cfg(tmp_state, docker, http, **kwargs):
    cfg = Config(
        registry=REGISTRY,
        repository=REPO,
        compose_file="/opt/raay-staging/docker-compose.staging.yml",
        state_file=tmp_state,
        base_url="http://127.0.0.1:8080",
        health_timeout_s=6,
        runner=docker,
        http=http,
        **kwargs,
    )
    return cfg


def no_sleep(_seconds):
    return None


def fake_clock():
    """Monotonic clock that jumps 1s per call so the health timeout is testable."""
    state = {"t": 0.0}

    def tick():
        state["t"] += 1.0
        return state["t"]

    return tick


def deploy(cfg, sha, **kwargs):
    from deploy_staging import deploy as _deploy

    return _deploy(
        cfg,
        sha,
        skip_login=True,
        sleep=no_sleep,
        monotonic=fake_clock(),
        **kwargs,
    )
