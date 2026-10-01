"""Shared helpers for CD workflow and staging tests."""

from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
WORKFLOW = REPO / ".github" / "workflows" / "cd.yml"
BENTOFILE = REPO / "bentofile.yaml"


@pytest.fixture(scope="module")
def workflow():
    return yaml.safe_load(WORKFLOW.read_text())


def _steps(workflow, job="docker-build"):
    return workflow["jobs"][job]["steps"]


def _run_commands(workflow, job="docker-build"):
    return "\n".join(s["run"] for s in _steps(workflow, job) if "run" in s)


def _step(workflow, prefix, job="docker-build"):
    return next(
        s for s in _steps(workflow, job) if s.get("name", "").startswith(prefix)
    )
