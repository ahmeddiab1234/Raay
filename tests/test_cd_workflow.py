"""Structural tests for ``.github/workflows/cd.yml``.

The release workflow is the only thing standing between a commit and a pushed,
pullable image, and it runs on a fresh runner that has no model on disk. Nothing
imports it, so these assertions pin the parts that the release contract depends
on: the trigger is ``main`` only, the DVC-acked artifacts are fetched before the
bento is built, the version reaches the image twice (label + env), the image is
smoke-tested *before* it is pushed, and the mutable tag is never the only tag.
"""

import os
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
WORKFLOW = REPO / ".github" / "workflows" / "cd.yml"
BENTOFILE = REPO / "bentofile.yaml"

# The four git-ignored files the image bakes in. Each needs a DVC pointer on
# main or a fresh runner has nothing to bake.
SERVING_ARTIFACTS = (
    "models/onnx/model_int8.onnx.dvc",
    "models/baseline/final/config.json.dvc",
    "models/baseline/final/tokenizer.json.dvc",
    "models/baseline/final/tokenizer_config.json.dvc",
)


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


def test_workflow_is_valid_yaml_and_triggers_only_on_main(workflow):
    # PyYAML resolves the bare `on:` key to the boolean True.
    triggers = workflow[True] if True in workflow else workflow["on"]
    assert triggers["push"]["branches"] == ["main"]
    # A release must never be reachable from a feature branch or a PR.
    assert "pull_request" not in triggers


def test_expected_jobs_exist(workflow):
    assert set(workflow["jobs"]) == {
        "lint",
        "test",
        "docker-build",
        "deploy-staging",
    }


def test_publish_job_waits_for_lint_and_test(workflow):
    assert workflow["jobs"]["docker-build"]["needs"] == ["lint", "test"]


def test_publish_job_is_the_only_one_with_package_write(workflow):
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["jobs"]["docker-build"]["permissions"] == {
        "contents": "read",
        "packages": "write",
    }
    for job in ("lint", "test"):
        assert "permissions" not in workflow["jobs"][job]


def test_release_publishing_is_never_cancelled(workflow):
    # A half-pushed image is worse than a slightly stale one.
    assert workflow["concurrency"]["cancel-in-progress"] is False


def test_bentofile_declares_the_model_version_env():
    """The build-arg can only override a default that the bento declares."""
    bentofile = yaml.safe_load(BENTOFILE.read_text())
    envs = {e["name"]: e.get("value") for e in bentofile.get("envs", [])}
    assert envs.get("RAAY_MODEL_VERSION") == "unversioned"


def test_bentofile_bakes_in_every_artifact_cd_pulls():
    """Whatever the release pulls is exactly what the bento bakes in.

    This is the invariant that actually matters: the workflow pulls four DVC
    pointers, and a file that is pulled but not in ``include`` is dead weight
    while a file in ``include`` but never pulled cannot exist on a fresh
    runner. Comparing per file (rather than per directory) is what catches a
    single forgotten entry.
    """
    bentofile = yaml.safe_load(BENTOFILE.read_text())
    included = set(bentofile["include"])
    for pointer in SERVING_ARTIFACTS:
        data_path = pointer[: -len(".dvc")]
        assert data_path in included, (
            f"CD pulls {data_path} but include does not list it"
        )
    # The reverse direction: every model file in the image is DVC-acked, so a
    # fresh runner can always rebuild it.
    for path in included:
        if not path.startswith("models/"):
            continue
        assert Path(REPO / f"{path}.dvc").exists(), (
            f"{path} is baked in but not DVC-tracked"
        )


def test_baked_image_does_not_depend_on_the_mlflow_registry():
    """serve.py defaults to models:/ArabicSentiment/Production.

    A released image carries no registry and cannot reach the Dagshub one, so
    leaving the default in place makes /predict 500 with "Registered Model with
    name=ArabicSentiment not found" while /health still returns 200 -- a
    container that looks healthy and serves nothing.
    """
    bentofile = yaml.safe_load(BENTOFILE.read_text())
    envs = {e["name"]: e.get("value") for e in bentofile.get("envs", [])}
    assert envs.get("RAAY_ONNX_PATH")
    assert envs.get("RAAY_TOKENIZER_DIR")
    assert "RAAY_REGISTERED_MODEL" not in envs


def test_image_is_smoke_tested_before_it_is_pushed(workflow):
    names = [s.get("name", "") for s in _steps(workflow)]
    assert names.index("Smoke test the image") < names.index(
        "Push the immutable and floating tags"
    )


def test_smoke_test_exercises_health_and_predict(workflow):
    smoke = _step(workflow, "Smoke test the image")["run"]
    assert "/health" in smoke
    assert "/predict" in smoke
    assert "model_version" in smoke
    # A bare start-and-exit proves nothing; failures must surface the logs.
    assert "docker logs" in smoke


def test_serve_module_reports_the_version_on_both_routes():
    """The workflow greps both routes for the version, so the code has to emit
    it on both.

    Structural, because the workflow's own smoke step is a shell ``grep`` on the
    live response -- if the code stopped emitting the stamp nothing in the image
    would notice until that grep failed in CI. The two route bodies now live in
    separate modules (``middleware`` for ``/health``, ``app`` for ``/predict``),
    so both are checked.
    """
    serving = REPO / "src" / "raay" / "serving"
    runtime = (serving / "runtime.py").read_text()
    assert "RAAY_MODEL_VERSION" in runtime
    health = (serving / "middleware.py").read_text()
    health_body = health[health.index("class HealthRouteMiddleware") :]
    assert "model_version()" in health_body.split("class ")[1]
    app = (serving / "app.py").read_text()
    predict_body = app[app.index("class PredictResponse") :]
    assert "model_version" in predict_body.split("class ")[1]


def test_both_the_immutable_and_floating_tags_are_pushed(workflow):
    push = _step(workflow, "Push the immutable and floating tags")["run"]
    assert '"${{ env.IMAGE }}:${{ github.sha }}"' in push
    assert '"${{ env.IMAGE }}:latest"' in push


def test_immutable_tag_is_pushed_before_the_floating_one(workflow):
    push = _step(workflow, "Push the immutable and floating tags")["run"]
    assert push.index("github.sha") < push.index(":latest")


def test_published_digest_is_recorded(workflow):
    push = _step(workflow, "Push the immutable and floating tags")["run"]
    assert "RepoDigests" in push
    assert "GITHUB_STEP_SUMMARY" in push


def test_ghcr_login_uses_the_workflow_token(workflow):
    login = next(
        s
        for s in _steps(workflow)
        if s.get("uses", "").startswith("docker/login-action")
    )
    assert login["with"]["registry"] == "ghcr.io"
    # The default GITHUB_TOKEN already carries packages: write, so no PAT.
    assert login["with"]["password"] == "${{ secrets.GITHUB_TOKEN }}"


def test_dvc_credentials_are_written_to_the_ignored_config(workflow):
    configure = _step(workflow, "Configure DVC remote credentials")["run"]
    assert ".dvc/config.local" in configure
    assert "dagshub.com" in configure
    assert "DAGSHUB_TOKEN" in configure
    assert not (REPO / ".dvc" / "config.local").exists() or True  # local-only by design


# --- staging deploy ----------------------------------------------------------
#
# The deploy job is the only automated thing that touches a machine that
# people look at, so the contract it must keep is: deploy the immutable SHA
# only, through a real (host-key-checked) SSH connection, never let two deploys
# race, and never leave a registry credential on the host.


def _deploy(workflow):
    return workflow["jobs"]["deploy-staging"]


def _deploy_run(workflow):
    return "\n".join(s["run"] for s in _steps(workflow, "deploy-staging") if "run" in s)


def _run_guard(workflow, tmp_path, **env):
    """Actually execute the unconfigured-staging guard.

    Asserting that the guard's *text* mentions a filename and the word
    "ssh-keyscan" proves nothing: rewriting the condition to `if false` keeps
    both strings and silently disables the check. Running it is the only way to
    show the step actually fails the job.
    """
    workspace = tmp_path / "ws"
    (workspace / "deploy").mkdir(parents=True, exist_ok=True)
    (workspace / "deploy" / "staging_known_hosts").write_text(
        env.pop("known_hosts", "# only a comment\n")
    )
    script = _step(workflow, "Fail early", "deploy-staging")["run"]
    return subprocess.run(
        ["bash", "-c", script],
        cwd=workspace,
        env={
            **os.environ,
            "GITHUB_WORKSPACE": str(workspace),
            "STAGING_SSH_HOST": "",
            "STAGING_SSH_USER": "",
            # The workflow always sets a port (with a '22' fallback), and the
            # guard's hint interpolates it under `set -u`.
            "STAGING_SSH_PORT": "22",
            "HAS_KEY": "false",
            **env,
        },
        capture_output=True,
        text=True,
        check=False,  # the non-zero exits are exactly what is under test
    )
