"""Staging deploy tests for ``.github/workflows/cd.yml``.

The deploy job is the only automated thing that touches a machine that
people look at, so the contract it must keep is: deploy the immutable SHA
only, through a real (host-key-checked) SSH connection, never let two deploys
race, and never leave a registry credential on the host.
"""

import os
import subprocess

import yaml
from cd_helpers import REPO, _step, _steps


def _deploy(workflow):
    return workflow["jobs"]["deploy-staging"]


def _deploy_run(workflow):
    return "\n".join(s["run"] for s in _steps(workflow, "deploy-staging") if "run" in s)


def test_staging_deploy_runs_after_the_image_is_published(workflow):
    job = _deploy(workflow)
    # needs the *build* job, not just lint/test: deploying before the image is
    # in the registry would fail on the pull.
    assert job["needs"] == ["docker-build"]


def test_staging_deploy_uses_an_environment(workflow):
    # An environment is what supplies the gate, the per-env secrets and the
    # deployment history; with reviewers configured every release waits.
    assert _deploy(workflow)["environment"] == "staging"


def test_staging_deploys_are_never_cancelled(workflow):
    # Interrupting between "pull the new image" and "roll back" is how the box
    # ends up serving neither version.
    assert _deploy(workflow)["concurrency"]["group"] == "deploy-staging"
    assert _deploy(workflow)["concurrency"]["cancel-in-progress"] is False


def test_staging_deploy_needs_no_package_write(workflow):
    # It only *reads* from GHCR, with a read:packages PAT over SSH. Giving this
    # job packages: write would hand the runner a credential it does not need.
    assert "permissions" not in _deploy(workflow)


def test_staging_deploys_the_commit_sha_not_a_floating_tag(workflow):
    run = _step(workflow, "Deploy", "deploy-staging")["run"]
    # Assert the argument itself, not just that "github.sha" appears somewhere:
    # swapping the value for the bare word `latest` contains no colon and would
    # slip past a search for ":latest".
    assert "--target-sha '${{ github.sha }}'" in run
    deploy_cmd = run[run.index("python3 /opt/raay-staging/deploy_staging.py") :]
    assert "latest" not in deploy_cmd


def test_staging_deploy_verifies_the_model_version(workflow):
    deploy = _step(workflow, "Deploy", "deploy-staging")
    # A bare health check would happily pass on a stale image left on the box.
    assert "--expect-model-version" in deploy["run"]
    assert (
        deploy["env"]["MODEL_VERSION"]
        == "${{ needs.docker-build.outputs.model_version }}"
    )


def test_publish_job_exposes_the_model_version_to_the_deploy(workflow):
    outputs = workflow["jobs"]["docker-build"]["outputs"]
    assert outputs["model_version"] == "${{ steps.build.outputs.MODEL_VERSION }}"


def test_staging_deploy_uses_the_rolling_back_tool(workflow):
    deploy = _step(workflow, "Deploy", "deploy-staging")["run"]
    assert "deploy_staging.py" in deploy
    # ...and it must exit non-zero unless the smoke test passed, or a broken
    # release would be reported as a green deploy.
    assert "if [ $rc -ne 0 ]" in deploy


def test_staging_tools_are_copied_to_a_stable_location(workflow):
    copy = _step(workflow, "Copy the deploy tool", "deploy-staging")["run"]
    assert "docker-compose.staging.yml" in copy
    assert "scripts/deploy_staging.py" in copy
    assert "/opt/raay-staging/" in copy
    check = _step(workflow, "Check the staging host", "deploy-staging")["run"]
    # The state file lives outside the copied tools so a redeploy of the tools
    # cannot wipe the rollback history.
    assert "/var/lib/raay-staging" in check


def test_staging_deploy_does_not_ship_the_compose_build_fallback(workflow):
    # docker-compose.staging.yml must not contain a build: block, or a staging
    # deploy could quietly build the repo instead of pulling the published image.
    # Parsed rather than grepped: the file's comments discuss `build:` and
    # `env_file` by name, so a text search would trip over its own explanation.
    compose = yaml.safe_load((REPO / "docker-compose.staging.yml").read_text())
    service = compose["services"]["raay-sentiment"]
    assert "build" not in service
    assert "env_file" not in service
    assert "image" in service


def test_staging_compose_requires_an_explicit_image(workflow):
    text = (REPO / "docker-compose.staging.yml").read_text()
    # The :? guard is what turns "RAAY_IMAGE unset" into an immediate error
    # instead of an empty image name.
    assert "${RAAY_IMAGE:?" in text
    assert "8080:3000" in text
    compose = yaml.safe_load((REPO / "docker-compose.staging.yml").read_text())
    # A pinned project name keeps the container/network distinct from prod, so
    # both can coexist on one host.
    assert compose["name"] == "staging"


def test_registry_token_never_reaches_argv_or_a_log(workflow):
    handover = _step(workflow, "Hand over the registry token", "deploy-staging")
    # Written over stdin into a 0600 file, then read with --registry-token-file.
    assert "cat >" in handover["run"]
    assert "umask 077" in handover["run"]
    deploy = _step(workflow, "Deploy", "deploy-staging")["run"]
    assert "--registry-token-file" in deploy
    assert "secrets.STAGING_GHCR_TOKEN" not in deploy


def test_registry_token_is_removed_even_when_the_deploy_fails(workflow):
    cleanup = _step(workflow, "Remove the registry token", "deploy-staging")
    # Keep cleanup on deploy failures, but don't SSH before the token was sent.
    handover = _step(workflow, "Hand over the registry token", "deploy-staging")
    assert handover["id"] == "registry_token_handover"
    assert (
        cleanup["if"]
        == "always() && steps.registry_token_handover.outcome == 'success'"
    )
    assert "registry-token" in cleanup["run"]


def test_staging_host_key_is_verified_not_disabled(workflow):
    run = _deploy_run(workflow)
    # StrictHostKeyChecking=no anywhere in the deploy would make the whole
    # exercise MITM-able.
    assert "StrictHostKeyChecking=no" not in run
    assert "UserKnownHostsFile=/dev/null" not in run
    assert run.count("${STAGING_SSH_PORT:-22}") == 6
    trust = _step(workflow, "Trust the staging host key", "deploy-staging")["run"]
    assert "deploy/staging_known_hosts" in trust
    assert "~/.ssh/known_hosts" in trust


def test_unconfigured_staging_fails_with_an_actionable_message(workflow):
    guard = _step(workflow, "Fail early", "deploy-staging")["run"]
    for name in ("STAGING_SSH_HOST", "STAGING_SSH_USER", "STAGING_SSH_PRIVATE_KEY"):
        assert name in guard
    # And it must refuse to run with an unpopulated known_hosts file rather than
    # disabling the check to get past it.
    assert "deploy/staging_known_hosts" in guard
    assert "ssh-keyscan" in guard


def test_staging_ssh_agent_is_pinned(workflow):
    agent = next(
        s
        for s in _steps(workflow, "deploy-staging")
        if s.get("uses", "").startswith("webfactory/ssh-agent")
    )
    assert agent["uses"] == "webfactory/ssh-agent@v0.9.0"
    assert agent["with"]["ssh-private-key"] == "${{ secrets.STAGING_SSH_PRIVATE_KEY }}"


def test_staging_deploy_reports_its_outcome_to_the_run_summary(workflow):
    deploy = _step(workflow, "Deploy", "deploy-staging")["run"]
    assert "GITHUB_STEP_SUMMARY" in deploy
    # The container logs are the first thing anyone wants after a failed deploy.
    assert "container logs" in deploy


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


def test_unconfigured_staging_actually_fails_the_job(workflow, tmp_path):
    proc = _run_guard(workflow, tmp_path)
    assert proc.returncode != 0
    for name in ("STAGING_SSH_HOST", "STAGING_SSH_USER", "STAGING_SSH_PRIVATE_KEY"):
        assert name in proc.stdout


def test_a_missing_private_key_fails_the_job(workflow, tmp_path):
    proc = _run_guard(
        workflow, tmp_path, STAGING_SSH_HOST="box", STAGING_SSH_USER="deploy"
    )
    assert proc.returncode != 0
    assert "STAGING_SSH_PRIVATE_KEY" in proc.stdout


def test_an_empty_known_hosts_file_fails_the_job(workflow, tmp_path):
    proc = _run_guard(
        workflow,
        tmp_path,
        STAGING_SSH_HOST="box",
        STAGING_SSH_USER="deploy",
        HAS_KEY="true",
    )
    assert proc.returncode != 0
    assert "ssh-keyscan" in proc.stdout


def test_a_fully_configured_staging_passes_the_guard(workflow, tmp_path):
    proc = _run_guard(
        workflow,
        tmp_path,
        known_hosts="staging.example.com ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIExample\n",
        STAGING_SSH_HOST="box",
        STAGING_SSH_USER="deploy",
        HAS_KEY="true",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_a_comment_only_known_hosts_file_still_fails_the_job(workflow, tmp_path):
    """The committed file documents the format, so its prose mentions a key type.

    Matching the raw text would let that documentation satisfy the guard and
    wave an entirely unpopulated file through.
    """
    proc = _run_guard(
        workflow,
        tmp_path,
        known_hosts="# a valid entry looks like <host> ssh-ed25519 AAAA...\n",
        STAGING_SSH_HOST="box",
        STAGING_SSH_USER="deploy",
        HAS_KEY="true",
    )
    assert proc.returncode != 0


def test_the_committed_known_hosts_file_is_either_empty_or_well_formed():
    """Works now (placeholder) and after the host key is added.

    It deliberately does not assert the file is empty forever: populating it is
    the intended next step once the VM exists, and a test that forbids that
    would be a test that fails on success. It asserts that any real entry has
    the shape the deploy job depends on.
    """
    text = (REPO / "deploy" / "staging_known_hosts").read_text()
    body = [
        line
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    for line in body:
        fields = line.split()
        assert len(fields) >= 3, f"not a known_hosts entry: {line!r}"
        assert fields[1] in {
            "ssh-rsa",
            "ssh-ed25519",
        } or fields[1].startswith("ecdsa-sha2-"), f"unexpected key type: {line!r}"
        assert fields[2].startswith(("AAAA", "AAAAB", "AAAAC")), (
            f"not key material: {line!r}"
        )
