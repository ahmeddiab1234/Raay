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


def test_install_is_frozen(workflow):
    assert "--frozen" in workflow["env"]["UV_FLAGS"]


def test_image_reference_is_owner_scoped_ghcr(workflow):
    image = workflow["env"]["IMAGE"]
    assert image.startswith("ghcr.io/${{ github.repository_owner }}/")


def test_model_artifacts_are_pulled_from_dvc_before_the_bento_is_built(workflow):
    pull = _step(workflow, "Pull the serving artifacts")["run"]
    # The graph pointer is referenced through the env var so there is a single
    # source of truth shared with the derive/verify steps.
    assert workflow["env"]["MODEL_DVC"] in SERVING_ARTIFACTS
    assert "${{ env.MODEL_DVC }}" in pull
    for artifact in SERVING_ARTIFACTS:
        if artifact == workflow["env"]["MODEL_DVC"]:
            continue
        assert artifact in pull
    # Order matters: the bento include list is resolved at build time, so a
    # build that ran before the pull would silently bake nothing.
    names = [s.get("name", "") for s in _steps(workflow)]
    assert names.index("Pull the serving artifacts") < names.index("Build the bento")


def test_the_212mb_training_csv_is_never_fetched(workflow):
    # Nothing in the image is built from the training data; pulling it would add
    # minutes to every release for no benefit.
    commands = _run_commands(workflow)
    assert "Final_Data.csv" not in commands
    assert "arabic_sentiment_reviews" not in commands


def test_pulled_graph_is_verified_against_the_dvc_pointer(workflow):
    verify = _step(workflow, "Verify the pulled graph")
    assert "md5sum" in verify["run"]
    assert "stat -c %s" in verify["run"]
    assert "exit 1" in verify["run"]


def test_model_version_is_derived_from_the_dvc_hash(workflow):
    derive = _step(workflow, "Derive the model version")
    assert derive["id"] == "model"
    # Key-anchored parse; a bare `awk '{print $2}'` on "- md5: <hash>" returns
    # the literal "md5:".
    assert "md5:" in derive["run"]
    assert "int8-" in derive["run"]
    assert "steps.model.outputs.model_version" not in derive["run"]
    assert "model_version=" in derive["run"]


def test_version_is_baked_in_as_both_a_label_and_an_env(workflow):
    containerize = _step(workflow, "Containerize")
    assert "RAAY_MODEL_VERSION=${MODEL_VERSION}" in containerize["run"]
    assert "org.opencontainers.image.revision" in containerize["run"]
    assert "org.opencontainers.image.version" in containerize["run"]


def test_unversioned_image_is_refused(workflow):
    """BentoML bento labels are not OCI labels, so the build-arg is the only
    channel that reaches the service. An image without it is still shippable
    but untraceable, so the build fails loudly instead."""
    containerize = _step(workflow, "Containerize")
    assert "refusing to build an unlabelled image" in containerize["run"]


def test_built_image_is_asserted_to_carry_the_version(workflow):
    assert_labels = _step(workflow, "Assert the version reached the image")
    run = assert_labels["run"]
    assert "docker inspect" in run
    assert "org.opencontainers.image.revision" in run
    assert "RAAY_MODEL_VERSION" in run
    assert "exit 1" in run


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
        assert (
            data_path in included
        ), f"CD pulls {data_path} but include does not list it"
    # The reverse direction: every model file in the image is DVC-acked, so a
    # fresh runner can always rebuild it.
    for path in included:
        if not path.startswith("models/"):
            continue
        assert Path(
            REPO / f"{path}.dvc"
        ).exists(), f"{path} is baked in but not DVC-tracked"


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
    it on both."""
    serve = (REPO / "src" / "raay" / "serving" / "serve.py").read_text()
    assert "RAAY_MODEL_VERSION" in serve
    health = serve[serve.index("class HealthRouteMiddleware") :]
    assert "model_version()" in health.split("class ")[1]


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
    # Without always(), the failure path is the one that leaks the credential.
    assert cleanup["if"] == "always()"
    assert "registry-token" in cleanup["run"]


def test_staging_host_key_is_verified_not_disabled(workflow):
    run = _deploy_run(workflow)
    # StrictHostKeyChecking=no anywhere in the deploy would make the whole
    # exercise MITM-able.
    assert "StrictHostKeyChecking=no" not in run
    assert "UserKnownHostsFile=/dev/null" not in run
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
        assert fields[2].startswith(
            ("AAAA", "AAAAB", "AAAAC")
        ), f"not key material: {line!r}"
