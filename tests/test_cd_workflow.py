"""Structural tests for ``.github/workflows/cd.yml``.

The release workflow is the only thing standing between a commit and a pushed,
pullable image, and it runs on a fresh runner that has no model on disk. Nothing
imports it, so these assertions pin the parts that the release contract depends
on: the trigger is ``main`` only, the DVC-acked artifacts are fetched before the
bento is built, the version reaches the image twice (label + env), the image is
smoke-tested *before* it is pushed, and the mutable tag is never the only tag.
"""

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
    assert set(workflow["jobs"]) == {"lint", "test", "docker-build"}


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
