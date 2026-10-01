"""Structural checks for the artifacts and version baked into the release image."""

from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
WORKFLOW = REPO / ".github" / "workflows" / "cd.yml"
BENTOFILE = REPO / "bentofile.yaml"
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
