"""Structural tests for ``.github/workflows/ci.yml``.

The workflow is the only place the CI contract lives, and it is not imported by
anything, so a typo in a step name or a deleted gate would otherwise only
surface when a real pull request runs. These assertions parse the YAML and pin
the parts that the checklist depends on.
"""

from pathlib import Path

import pytest
import yaml

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"


@pytest.fixture(scope="module")
def workflow():
    return yaml.safe_load(WORKFLOW.read_text())


def _steps(workflow, job):
    return workflow["jobs"][job]["steps"]


def _run_commands(workflow, job):
    commands = []
    for step in _steps(workflow, job):
        if "run" in step:
            commands.append(step["run"])
    return "\n".join(commands)


def test_workflow_is_valid_yaml_and_triggers_on_pull_request(workflow):
    # PyYAML resolves the bare `on:` key to the boolean True.
    triggers = workflow[True] if True in workflow else workflow["on"]
    assert "pull_request" in triggers
    assert triggers["pull_request"]["branches"] == ["dev"]


def test_expected_jobs_exist(workflow):
    assert set(workflow["jobs"]) == {"lint", "test", "pipeline"}


def test_pipeline_job_waits_for_lint_and_test(workflow):
    # Running the 3-minute DVC repro before knowing the lint is clean wastes
    # runner minutes on every broken PR.
    assert workflow["jobs"]["pipeline"]["needs"] == ["lint", "test"]


def test_lint_job_runs_the_same_checks_as_the_local_gate(workflow):
    commands = _run_commands(workflow, "lint")
    assert "ruff check" in commands
    assert "ruff format --check" in commands
    assert "mypy src" in commands


def test_test_job_reports_coverage_and_junit(workflow):
    commands = _run_commands(workflow, "test")
    assert "--cov=raay" in commands
    assert "--junitxml=reports/junit.xml" in commands


def test_pipeline_checks_out_full_history(workflow):
    # The metrics gate diffs against a base revision; a shallow clone has none.
    checkout = next(
        step
        for step in _steps(workflow, "pipeline")
        if str(step.get("uses", "")).startswith("actions/checkout")
    )
    assert checkout["with"]["fetch-depth"] == 0


def test_pipeline_pulls_only_the_raw_input(workflow):
    # `dvc pull <stage>` misses stage dependencies, and a bare `dvc pull`
    # would fetch the unrelated 212 MB reviews CSV.
    commands = _run_commands(workflow, "pipeline")
    assert "dvc pull data/raw/Final_Data.csv.dvc" in commands
    assert "dvc repro" in commands
    assert "dvc pull\n" not in commands


def test_pipeline_does_not_bake_credentials_into_the_tracked_config(workflow):
    commands = _run_commands(workflow, "pipeline")
    assert ".dvc/config.local" in commands
    assert "DAGSHUB_TOKEN}" in commands


def test_pipeline_propagates_the_dagshub_mlflow_env(workflow):
    env = workflow["env"]
    assert env["MLFLOW_TRACKING_URI"].startswith("https://dagshub.com/")
    assert env["MLFLOW_TRACKING_USERNAME"] == "${{ secrets.DAGSHUB_USER }}"
    assert env["MLFLOW_TRACKING_PASSWORD"] == "${{ secrets.DAGSHUB_TOKEN }}"


def test_metrics_gate_uses_the_pinned_threshold(workflow):
    commands = _run_commands(workflow, "pipeline")
    assert "scripts/ci_metrics_gate.py" in commands
    assert "--threshold 0.005" in commands
    assert "--markdown reports/metrics_diff.md" in commands


def test_gate_failure_is_reported_after_the_comment_is_posted(workflow):
    steps = _steps(workflow, "pipeline")
    names = [step.get("name", "") for step in steps]

    gate_index = next(i for i, n in enumerate(names) if "drift gate" in n)
    comment_index = next(i for i, n in enumerate(names) if "Post the metrics" in n)
    fail_index = next(i for i, n in enumerate(names) if "Fail on metrics drift" in n)

    assert gate_index < comment_index < fail_index
    # continue-on-error is what lets the report reach the PR before the job fails.
    gate = steps[gate_index]
    assert gate["continue-on-error"] is True
    assert steps[fail_index]["if"] == "steps.gate.outcome == 'failure'"


def test_pr_comment_is_skipped_on_forks(workflow):
    steps = _steps(workflow, "pipeline")
    comment = next(
        s for s in steps if s.get("name") == "Post the metrics report on the PR"
    )
    assert "head.repo.full_name == github.repository" in comment["if"]


def test_lock_reproducibility_check_runs_last(workflow):
    # dvc.lock embeds the metrics file hashes, so the gate must explain a
    # metrics change before this raw diff reports it.
    steps = _steps(workflow, "pipeline")
    names = [step.get("name", "") for step in steps]
    lock_index = next(i for i, n in enumerate(names) if "dvc.lock" in n)
    gate_index = next(i for i, n in enumerate(names) if "drift gate" in n)
    assert lock_index > gate_index
    commands = "\n".join(s.get("run", "") for s in steps)
    assert "git diff --exit-code dvc.lock" in commands


def test_dvc_status_is_asserted_clean(workflow):
    commands = _run_commands(workflow, "pipeline")
    assert "dvc status" in commands
    assert "up to date" in commands


def test_workflow_only_writes_permissions_where_needed(workflow):
    # Comments need PR write access; lint and test must stay read-only.
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["jobs"]["pipeline"]["permissions"] == {
        "contents": "read",
        "pull-requests": "write",
    }
    assert "permissions" not in workflow["jobs"]["lint"]
    assert "permissions" not in workflow["jobs"]["test"]


def test_install_is_frozen_so_ci_cannot_rewrite_the_lock(workflow):
    assert "--frozen" in workflow["env"]["UV_FLAGS"]
    for job in workflow["jobs"]:
        commands = _run_commands(workflow, job)
        if "uv sync" in commands:
            assert "uv sync ${{ env.UV_FLAGS }}" in commands
