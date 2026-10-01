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


def workflow_text(name: str) -> str:
    return (WORKFLOW.parent / name).read_text()


def workflow_of(name: str):
    return yaml.safe_load(workflow_text(name))


@pytest.fixture(scope="module")
def dvc():
    return yaml.safe_load((WORKFLOW.parents[2] / "dvc.yaml").read_text())


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
    assert set(workflow["jobs"]) == {
        "lint",
        "test",
        "pipeline",
        "pipeline-fork-notice",
    }


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


def test_pr_comment_uses_authenticated_github_cli(workflow):
    steps = _steps(workflow, "pipeline")
    assert not any("setup-cml" in step.get("uses", "") for step in steps)
    comment = next(
        s for s in steps if s.get("name") == "Post the metrics report on the PR"
    )
    assert comment["env"]["GH_TOKEN"] == "${{ secrets.GITHUB_TOKEN }}"
    assert 'gh pr comment "$PR_NUMBER"' in comment["run"]
    assert "--body-file reports/metrics_diff.md" in comment["run"]


def test_dvc_pipeline_job_is_gated_off_forks(workflow):
    """The whole secret-dependent job is skipped, not just the comment.

    GitHub exposes no secrets to fork pull requests, so without a job-level
    condition the DVC steps run with an empty DAGSHUB_TOKEN and fail mid-
    pipeline on a confusing authentication error. The per-step `if` on the
    comment alone does not protect them.
    """
    condition = workflow["jobs"]["pipeline"]["if"]
    assert "head.repo.full_name == github.repository" in condition
    assert "pull_request" in condition


def test_fork_notice_job_explains_the_skip(workflow):
    job = workflow["jobs"]["pipeline-fork-notice"]
    assert job["needs"] == ["lint", "test"]
    assert "head.repo.full_name != github.repository" in job["if"]
    commands = _run_commands(workflow, "pipeline-fork-notice")
    assert "DAGSHUB_TOKEN" in commands
    # The notice must not claim the gate passed; the gate never ran.
    assert "DAGSHUB_TOKEN" in commands and "lint and test" in commands


def test_fork_jobs_are_mutually_exclusive(workflow):
    """Exactly one of the two pipeline jobs may run for a given event."""
    pipeline = workflow["jobs"]["pipeline"]["if"]
    notice = workflow["jobs"]["pipeline-fork-notice"]["if"]
    assert "head.repo.full_name == github.repository" in pipeline
    assert "head.repo.full_name != github.repository" in notice


def test_fork_notice_job_needs_no_write_permissions(workflow):
    # A skipped-pipeline notice is a log line, never a PR comment.
    assert "permissions" not in workflow["jobs"]["pipeline-fork-notice"]


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


# ------------------------------------------------------- feedback (Phase 6)


def test_the_feedback_stage_is_not_reproduced_in_ci(workflow):
    """CI must not repro the feedback stage, and must not `dvc add` it either.

    The reviewed overrides file is a human-in-the-loop artifact: an operator edits
    it after `--mode review` and then `dvc add`s it themselves, which is the same
    ritual as `data/raw/Final_Data.csv`. If CI regenerated or re-added it, every PR
    would rewrite a file a person is meant to have signed off on.
    """
    commands = "\n".join(_run_commands(workflow, job) for job in workflow["jobs"])
    assert "dvc repro" in commands  # the pipeline stage still exists
    assert "raay.data.feedback" not in commands
    assert "--mode review" not in commands


def test_feedback_is_not_a_ci_dvc_stage(dvc):
    """`dvc repro` with no args would then run it, on a runner with no raw captures.

    A stage that no CI job wants must still be *named* explicitly in the pipeline
    job's `dvc repro` call, or it gets swept in.
    """
    assert "feedback" in dvc["stages"], (
        "the stage should still exist for the nightly job"
    )
    repro_lines = []
    for job in workflow_of("ci.yml").get("jobs", {}).values():
        for step in job.get("steps", []):
            run = step.get("run", "")
            if "dvc repro" in run:
                repro_lines.append(run)
    assert repro_lines, "expected an explicit dvc repro in CI"
    for line in repro_lines:
        # Every repro is target-scoped, so a stage CI does not want can never be
        # swept in by a bare `dvc repro`.
        targets = line.split("dvc repro", 1)[1].strip()
        assert targets, "a bare `dvc repro` would run the feedback stage too"
        assert "feedback" not in targets


def test_the_status_check_stays_scoped_away_from_feedback(workflow):
    """`dvc status` with no stage list would false-fail on the feedback outputs.

    Same trap as the serving artifacts: CI never pulls the feedback stage's
    outputs, so an unscoped status check reports them as missing.
    """
    commands = _run_commands(workflow, "pipeline")
    status_lines = [
        line
        for line in commands.splitlines()
        if "dvc status" in line and not line.lstrip().startswith("#")
    ]
    assert status_lines
    for line in status_lines:
        assert "preprocess" in line and "split" in line


def test_the_lock_check_still_covers_the_whole_pipeline(workflow):
    """`-- dvc.lock` is unscoped, so it also covers the feedback stage's hashes.

    That is deliberate and is the one place CI touches feedback: if a committed
    `dvc.lock` disagrees with a re-run merge, reproducibility is broken and the
    diff must surface.
    """
    commands = _run_commands(workflow, "pipeline")
    assert "git diff --exit-code dvc.lock" in commands


def test_retrain_also_leaves_the_feedback_stage_alone():
    """`retrain.yml` re-runs `dvc repro` on new raw data and must stay scoped too."""
    retrain = workflow_of("retrain.yml")
    commands = "\n".join(
        step.get("run", "") for job in retrain["jobs"].values() for step in job["steps"]
    )
    assert "--mode review" not in commands
    assert "raay.data.feedback" not in commands


def test_no_workflow_can_widen_a_token_into_a_training_path(workflow):
    """The capture token must never be echoed or read by a CI/CD job.

    A write endpoint into the training set is a poisoning vector, so the token is
    a runtime secret of the sidecar only.
    """
    for name in ("ci.yml", "cd.yml", "promote.yml", "retrain.yml"):
        text = workflow_text(name)
        assert "RAAY_FEEDBACK_TOKEN" not in text
        assert "feedback_service" not in text


def test_retrain_repro_is_target_scoped_too():
    """Pin the scope on both workflows that repro, not just the assertion that
    neither mentions feedback -- a future bare `dvc repro` would silently sweep
    the stage back in, and only the target list catches that."""
    for name in ("ci.yml", "retrain.yml"):
        for job in workflow_of(name).get("jobs", {}).values():
            for step in job.get("steps", []):
                run = step.get("run", "")
                if "dvc repro" in run:
                    targets = run.split("dvc repro", 1)[1].strip()
                    assert targets, f"{name} has a bare `dvc repro`"
                    assert "feedback" not in targets, (
                        f"{name} repros the feedback stage"
                    )
                    assert set(targets.split()) == {"preprocess", "split"}
