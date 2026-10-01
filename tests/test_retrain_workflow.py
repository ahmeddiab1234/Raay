"""Structural tests for ``.github/workflows/retrain.yml``.

Step 6 -- scheduled/automated retraining hooks. Like the CI contract, the
workflow is the only place its behaviour lives, so a typoed step name or a
deleted gate would only surface on a real run. These assertions pin:

* the trigger set (weekly cron + dispatch + repository_dispatch receiver),
* the branch contract (file on main for scheduling, work against `dev`),
* the reuse of the SAME gate CI uses, with the sizes/counts escape hatch,
* the no-op early exit (unchanged data costs nothing),
* the pass -> data-refresh-PR and fail -> alarm paths,
* Phase 6 step 3's `notify` job: dispatch-only, issue-only, and explicitly
  incapable of retraining (the refresh job's early exit is intentionally left
  intact -- a psi_breach means the raw data did not change), and
* the omissions that matter (no secrets in argv, no fork path).
"""

from pathlib import Path

import pytest
import yaml

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "retrain.yml"


@pytest.fixture(scope="module")
def workflow():
    return yaml.safe_load(WORKFLOW.read_text())


def _steps(workflow):
    return workflow["jobs"]["refresh"]["steps"]


def _commands(workflow):
    return "\n".join(s.get("run", "") for s in _steps(workflow))


def _notify_steps(workflow):
    return workflow["jobs"]["notify"]["steps"]


def _notify_commands(workflow):
    return "\n".join(s.get("run", "") for s in _notify_steps(workflow))


def test_workflow_is_valid_yaml_and_has_the_expected_job(workflow):
    triggers = workflow[True] if True in workflow else workflow["on"]
    assert "schedule" in triggers
    assert "workflow_dispatch" in triggers
    assert "repository_dispatch" in triggers
    assert set(workflow["jobs"]) == {"refresh", "notify"}


def test_weekly_schedule_is_kept_next_to_the_promote_cadence(workflow):
    triggers = workflow[True] if True in workflow else workflow["on"]
    assert triggers["schedule"] == [{"cron": "12 4 * * 1"}]


def test_repository_dispatch_receives_the_retrain_event(workflow):
    triggers = workflow[True] if True in workflow else workflow["on"]
    assert triggers["repository_dispatch"]["types"] == ["retrain"]


def test_workflow_operates_on_dev_though_it_is_scheduled_from_main(workflow):
    # GitHub schedules cron only from the default branch (main). The workflow
    # file therefore lives on main -- where a merge is a fast-forward of dev --
    # but must never work there: data changes land on dev.
    checkout = next(
        s
        for s in _steps(workflow)
        if str(s.get("uses", "")).startswith("actions/checkout")
    )
    assert checkout["with"]["ref"] == "dev"
    assert checkout["with"]["fetch-depth"] == 0


def test_write_permissions_are_declared_and_scoped(workflow):
    # The writes are the data-refresh branch push (contents), the PR against dev
    # (pull-requests), and Phase 6 step 3's retrain issue (issues). Declared
    # explicitly and reviewed here so a fourth, unreviewed scope cannot appear.
    assert workflow["permissions"] == {
        "contents": "write",
        "pull-requests": "write",
        "issues": "write",
    }


def test_concurrency_never_interleaves_two_refreshes(workflow):
    assert workflow["concurrency"]["group"] == "retrain-data"
    assert workflow["concurrency"]["cancel-in-progress"] is False


def test_dagshub_credentials_go_to_config_local_never_argv(workflow):
    commands = _commands(workflow)
    assert ".dvc/config.local" in commands
    assert "DAGSHUB_TOKEN}" in commands
    assert "password = ${DAGSHUB_TOKEN}" in commands


def test_repro_env_matches_ci(workflow):
    env = workflow["env"]
    assert "--frozen" in env["UV_FLAGS"]
    assert env["MLFLOW_TRACKING_URI"].startswith("https://dagshub.com/")
    assert env["MLFLOW_TRACKING_USERNAME"] == "${{ secrets.DAGSHUB_USER }}"
    assert env["MLFLOW_TRACKING_PASSWORD"] == "${{ secrets.DAGSHUB_TOKEN }}"


def test_pipeline_steps_mirror_ci(workflow):
    commands = _commands(workflow)
    assert "dvc pull data/raw/Final_Data.csv.dvc" in commands
    assert "dvc repro" in commands
    assert "dvc status" in commands
    assert "up to date" in commands


def test_unchanged_data_exits_before_the_gate(workflow):
    # dvc.lock embeds the raw input md5; a clean lock after repro means nothing
    # changed, and the weekly no-op should stop here -- not gate an empty diff.
    steps = _steps(workflow)
    names = [s.get("name", "") for s in steps]
    early = next(i for i, n in enumerate(names) if "Early-exit" in n)
    gate = next(i for i, n in enumerate(names) if "Refresh gate" in n)
    assert early < gate
    assert "git diff --exit-code --quiet dvc.lock" in steps[early]["run"]
    assert "exit 0" in steps[early]["run"]


def test_gate_is_the_same_ci_gate_with_the_sizes_escape_hatch(workflow):
    steps = _steps(workflow)
    gate = next(
        s for s in steps if s.get("name", "") == "Refresh gate (proportions only)"
    )
    run = gate["run"]
    assert "scripts/ci_metrics_gate.py" in run
    assert "--base HEAD" in run
    assert "--targets reports/split_metrics.json" in run
    assert "--ignore '.*_size$'" in run
    assert '--threshold "$REFRESH_THRESHOLD"' in run
    assert '--markdown "reports/data_refresh_$date.md"' in run
    assert gate["continue-on-error"] is True
    assert gate["id"] == "gate"


def test_refresh_threshold_defaults_and_accepts_dispatch_override(workflow):
    assert (
        workflow["env"]["REFRESH_THRESHOLD"]
        == "${{ github.event.client_payload.threshold || '0.005' }}"
    )


def test_pass_path_opens_a_data_refresh_pr_to_dev(workflow):
    steps = _steps(workflow)
    pr = next(s for s in steps if s.get("name") == "Open the data-refresh PR on dev")
    assert pr["if"] == "steps.gate.outcome == 'success'"
    assert pr["env"] == {"GH_TOKEN": "${{ secrets.GITHUB_TOKEN }}"}
    run = pr["run"]
    assert "retrain/data-$date" in run
    assert (
        "git add dvc.lock reports/preprocess_metrics.json reports/split_metrics.json"
        in run
    )
    assert 'gh pr create --base dev --head "$branch"' in run
    assert '--body-file "reports/data_refresh_$date.md"' in run


def test_existing_refresh_branch_is_an_idempotent_noop(workflow):
    steps = _steps(workflow)
    pr = next(s for s in steps if s.get("name") == "Open the data-refresh PR on dev")
    assert "git show-ref --verify --quiet" in pr["run"]
    assert "already exists; nothing to do" in pr["run"]
    assert pr["run"].count("exit 0") == 1


def test_fail_path_fails_the_workflow_loudly(workflow):
    steps = _steps(workflow)
    fail = next(s for s in steps if s.get("name") == "Fail on refresh drift")
    assert fail["if"] == "steps.gate.outcome == 'failure'"
    assert "::error::" in fail["run"]
    assert "exit 1" in fail["run"]
    # A failed refresh must land between the gate and the summary.
    names = [s.get("name", "") for s in steps]
    assert names.index("Refresh gate (proportions only)") < names.index(
        "Fail on refresh drift"
    )
    assert names.index("Fail on refresh drift") < names.index("Summarise the refresh")


def test_report_is_uploaded_and_summarised_on_every_path(workflow):
    steps = _steps(workflow)
    upload = next(s for s in steps if s.get("name") == "Upload the refresh report")
    assert upload["if"] == "always()"
    summary = next(s for s in steps if s.get("name") == "Summarise the refresh")
    assert summary["if"] == "always()"
    assert "GITHUB_STEP_SUMMARY" in summary["run"]


def test_dispatch_reason_is_echoed_but_never_trusted(workflow):
    commands = _commands(workflow)
    assert "client_payload.reason || 'schedule'" in commands


def test_notify_job_is_dispatch_only_and_follows_the_refresh(workflow):
    # A schedule run carries no drift evidence, so the notify job must not run
    # for it; and it needs the refresh job so the two never race on the same run.
    notify = workflow["jobs"]["notify"]
    assert notify["if"] == "github.event_name == 'repository_dispatch'"
    assert notify["needs"] == "refresh"


def test_notify_job_cannot_retrain(workflow):
    # The whole point of a separate job: it decides and records, it does not
    # fine-tune. Naming kaggle_train_runs.py in prose is fine; *invoking*
    # anything that could train or ship is not, so this pins the execution
    # vectors rather than mere mentions.
    commands = _notify_commands(workflow)
    for forbidden in ("uv run", "dvc ", "python -m ", "bentoml ", "docker "):
        assert forbidden not in commands


def test_notify_reads_every_payload_field_it_reports(workflow):
    # Dump the whole job, not just the run strings: the payload fields are bound
    # in each step's `env:`, and asserting on `run` alone would miss them.
    job = yaml.safe_dump(workflow["jobs"]["notify"])
    for field in ("reason", "trigger_date", "psi", "psi_column", "triage"):
        assert f"client_payload.{field}" in job
    # The evidence must reach the issue body, not just the shell environment.
    commands = _notify_commands(workflow)
    assert "worst column" in commands
    assert "| PSI |" in commands


def test_notify_issue_is_idempotent(workflow):
    # A retried dispatch must not stack duplicate issues.
    commands = _notify_commands(workflow)
    assert "gh issue list" in commands
    assert "already exists; nothing to do" in commands
    assert commands.count("gh issue create") == 1


def test_refresh_early_exit_is_deliberately_left_intact(workflow):
    # Phase 6 step 3 must NOT bypass this. A psi_breach means the raw data is
    # unchanged, so the refresh job is correctly a no-op; forcing past the exit
    # would reach `git commit` with nothing staged and fail. The notify job is
    # the receiver instead, so the early exit stays exactly where it was.
    steps = _steps(workflow)
    early = next(s for s in steps if "Early-exit" in s.get("name", ""))
    assert "git diff --exit-code --quiet dvc.lock" in early["run"]
    assert "exit 0" in early["run"]


def test_no_fork_handling_is_needed(workflow):
    # Cron/dispatch runs are always on this repository, never a fork's PR, so
    # the fork gates CI needs are legitimately absent -- but so is any pull
    # request trigger, which is what would introduce them.
    triggers = workflow[True] if True in workflow else workflow["on"]
    assert "pull_request" not in triggers
    assert "head.repo.full_name" not in _commands(workflow)


# The .gitignore entries are a documented contract this workflow depends on,
# mirroring how the canary/state files are pinned elsewhere.
def test_refresh_reports_are_git_ignored():
    gitignore = Path(__file__).resolve().parents[1] / ".gitignore"
    text = gitignore.read_text()
    assert "reports/data_refresh_*.json" in text
    assert "reports/data_refresh_*.md" in text
