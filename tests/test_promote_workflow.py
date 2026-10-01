"""Structural tests for ``.github/workflows/promote.yml``.

The promotion workflow is the only thing that can move the ``Production`` alias,
and it runs unattended on a schedule as well as on demand. Nothing imports it,
so these tests pin the properties the safety argument rests on: the evaluating
job physically cannot promote, the promoting job cannot start without a human,
the two are separated by exactly one ``needs`` edge, and no trigger can be
reached from a fork.
"""

from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
WORKFLOW = REPO / ".github" / "workflows" / "promote.yml"

# What the gate job must pull to be able to score anything. The test split has
# no DVC pointer of its own -- it is an output of the `split` stage -- so a
# workflow that pulls a bare file, or only `preprocess`, silently ends up with
# no data and a gate that cannot run.
REQUIRED_PULLS = (
    "dvc pull split",
    "dvc pull models/baseline/final/config.json.dvc",
    "dvc pull models/baseline/final/tokenizer.json.dvc",
    "dvc pull models/baseline/final/tokenizer_config.json.dvc",
)


@pytest.fixture(scope="module")
def workflow():
    return yaml.safe_load(WORKFLOW.read_text())


def _triggers(workflow):
    # PyYAML resolves the bare `on:` key to the boolean True.
    return workflow[True] if True in workflow else workflow["on"]


def _steps(workflow, job):
    return workflow["jobs"][job]["steps"]


def _runs(workflow, job):
    return "\n".join(s["run"] for s in _steps(workflow, job) if "run" in s)


def _commands(workflow, job):
    """The run scripts with comments and blank lines removed.

    Several steps explain *why* a flag is absent, so a plain substring search
    over the raw YAML matches the prose rather than the behaviour.
    """
    out = []
    for script in _runs(workflow, job).splitlines():
        line = script.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return "\n".join(out)


def test_workflow_is_valid_yaml(workflow):
    assert set(workflow["jobs"]) == {"gate", "promote"}


def test_no_fork_reachable_trigger(workflow):
    """A PR-triggered promotion would run with a stranger's DVC token.

    There is no pull_request and no push trigger at all: the workflow runs only
    when a maintainer dispatches it or on the weekly schedule.
    """
    triggers = _triggers(workflow)
    assert set(triggers) == {"workflow_dispatch", "schedule"}
    assert "pull_request" not in triggers
    assert "push" not in triggers


def test_schedule_is_weekly(workflow):
    assert _triggers(workflow)["schedule"] == [{"cron": "17 6 * * 1"}]


def test_promote_needs_the_gate_and_nothing_else(workflow):
    """One edge, in one direction: evaluation strictly before promotion."""
    assert workflow["jobs"]["promote"]["needs"] == "gate"
    assert "needs" not in workflow["jobs"]["gate"]


def test_promotion_requires_a_human(workflow):
    """The approval gate is a GitHub Environment, not a comment in the YAML."""
    assert workflow["jobs"]["promote"]["environment"] == {"name": "production"}


def test_a_running_promotion_is_never_cancelled(workflow):
    """Half-moved alias plus a killed job is worse than a stale alias."""
    assert workflow["concurrency"] == {
        "group": "promote-production",
        "cancel-in-progress": False,
    }


def test_gate_job_cannot_promote(workflow):
    """The evaluating job passes --dry-run and nothing that could override it."""
    runs = _commands(workflow, "gate")
    assert "--dry-run" in runs
    assert "--skip-registry" not in runs, (
        "the gate must still register the Candidate alias so the version is "
        "inspectable before a human approves it"
    )
    # A second, un-flagged invocation would promote behind the human's back.
    assert runs.count("scripts/promote_model.py") == 1, (
        "a second invocation could promote"
    )


def test_promote_job_is_the_one_that_moves_the_alias(workflow):
    runs = _commands(workflow, "promote")
    assert "scripts/promote_model.py" in runs
    assert "--dry-run" not in runs, "the promote job must not pass --dry-run"


def test_promote_job_re_measures_before_promoting(workflow):
    """An approval covers the numbers a human read, so they must still hold."""
    runs = _runs(workflow, "promote")
    assert "--candidate-version" in runs
    assert "models/onnx/model_int8.onnx" in runs


def test_both_jobs_resolve_the_graph_on_their_own_runner(workflow):
    """A path from one job's filesystem is meaningless in the next.

    The gate and promote jobs run on separate runners. If the promote job read
    the candidate path from a job output, it would be handed a path into a
    machine that no longer exists -- and it would fail at the moment of
    promotion, after the human had already approved. Each job resolves its own
    copy, through one shared definition so the two cannot drift apart.
    """
    for job in ("gate", "promote"):
        steps = [s for s in _steps(workflow, job) if "uses" in s]
        assert any(s["uses"] == "./.github/actions/resolve-candidate" for s in steps), (
            f"{job} does not resolve the candidate graph itself"
        )


def test_no_filesystem_path_crosses_the_job_boundary(workflow):
    """Job outputs are scalars; a resolved path is not."""
    outputs = workflow["jobs"]["gate"].get("outputs", {})
    text = yaml.safe_dump(outputs)
    assert "candidate_onnx" not in text, (
        "the gate job must not export a path from its own filesystem"
    )
    assert outputs["candidate_version"] == ("${{ steps.inputs.outputs.version }}"), (
        "only the version may cross the boundary"
    )


def test_the_promote_job_gates_the_version_a_human_approved(workflow):
    """The version is pinned to the gate job; the path is re-resolved locally.

    Re-resolving the *version* independently in both jobs would be a second bug:
    on a scheduled run with no explicit version, the gate job reads the
    Production alias, and if anything promoted in between, the promote job
    would gate a different model than the one that was approved.
    """
    uses = [s for s in _steps(workflow, "promote") if "uses" in s]
    resolver = next(
        s for s in uses if s["uses"] == "./.github/actions/resolve-candidate"
    )
    assert resolver["with"]["version"] == "${{ needs.gate.outputs.candidate_version }}"
    assert "needs.gate.outputs.candidate_onnx" not in yaml.safe_dump(workflow)


def test_both_jobs_pull_the_split_and_tokenizer(workflow):
    for job in ("gate", "promote"):
        runs = _runs(workflow, job)
        for pull in REQUIRED_PULLS:
            assert pull in runs, f"{job} is missing `{pull}`"


def test_both_jobs_check_out_with_full_history(workflow):
    for job in ("gate", "promote"):
        checkout = next(
            s for s in _steps(workflow, job) if "checkout" in s.get("uses", "")
        )
        assert int(checkout["with"]["fetch-depth"]) == 0


def test_dvc_credentials_are_written_privately(workflow):
    for job in ("gate", "promote"):
        runs = _runs(workflow, job)
        assert "umask 077" in runs
        assert ".dvc/config.local" in runs


def test_the_token_is_never_echoed(workflow):
    """It goes into a 0600 file, never onto a log or the step summary."""
    for job in ("gate", "promote"):
        for step in _steps(workflow, job):
            script = step.get("run", "")
            for line in script.splitlines():
                stripped = line.strip()
                if "DAGSHUB_TOKEN" in stripped:
                    assert (
                        stripped.startswith("printf") or '$DAGSHUB_TOKEN"' in stripped
                    ), f"{job}: {stripped!r} would print the token"
    summary_steps = [
        s
        for job in ("gate", "promote")
        for s in _steps(workflow, job)
        if "GITHUB_STEP_SUMMARY" in s.get("run", "")
    ]
    assert summary_steps
    for step in summary_steps:
        assert "TOKEN" not in step["run"]


def test_secrets_come_from_the_secret_store(workflow):
    raw = WORKFLOW.read_text()
    assert "${{ secrets.DAGSHUB_USER }}" in raw
    assert "${{ secrets.DAGSHUB_TOKEN }}" in raw
    # A literal credential in a committed workflow is the failure this catches.
    assert "dagshub.com/shhth0034" in raw
    for line in raw.splitlines():
        if "dagshub.com" in line and "%s" not in line and "url: https://" not in line:
            assert "@" not in line, f"possible inline credential: {line.strip()!r}"


def test_permissions_are_least_privilege(workflow):
    """Registry access comes from Dagshub tokens, not from a GITHUB_TOKEN."""
    assert workflow["permissions"] == {"contents": "read"}


def test_the_decision_artifact_is_always_uploaded(workflow):
    upload = next(
        s for s in _steps(workflow, "gate") if "upload-artifact" in s.get("uses", "")
    )
    assert upload["if"] == "always()"
    assert upload["with"]["if-no-files-found"] == "error"
    assert "promotion_*.json" in upload["with"]["path"]


def test_a_rejected_decision_fails_the_job(workflow):
    """Exit 1 from the script is the CI failure; a 'rejected' string is not."""
    runs = _runs(workflow, "gate")
    assert "promotion_*.json" in runs
    assert '"rejected"' in runs
    assert "exit 1" in runs


def test_floor_tolerance_default_matches_the_script(workflow):
    """Two places declare the floor tolerance; they must not drift apart."""
    dispatch = _triggers(workflow)["workflow_dispatch"]["inputs"]
    assert dispatch["floor_tolerance"]["default"] == "0.01"
    # The value is threaded through the step env, so check the whole job.
    assert "0.01" in yaml.safe_dump(workflow["jobs"]["gate"])
    assert "--floor-tolerance" in _commands(workflow, "gate")
    script = (REPO / "scripts" / "promote_model.py").read_text()
    # promote_model.py is a facade over promotion_*.py siblings, so the default
    # now lives in promotion_types.py. Read the whole set: pinning the facade
    # alone would pass against a file with no thresholds in it.
    script += "\n".join(
        path.read_text() for path in sorted((REPO / "scripts").glob("promotion_*.py"))
    )
    assert "floor_tolerance: float = 0.01" in script


def test_candidate_version_is_required_on_dispatch(workflow):
    dispatch = _triggers(workflow)["workflow_dispatch"]["inputs"]
    assert dispatch["candidate_version"]["required"] is True
