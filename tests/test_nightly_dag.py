"""Structural tests for the nightly Airflow DAG.

Airflow is deliberately *not* a project dependency (AGENTS.md: host-isolated via
`uv tool install`), so importing the DAG here would fail. Instead these tests
parse the file as text and pin the parts of the contract that are invisible in the
UI until they go wrong: the task chain, the short-circuit, and the fact that task 6
cannot half-apply a batch on top of an incomplete night.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

DAG = (
    Path(__file__).resolve().parents[1]
    / "airflow"
    / "dags"
    / "raay_nightly_batch_scoring.py"
)


@pytest.fixture(scope="module")
def source() -> str:
    return DAG.read_text()


@pytest.fixture(scope="module")
def tree(source: str) -> ast.Module:
    return ast.parse(source)


def test_the_dag_file_parses(source: str):
    """A syntax error here means the scheduler silently drops the DAG."""
    ast.parse(source)
    assert 'dag_id="raay_nightly_batch_scoring"' in source


def test_the_dag_declares_six_tasks(source: str):
    task_ids = [
        line.split('task_id="')[1].split('"')[0]
        for line in source.splitlines()
        if "task_id=" in line
    ]
    assert task_ids == [
        "materialize_daily_input",
        "score_daily_batch",
        "run_drift_check",
        "run_prediction_drift_check",
        "evaluate_retrain_trigger",
        "merge_validated_feedback",
        "check_pending_feedback",
    ]


def test_the_merge_task_is_chained_after_the_drift_chain(source: str):
    """`trigger >> has_feedback >> merge_feedback`, not a detached island.

    Detached means the merge can run while the drift chain has failed, half-applying
    a batch on top of an incomplete night -- and an operator looking at the UI
    sees a green task that raced the five it should follow.
    """
    assert "trigger >> has_feedback >> merge_feedback" in source
    # The pre-fix shape must not come back.
    assert "\n    has_feedback >> merge_feedback" not in source


def test_the_chain_still_runs_the_drift_tasks_in_order(source: str):
    assert "materialize >> score >> drift >> predict_drift >> trigger" in source


def test_the_predicate_uses_short_circuit_operator(source: str):
    """A skip is a readable task state; a Bash no-op looks like real work."""
    assert "ShortCircuitOperator(" in source
    assert 'task_id="check_pending_feedback"' in source
    assert "python_callable=has_pending_feedback" in source


def test_the_predicate_reads_the_files_not_a_counter(tree: ast.Module):
    """A counter would re-merge after a crash; reading the merged file cannot."""
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "has_pending_feedback"
    )
    assert "has_new_feedback" in ast.unparse(fn)
    assert "DefaultPaths.FEEDBACK_REVIEWED" in ast.unparse(fn)
    assert "DefaultPaths.FEEDBACK_MERGED" in ast.unparse(fn)


def test_the_predicate_imports_feedback_lazily(tree: ast.Module):
    """Module scope would drag pandas + rapidfuzz into every scheduler import."""
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "has_pending_feedback"
    )
    assert any(
        isinstance(node, (ast.Import, ast.ImportFrom))
        for node in fn.body
        if isinstance(node, ast.Expr) is False
    ), "expected a lazy import inside the predicate"


def test_the_merge_step_shells_out_to_the_feedback_module(tree: ast.Module):
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "feedback_step"
    )
    # ast.unparse keeps the docstring, so assert on the return expression: the
    # command must not invoke a scoring mode (no graph, no encoder, no frame
    # scoring), which is a property of what runs, not of the prose about it.
    returns = [ast.unparse(node) for node in fn.body if isinstance(node, ast.Return)]
    assert returns == [
        "return f'cd {REPO} && uv run python -m raay.data.feedback --mode merge'"
    ]
    assert "batch_score" not in returns[0]
    assert "make-input" not in returns[0]


def test_the_merge_step_never_reviews_raw_captures(tree: ast.Module):
    """`--mode review` would rewrite the operator-approved reviewed file.

    It drops every adjudicated_label a human typed in, and the reviewed file is
    git-tracked and DVC-hashed -- so the nightly job must only ever consume it.
    """
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "feedback_step"
    )
    assert "--mode review" not in ast.unparse(fn)


def test_the_merge_task_is_not_dvc_repro(tree: ast.Module):
    """`dvc repro` here would rewrite reports/feedback_metrics.json and dvc.lock
    on every night, including the ones where nothing was pending."""
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "feedback_step"
    )
    assert "dvc" not in ast.unparse(fn)


def test_no_task_is_allowed_to_touch_the_frozen_test_split(tree: ast.Module):
    """The feedback module is the only writer, and it is merge-only.

    A nightly job that regenerated data/processed/test.csv would change the hash
    scripts/promote_model.py:check_frozen_split verifies against dvc.lock, making
    every future promotion exit 2 with no report.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.endswith("_step"):
            body = ast.unparse(node)
            assert "raay.data.split" not in body
            assert "--mode make-input" not in body or node.name == "materialize"
