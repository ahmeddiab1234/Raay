"""Tests for the candidate resolver and action workflows."""

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
WORKFLOW = REPO / ".github" / "workflows" / "promote.yml"
RESOLVER = REPO / ".github" / "actions" / "resolve-candidate" / "action.yml"


def test_the_shared_resolver_is_valid_yaml_and_refuses_external_weights():
    resolver = yaml.safe_load(RESOLVER.read_text())
    assert resolver["runs"]["using"] == "composite"
    assert set(resolver["outputs"]) == {"version", "onnx"}
    script = yaml.safe_dump(resolver)
    assert ".onnx.data" in script, (
        "a graph with an external sidecar must be refused, not gated as one file"
    )
    assert "download_artifacts" in script


def _resolver_program() -> str:
    """The python program embedded in the resolver's heredoc.

    Extracted rather than reimplemented, so the test exercises the script that
    actually runs. A substring assertion on the source only proves the words
    are present -- the message naming ``.onnx.data`` is in the source whether or
    not the guard is wired up.
    """
    resolver = yaml.safe_load(RESOLVER.read_text())
    script = "\n".join(s["run"] for s in resolver["runs"]["steps"] if "run" in s)
    # The marker carries a redirect ("<<'PY' >> \"$GITHUB_OUTPUT\""), so split
    # on it and then drop the remainder of that line.
    _, _, rest = script.partition("<<'PY'")
    body = rest.split("\n", 1)[1]
    body = body.rsplit("\nPY", 1)[0]
    return textwrap.dedent(body)


def _run_resolver_program(
    tmp_path: Path, artifact_dir: Path
) -> subprocess.CompletedProcess:
    """Execute the resolver's program against a stubbed MLflow.

    No registry, no network: the stub returns a local directory so the real
    logic -- the glob, the sidecar refusal, the emitted output line -- runs
    unchanged against a graph we control.
    """
    stub = tmp_path / "stubs"
    (stub / "raay" / "config").mkdir(parents=True)
    (stub / "raay" / "__init__.py").write_text("")
    (stub / "raay" / "config" / "__init__.py").write_text("")
    (stub / "raay" / "config" / "env.py").write_text(
        "def load_environment():\n    pass\n"
    )
    (stub / "mlflow.py").write_text(
        "import os\n"
        "class artifacts:\n"
        f"    @staticmethod\n"
        f"    def download_artifacts(uri):\n        return {str(artifact_dir)!r}\n"
        "def __getattr__(name):\n    raise AttributeError(name)\n"
    )
    program = tmp_path / "resolver.py"
    program.write_text(_resolver_program())
    env = {
        **os.environ,
        "PYTHONPATH": str(stub),
        "MODEL_NAME": "ArabicSentiment",
    }
    return subprocess.run(
        [sys.executable, str(program), "7"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def test_the_resolver_emits_a_path_for_a_self_contained_graph(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "model.onnx").write_bytes(b"")
    result = _run_resolver_program(tmp_path, artifacts)
    assert result.returncode == 0, result.stderr
    assert "onnx=" in result.stdout
    assert "model.onnx" in result.stdout


def test_a_version_with_several_graphs_is_refused(tmp_path: Path) -> None:
    """Two graphs in one version is ambiguous, so the resolver must not choose.

    Picking the first would be a silent coin flip about which model actually
    gets gated, and the report would name only the version. The sidecar refusal
    below exists for the same reason: a graph that is not what will be served
    must be refused rather than measured.
    """
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "b.onnx").write_bytes(b"")
    (artifacts / "a.onnx").write_bytes(b"")
    result = _run_resolver_program(tmp_path, artifacts)
    assert result.returncode != 0
    assert "several graphs" in result.stderr
    assert "a.onnx" in result.stderr and "b.onnx" in result.stderr
    assert "onnx=" not in result.stdout, "it emitted a path it had just refused"


def test_a_graph_with_external_weights_is_refused(tmp_path: Path) -> None:
    """The distilled graph needs its .onnx.data alongside it.

    Gating the .onnx without the sidecar would load a graph with missing
    initializers: ORT may still open the file and produce numbers, which is the
    worst outcome for a gate. The resolver has to refuse rather than measure
    garbage, and that behaviour is checked here by running the resolver.
    """
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "distilled.onnx").write_bytes(b"")
    (artifacts / "distilled.onnx.data").write_bytes(b"")
    result = _run_resolver_program(tmp_path, artifacts)
    assert result.returncode != 0
    assert "external weights" in result.stderr
    assert "onnx=" not in result.stdout, "it emitted a path it had just refused"


def test_a_missing_graph_is_an_error_not_an_empty_path(tmp_path: Path) -> None:
    """No graph in the artifacts must fail loudly, not gate a blank path."""
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    result = _run_resolver_program(tmp_path, artifacts)
    assert result.returncode != 0
    assert "no .onnx graph" in result.stderr
    assert "onnx=" not in result.stdout


def test_the_shared_resolver_passes_shellcheck():
    """The resolver's bash gets the same bar as the workflow's.

    Moving the step into a composite action takes it out of actionlint's
    reach, so without this the script would be the one piece of shell in the
    promotion path that nothing checks.
    """
    if shutil.which("shellcheck") is None:
        pytest.skip("shellcheck is not installed")
    resolver = yaml.safe_load(RESOLVER.read_text())
    scripts = [s["run"] for s in resolver["runs"]["steps"] if "run" in s]
    assert scripts, "the resolver has no shell to check"
    with tempfile.TemporaryDirectory() as tmp:
        for index, script in enumerate(scripts):
            path = Path(tmp) / f"resolve{index}.sh"
            path.write_text("#!/bin/bash\nset -euo pipefail\n" + script)
            result = subprocess.run(
                ["shellcheck", "-S", "warning", str(path)],
                capture_output=True,
                text=True,
                check=False,
            )
            assert result.returncode == 0, result.stdout


def test_workflow_passes_actionlint():
    """Catches expression/shell problems the YAML parser cannot see."""
    binary = os.environ.get("ACTIONLINT_BIN")
    if binary is None:
        pytest.skip("actionlint is not installed")
    # actionlint 1.7.7 parses anything handed to it as a workflow, so it
    # cannot check a composite action (verified against a minimal one: it
    # reports the missing `on:` and `jobs:`). The action's bash is checked by
    # shellcheck below instead; its YAML by the structural tests.
    result = subprocess.run(
        [binary, str(WORKFLOW)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
