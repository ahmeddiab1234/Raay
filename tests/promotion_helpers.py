"""Shared fakes, fixtures and fixture payloads for the promotion-gate tests.

The gate decides whether a model is allowed to start serving traffic, so the
tests care about the failure paths rather than the happy one: a macro-F1
regression hidden behind a good accuracy, a Neutral class quietly collapsing, a
test split that has drifted from ``dvc.lock``, and -- the one that would be most
damaging -- a failed candidate reaching the ``Production`` alias anyway.

The graph evaluation is faked, so everything here is hermetic and instant. The
real thresholds are exercised against the real recorded numbers
(``eval_baseline`` 0.6407 vs the INT8 graph's 0.6369) because that pair is what
the tolerances were calibrated against, and a threshold change that quietly
rejects the model already in Production is exactly the regression these tests
exist to catch.

Graph evaluation is faked because the real thing needs an ORT session; the
recorded production numbers below are what keep the fakes honest.

Split out of ``test_promote_model.py`` so the four gate test modules share one
copy. These are plain functions, not fixtures: each test module declares its own
three-line ``@pytest.fixture`` wrapper, which keeps the names ``repo``/``cfg``
resolvable in that module and avoids an autouse patch that is global to the
suite. ``tests/`` has no ``__init__.py``, so the directory is on ``sys.path`` and
``from promotion_helpers import ...`` works directly.
"""

import json
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import promote_model as pm
import promotion_graph
import promotion_split
from promote_model import Config, file_md5

LABELS = ["positive", "negative", "neutral"]
PROD_ONNX = "models/onnx/model_int8.onnx"
CAND_ONNX = "models/onnx/candidate.onnx"

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def promotion_sources() -> str:
    """Every source file the promotion gate actually runs.

    ``scripts/promote_model.py`` is a facade over ``promotion_*.py`` siblings,
    so a structural pin that read only the facade would pass against a file
    containing no arithmetic at all -- which is exactly how such a pin rots.
    Reading the whole set keeps the assertion pointed at the code that runs, and
    a negative assertion ("this anti-pattern is absent") stays honest too.
    """
    files = sorted(SCRIPTS.glob("promotion_*.py")) + [SCRIPTS / "promote_model.py"]
    return "\n".join(path.read_text() for path in files)


def metrics(**over: object) -> dict:
    """A metrics block shaped exactly like ``evaluate_graph``'s output."""
    base: dict = {
        "onnx_path": PROD_ONNX,
        "size_mb": 129.81,
        "label_names": list(LABELS),
        "id2label": {"0": "positive", "1": "negative", "2": "neutral"},
        "accuracy": 0.8503,
        "f1_macro": 0.6369,
        "f1_weighted": 0.8612,
        "per_class": {
            "positive": {
                "precision": 0.93,
                "recall": 0.9331,
                "f1": 0.9315,
                "support": 5500,
            },
            "negative": {
                "precision": 0.852,
                "recall": 0.8599,
                "f1": 0.8559,
                "support": 1300,
            },
            "neutral": {
                "precision": 0.198,
                "recall": 0.123,
                "f1": 0.1553,
                "support": 409,
            },
        },
        "confusion_matrix": [[5000, 200, 300], [100, 1100, 100], [40, 40, 329]],
        "sample_size": 7209,
        "latency": {"p50_ms": 12.0, "p95_ms": 27.5, "runs": 30},
        "metadata": {"model_name": "aubmindlab/bert-base-arabertv02"},
        "dialect_breakdown": None,
    }
    base.update(over)
    return base


def baseline_report() -> dict:
    """``reports/eval_baseline.json`` as recorded, at full precision."""
    return {
        "model_dir": "models/baseline/final",
        "split": "test",
        "sample_size": 7209,
        "accuracy": 0.84917,
        "f1_macro": 0.6406636813692393,
        "f1_weighted": 0.86197,
        "label_names": list(LABELS),
        "per_class": {
            "positive": {
                "precision": 0.928,
                "recall": 0.9361,
                "f1": 0.932,
                "support": 5500,
            },
            "negative": {
                "precision": 0.855,
                "recall": 0.8577,
                "f1": 0.8564,
                "support": 1300,
            },
            "neutral": {
                "precision": 0.232,
                "recall": 0.1393,
                "f1": 0.1746,
                "support": 409,
            },
        },
        "dialect_breakdown": None,
    }


class FakeRegistry:
    """Stands in for ``MlflowClient`` and records the alias mutations."""

    def __init__(self, production: str | None = "4") -> None:
        self.production = production
        self.calls: list[tuple] = []
        self.aliases: dict[str, str] = {"Production": production} if production else {}
        # Phase 6 step 3 provenance: tag writes are recorded separately from
        # alias moves so a test can assert the Production flip happened *and*
        # the trigger reason was stamped.
        self.tags: dict[tuple[str, str], dict[str, str]] = {}

    def get_model_version_by_alias(self, name: str, alias: str) -> str | None:
        self.calls.append(("get", name, alias))
        if alias == "Production" and self.production is None:
            raise RuntimeError(f"Registered Model alias {alias} not found")
        return self.aliases.get(alias)

    def set_registered_model_alias(self, name: str, alias: str, version: str) -> None:
        self.calls.append(("set", name, alias, version))
        self.aliases[alias] = version

    def set_model_version_tag(
        self, name: str, version: str, key: str, value: str
    ) -> None:
        self.calls.append(("tag", name, version, key, value))
        self.tags.setdefault((name, version), {})[key] = value

    def tags_for(self, version: str) -> dict[str, str]:
        return dict(self.tags.get(("ArabicSentiment", version), {}))

    def sets(self, alias: str) -> list[str]:
        return [call[3] for call in self.calls if call[0] == "set" and call[2] == alias]


def make_repo(tmp_path: Path) -> Path:
    """A miniature repo: split, dvc.lock, floor report, parity report, graphs."""
    split = tmp_path / "test.csv"
    split.write_text(
        "text,label\n" + "\n".join(f"مراجعة {i},{i % 3}" for i in range(30))
    )
    digest = file_md5(split)

    lock = tmp_path / "dvc.lock"
    lock.write_text(
        "stages:\n"
        "  split:\n"
        "    outs:\n"
        "    - path: data/processed/test.csv\n"
        f"      md5: {digest}\n"
    )
    (tmp_path / "floor.json").write_text(json.dumps(baseline_report()))
    (tmp_path / "parity.json").write_text(
        json.dumps({"n_samples": 6, "max_abs_diff": 0.4646, "label_agreement": 1.0})
    )
    (tmp_path / "prod.onnx").write_bytes(b"x" * 2048)
    (tmp_path / "cand.onnx").write_bytes(b"x" * 1024)
    return tmp_path


def make_cfg(repo: Path) -> Config:
    """A ``Config`` pointing at the artefacts ``make_repo`` just wrote."""
    return Config(
        test_split=repo / "test.csv",
        floor_report=repo / "floor.json",
        parity_report=repo / "parity.json",
        dvc_lock=repo / "dvc.lock",
        report_dir=repo / "reports",
    )


def patch_eval(
    candidate: dict | None = None,
    production: dict | None = None,
    cand_paths: tuple[str, ...] = (CAND_ONNX,),
) -> None:
    """Replace graph loading with fixture metrics, keyed by path.

    ``cand_paths`` exists because the CLI tests point ``--candidate-onnx`` at a
    tmp file rather than the module constant, and a fake that silently answered
    "production" for the candidate would have made a rejection test pass for the
    wrong reason.
    """
    cand, prod = candidate or metrics(), production or metrics()

    def fake_load(cfg: Config, onnx_path: str, frame) -> "pm.Graph":
        return pm.Graph(metrics=dict(cand if onnx_path in cand_paths else prod))

    # A Graph with no session is what the fake returns, and
    # measure_latency_pair skips graphs without one, so the fixture latency
    # values survive untouched -- which is what lets these tests drive the
    # latency and size gates at all.
    pm.load_graph = fake_load  # type: ignore[assignment]
    promotion_graph.load_graph = fake_load  # type: ignore[assignment]


@contextmanager
def restore_eval():
    """Undo the module-attribute patching that ``patch_eval`` and friends do.

    A context manager rather than a shared fixture: each test module declares its
    own thin ``@pytest.fixture(autouse=True)`` wrapper, because importing a
    fixture by name would collide with the many tests whose *parameter* is also
    called ``cfg`` (ruff F811), and putting it in ``tests/conftest.py`` would
    make an autouse patch global to the whole suite for no gain.
    """
    original = pm.load_graph
    original_graph = promotion_graph.load_graph
    original_split = promotion_split.load_split
    try:
        yield
    finally:
        pm.load_graph = original
        promotion_graph.load_graph = original_graph
        promotion_split.load_split = original_split


class FakeSession:
    """Stands in for an ORT session, which starts with no ``.config``."""

    def __init__(self) -> None:
        self.config = None
        self.runs = 0

    def run(self, names, feeds):
        self.runs += 1
        return [np.zeros((1, 3), dtype=np.float32)]


class FakeConfig:
    def __init__(self) -> None:
        self.id2label = {0: "positive", 1: "negative", 2: "neutral"}


class FakeTokenizer:
    def __call__(self, texts, **kwargs):
        return {
            "input_ids": np.zeros((len(texts), 8), dtype=np.int64),
            "attention_mask": np.ones((len(texts), 8), dtype=np.int64),
        }
