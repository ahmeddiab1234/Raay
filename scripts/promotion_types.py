"""Shared vocabulary for the promotion gate: config, gate results, helpers.

Every threshold the decision depends on lives in :class:`Config` so a caller can
thread an override without reaching into a function. The defaults are the numbers
this project actually agreed to, not round placeholders -- ``floor_tolerance`` in
particular is calibrated against the graph that is already serving.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from raay.enums.constants import DefaultPaths, Models

_MODEL = Models.REGISTERED_BASELINE.value
_ALIAS_CANDIDATE = "Candidate"
_ALIAS_PRODUCTION = "Production"

_TOOL_TAG = "scripts.promote_model"

# The classes whose recall is watched. Neutral is the one that carries the whole
# imbalance problem (5% of the split), and Negative is the class a business
# cares most about getting right; a gate that only watched macro F1 could be
# satisfied while both of these quietly collapse.
_WATCHED_CLASSES = ("neutral", "negative")


class PromotionError(RuntimeError):
    """Something prevented the gate from running at all."""


class RegistryClient(Protocol):
    """The slice of the MLflow client this tool needs."""

    def get_model_version_by_alias(self, name: str, alias: str) -> str | None: ...

    def set_registered_model_alias(
        self, name: str, alias: str, version: str
    ) -> None: ...

    def get_model_version(self, name: str, version: str) -> Any: ...

    def get_model_version_download_uri(self, name: str, version: str) -> str: ...

    def set_model_version_tag(
        self, name: str, version: str, key: str, value: str
    ) -> None: ...


@dataclass(frozen=True)
class Config:
    """Every threshold the decision depends on, in one place.

    Defaults are the numbers this project actually agreed to, not round
    placeholders: ``floor_tolerance`` is 0.01 because the int8 graph that is
    currently in Production sits 0.0038 *below* the fp32 baseline's macro F1,
    so a zero-tolerance floor would reject the model that is already serving.
    """

    test_split: Path = Path(DefaultPaths.TEST_SPLIT.value)
    tokenizer_dir: str = DefaultPaths.BASELINE_MODEL.value
    floor_report: Path = Path(DefaultPaths.EVAL_BASELINE.value)
    parity_report: Path = Path("reports/onnx_int8_parity.json")
    dvc_lock: Path = Path("dvc.lock")
    report_dir: Path = Path("reports")

    model_name: str = Models.TEACHER.value
    max_length: int = 128
    latency_runs: int = 50
    # Both graphs have been scored over 7,209 rows by the time the first call is
    # timed, so the process is warm but neither thread pool is in steady state.
    # Three warm-up calls were not enough to stop the first series paying for
    # the other: 10 is, and it costs about a second.
    latency_warmup: int = 10
    eval_limit: int | None = None
    # The per-dialect breakdown re-scores the whole split once per dialect, so
    # it roughly doubles the gate's runtime. Off by default: the split is already
    # dialect-stratified, so the headline numbers are not dialect-blind, and a
    # human reading the report can turn it on when they want the breakdown.
    dialect_breakdown: bool = False

    # Gates.
    f1_tolerance: float = 0.005
    floor_tolerance: float = 0.01
    accuracy_tolerance: float = 0.01
    recall_tolerance: float = 0.02
    min_recall: dict[str, float] = field(
        default_factory=lambda: {"neutral": 0.10, "negative": 0.50}
    )
    latency_regression: float = 0.10
    max_size_mb: float | None = None
    min_label_agreement: float = 1.0
    max_parity_diff: float = 0.5
    require_frozen_split: bool = True


@dataclass(frozen=True)
class GateResult:
    """One gate, its verdict, and the numbers behind it."""

    name: str
    passed: bool
    observed: float | str | None
    threshold: float | str | None
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "observed": self.observed,
            "threshold": self.threshold,
            "detail": self.detail,
        }


@dataclass
class Graph:
    """A loaded graph: its metrics, plus the live objects needed to time it."""

    metrics: dict[str, Any]
    session: Any = None
    tokenizer: Any = None


@dataclass(frozen=True)
class Decision:
    """What the gate concluded, and on what evidence."""

    promoted: bool
    gates: list[GateResult]
    payload: dict[str, Any]

    @property
    def failed(self) -> list[GateResult]:
        return [gate for gate in self.gates if not gate.passed]


def _round(value: float, digits: int = 4) -> float:
    return float(round(float(value), digits))


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
