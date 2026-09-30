"""Gate a candidate model before it is allowed to become Production.

The registry is not a place where a decision should be made by hand, because
the interesting decisions are comparisons: is this model better than the one
that is currently serving, and is it still above the floor we agreed to? Both
are easy to eyeball and get wrong -- a 0.5% macro-F1 regression is invisible
next to an 85% accuracy headline, and accuracy is exactly the number that hides
a model which has learned to ignore the minority classes. The held-out Neutral
class is 5% of the test set, so a model can lose *all* of it and still report a
comfortable accuracy.

The flow is therefore deliberately one-directional:

1. register the candidate under the ``Candidate`` alias -- never Production;
2. evaluate it on the frozen, DVC-pinned test split, using the same inference
   path the service uses, on the same machine as the production graph so the
   latency comparison is like-for-like;
3. run every gate, all-or-nothing;
4. only then move the ``Production`` alias, and only if a human approved it.

On any failure the aliases are left exactly as they were and the exit code is
non-zero, so CI fails instead of quietly shipping. The decision, every gate's
observed value and its threshold, and the provenance of each number are written
to ``reports/promotion_<version>.json`` either way.

Local rehearsal without a registry:

    python scripts/promote_model.py --candidate-version 5 --skip-registry
"""

import argparse
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import yaml
from loguru import logger

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


def _round(value: float, digits: int = 4) -> float:
    return float(round(float(value), digits))


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locked_test_split_md5(lock_path: Path) -> str | None:
    """The md5 ``dvc.lock`` pins for the test split, or None if not recorded."""
    if not lock_path.exists():
        return None
    lock = yaml.safe_load(lock_path.read_text()) or {}
    for stage in (lock.get("stages") or {}).values():
        for out in stage.get("outs") or []:
            if str(out.get("path", "")).endswith("test.csv"):
                return out.get("md5")
    return None


def check_frozen_split(cfg: Config) -> GateResult:
    """Refuse to gate against a test set that has drifted from ``dvc.lock``.

    Without this the gate would happily evaluate on whatever happens to be on
    disk, and a "the candidate regressed" verdict could really mean "someone
    regenerated the split".
    """
    locked = locked_test_split_md5(cfg.dvc_lock)
    if locked is None:
        return GateResult(
            "frozen_test_split",
            not cfg.require_frozen_split,
            None,
            cfg.dvc_lock.name,
            "dvc.lock does not pin the test split",
        )
    if not cfg.test_split.exists():
        raise PromotionError(f"{cfg.test_split} is missing; run `dvc repro` first")
    actual = file_md5(cfg.test_split)
    return GateResult(
        "frozen_test_split",
        actual == locked,
        actual,
        locked,
        f"{cfg.test_split} matches dvc.lock"
        if actual == locked
        else "test split drifted",
    )


def load_split(cfg: Config) -> pd.DataFrame:
    if not cfg.test_split.exists():
        raise PromotionError(
            f"{cfg.test_split} is missing; the gate never evaluates on a split it "
            "cannot verify against dvc.lock"
        )
    frame = pd.read_csv(cfg.test_split)
    if cfg.eval_limit:
        frame = frame.head(cfg.eval_limit)
    return frame


@dataclass
class Graph:
    """A loaded graph: its metrics, plus the live objects needed to time it."""

    metrics: dict[str, Any]
    session: Any = None
    tokenizer: Any = None


def _one_call(graph: Graph, texts: list[str], index: int, cfg: Config) -> float:
    from raay.serving.serve import _preprocess

    text = _preprocess(texts[index % len(texts)], cfg.model_name)
    enc = graph.tokenizer(
        [text],
        truncation=True,
        padding=True,
        max_length=cfg.max_length,
        return_tensors="np",
    )
    start = time.perf_counter()
    graph.session.run(
        ["logits"], {key: enc[key] for key in ("input_ids", "attention_mask")}
    )
    return (time.perf_counter() - start) * 1000.0


def _summarize(samples: list[float], method: str) -> dict[str, Any]:
    return {
        "p50_ms": _round(np.percentile(samples, 50), 3),
        "p95_ms": _round(np.percentile(samples, 95), 3),
        "runs": len(samples),
        "method": method,
    }


def measure_latency(graph: Graph, texts: list[str], cfg: Config) -> None:
    """Time one graph in place, mirroring the batch-1 method in ``benchmark.py``."""
    if graph.session is None:
        return
    samples: list[float] = []
    for run in range(cfg.latency_warmup + cfg.latency_runs):
        if run == cfg.latency_warmup:
            samples.clear()
        samples.append(_one_call(graph, texts, run, cfg))
    graph.metrics["latency"] = _summarize(samples, "single graph, same process")


def _noise_pct(reference: dict[str, Any], control_p95: float) -> float | None:
    """How far apart two series of the *same* graph landed, as a percentage."""
    base = reference.get("p95_ms")
    if not base:
        return None
    return _round(abs(control_p95 - base) / base * 100.0, 2)


def measure_latency_pair(
    candidate: Graph, production: Graph, texts: list[str], cfg: Config
) -> None:
    """Time both graphs in one interleaved loop, in place.

    Measuring them one after the other is not good enough. Two runs of the *same*
    int8 graph on an otherwise idle 2-core box came out 21% apart -- larger than
    the 10% regression band this gate allows -- so a sequential measurement would
    be deciding on CPU contention rather than on model quality.

    Interleaving puts both graphs through the same load conditions, and swapping
    the order every round cancels any first-slot advantage. What is left is the
    difference, which is the only part that is about the model.

    The third series is a **control**: the production graph, timed again through
    the same loop. Two series of the *same* graph differing by N% is a direct
    measurement of how much of any observed gap is the host rather than the
    model. On an idle 2-core box that number is 15-21%, which is larger than the
    10% band this gate allows -- so without the control the gate would report
    "your model is slower" about a machine that cannot tell the difference.

    The measurement is relative by construction, never compared against
    ``reports/benchmark_table.csv``: a GitHub runner is several times slower than
    the box that produced that table, and an absolute budget copied from it would
    fail for reasons that have nothing to do with the candidate.
    """
    if candidate.session is None or production.session is None:
        return
    left: list[float] = []
    right: list[float] = []
    control: list[float] = []
    for run in range(cfg.latency_warmup + cfg.latency_runs):
        if run == cfg.latency_warmup:
            left.clear()
            right.clear()
            control.clear()
        series = [(candidate, left), (production, right), (production, control)]
        # Rotating rather than reversing: three slots, so every graph takes every
        # position the same number of times.
        offset = run % 3
        series = series[offset:] + series[:offset]
        for graph, sink in series:
            sink.append(_one_call(graph, texts, run, cfg))
    candidate.metrics["latency"] = _summarize(left, "interleaved A/B + control")
    production.metrics["latency"] = _summarize(right, "interleaved A/B + control")
    left_p95 = float(np.percentile(left, 95))
    right_p95 = float(np.percentile(right, 95))
    control_p95 = float(np.percentile(control, 95))
    production.metrics["latency"]["control_p95_ms"] = _round(control_p95, 3)
    production.metrics["latency"]["control_gap_pct"] = _noise_pct(
        production.metrics["latency"], production.metrics["latency"]["control_p95_ms"]
    )
    # The control gap is one sample of the host's noise; the spread across all
    # three series is the honest bound. Two identical graphs measured at 56.0
    # and 64.1 ms can produce a control gap of only 4% while the true spread is
    # 14% -- and a 4% "noise" would then read as a real 14% regression.
    production.metrics["latency"]["noise_pct"] = _round(
        (max(left_p95, right_p95, control_p95) - min(left_p95, right_p95, control_p95))
        / min(left_p95, right_p95, control_p95)
        * 100.0,
        2,
    )


def latency_session_options():
    """Session options for the graph being timed, not for the graph being scored.

    The default ORT configuration sizes a thread pool to the machine and leaves
    its workers spinning between runs. One such pool is fine; the gate needs
    *two* in one process, and they fight: measured here, the spread across the
    three series of one identical graph was 1.2% with a single session and 6.3%
    with two, purely from that interference. Turning spinning off leaves the
    pool in place -- the graph is ~3x slower on one thread, so the pool is not
    optional -- and brings the two-session figure back to 2.8%.

    Only the gate does this. ``serve.py`` builds its own session and is
    deliberately left alone: this is a measurement fix, not a serving change.
    """
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return options


def load_graph(cfg: Config, onnx_path: str, frame: pd.DataFrame) -> Graph:
    """Score one graph on the frozen split the way the service would.

    Latency is deliberately *not* measured here. Two graphs have to be timed
    against each other in one interleaved loop to get a number that means
    anything, and that is :func:`measure_latency_pair`'s job.
    """
    from transformers import AutoConfig, AutoTokenizer

    from raay.training.evaluate import (
        dialect_breakdown,
        evaluate_on_split,
        load_onnx_session,
    )

    if not Path(onnx_path).exists():
        raise PromotionError(f"graph not found: {onnx_path}")
    session = load_onnx_session(onnx_path, session_options=latency_session_options())

    # An ORT session has no `.config`, and `evaluate_on_split` reads id2label
    # from there. Without this the labels fall back to alphabetical order
    # (negative, neutral, positive) and Neutral/Negative get silently swapped
    # against a model whose ids are positive=0, negative=1, neutral=2.
    session.config = AutoConfig.from_pretrained(cfg.tokenizer_dir)
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_dir)

    report = evaluate_on_split(
        frame, session, tokenizer, cfg.model_name, cfg.max_length
    )
    report["dialect_breakdown"] = (
        dialect_breakdown(frame, session, tokenizer, cfg.model_name, cfg.max_length)
        if cfg.dialect_breakdown
        else None
    )
    id2label = {
        int(k): v for k, v in (getattr(session.config, "id2label", None) or {}).items()
    }
    size_mb = Path(onnx_path).stat().st_size / 1_048_576
    metrics = {
        "onnx_path": onnx_path,
        "size_mb": _round(size_mb, 2),
        "label_names": report["label_names"],
        "id2label": {str(k): v for k, v in id2label.items()},
        "accuracy": _round(report["accuracy"]),
        "f1_macro": _round(report["f1_macro"]),
        "f1_weighted": _round(report["f1_weighted"]),
        "per_class": {
            name: {
                "precision": _round(values["precision"]),
                "recall": _round(values["recall"]),
                "f1": _round(values["f1"]),
                "support": values["support"],
            }
            for name, values in report["per_class"].items()
        },
        "confusion_matrix": report["confusion_matrix"],
        "sample_size": report["sample_size"],
        "metadata": report.get("metadata", {}),
    }
    return Graph(metrics=metrics, session=session, tokenizer=tokenizer)


def evaluate_graph(cfg: Config, onnx_path: str, frame: pd.DataFrame) -> dict[str, Any]:
    """``load_graph`` plus a standalone timing of that one graph."""
    graph = load_graph(cfg, onnx_path, frame)
    measure_latency(graph, frame["text"].tolist()[:64], cfg)
    return graph.metrics


def read_parity(cfg: Config) -> dict[str, Any]:
    """The existing ONNX-vs-PyTorch logit check, loaded rather than re-derived.

    Re-exporting the graph here would need the 540 MB fp32 checkpoint and a
    torch install, neither of which belongs in a promotion gate. The report is a
    committed artifact of the export step, so it is read -- and its absence is a
    gate failure rather than a silent skip.
    """
    if not cfg.parity_report.exists():
        raise PromotionError(
            f"{cfg.parity_report} is missing; run "
            "`python -m raay.inference.quantize_onnx` to produce it"
        )
    return json.loads(cfg.parity_report.read_text())


def _parity_section(report: dict[str, Any]) -> dict[str, Any]:
    """Flatten the report's shape: it is either flat or keyed by variant."""
    if "max_abs_diff" in report:
        return report
    for value in report.values():
        if isinstance(value, dict) and "max_abs_diff" in value:
            return value
    raise PromotionError("parity report has no max_abs_diff")


# --- the gates ---------------------------------------------------------------


def gate_f1_macro(candidate: dict, production: dict, cfg: Config) -> GateResult:
    """The checklist rule: CI fails if f1_macro drops.

    Compared against the *production* number, with a small tolerance, so noise
    of a fraction of a point does not block a release but a real regression does.
    """
    floor = production["f1_macro"] - cfg.f1_tolerance
    passed = candidate["f1_macro"] >= floor
    return GateResult(
        "f1_macro_not_regressed",
        passed,
        candidate["f1_macro"],
        _round(floor),
        f"production={production['f1_macro']} tolerance={cfg.f1_tolerance}",
    )


def gate_f1_macro_floor(candidate: dict, baseline: dict, cfg: Config) -> GateResult:
    floor = baseline["f1_macro"] - cfg.floor_tolerance
    passed = candidate["f1_macro"] >= floor
    return GateResult(
        "f1_macro_above_floor",
        passed,
        candidate["f1_macro"],
        _round(floor),
        f"baseline={baseline['f1_macro']} tolerance={cfg.floor_tolerance}",
    )


def gate_accuracy(candidate: dict, production: dict, cfg: Config) -> GateResult:
    floor = production["accuracy"] - cfg.accuracy_tolerance
    passed = candidate["accuracy"] >= floor
    return GateResult(
        "accuracy_not_regressed",
        passed,
        candidate["accuracy"],
        _round(floor),
        "redundant with f1_macro on purpose: a uniformly worse model is a real "
        "regression even when the macro average hides it",
    )


def gate_recall(candidate: dict, production: dict, cfg: Config) -> list[GateResult]:
    """Per-class recall against production, and an absolute floor for the
    watched classes.

    Both halves matter. The relative half stops a model losing a class it used
    to handle; the absolute half stops Production and the candidate from both
    being bad at Neutral, which a pure relative comparison would happily allow
    forever.
    """
    results: list[GateResult] = []
    for name in candidate["per_class"]:
        got = candidate["per_class"][name]["recall"]
        was = production["per_class"].get(name, {}).get("recall")
        if was is not None:
            floor = was - cfg.recall_tolerance
            results.append(
                GateResult(
                    f"recall_{name}",
                    got >= floor,
                    got,
                    _round(floor),
                    f"production={was} tolerance={cfg.recall_tolerance}",
                )
            )
        else:
            results.append(
                GateResult(
                    f"recall_{name}",
                    False,
                    got,
                    None,
                    f"production has no {name!r} class to compare against",
                )
            )
        if name in _WATCHED_CLASSES:
            absolute = cfg.min_recall.get(name, 0.0)
            results.append(
                GateResult(
                    f"recall_{name}_absolute_floor",
                    got >= absolute,
                    got,
                    absolute,
                    f"{name} is the imbalance class; an absolute floor is the only "
                    "check that fails when production itself is weak",
                )
            )
    return results


def gate_latency(candidate: dict, production: dict, cfg: Config) -> GateResult:
    """p95 against a relative budget, with a null control to blame correctly.

    A candidate over budget fails, full stop -- a slow model is never waved
    through. But *why* it is over budget has to be honest: when the control
    series says two runs of the production graph itself differ by more than the
    allowed band, this host cannot resolve that band, and the useful report is
    "the measurement is inconclusive", not "your model is slow". Both outcomes
    block the promotion; only the diagnosis differs.
    """
    got = candidate["latency"]["p95_ms"]
    was = production["latency"]["p95_ms"]
    budget = was * (1.0 + cfg.latency_regression)
    noise = production["latency"].get("noise_pct")
    control_gap = production["latency"].get("control_gap_pct")
    passed = got <= budget
    detail = (
        f"production={was}ms regression_allowed={cfg.latency_regression:.0%} "
        f"(same host, same method)"
    )
    excess = ((got - budget) / budget * 100.0) if budget else 0.0
    if noise is not None:
        detail += f"; same-graph noise={noise}%"
    if control_gap is not None:
        detail += f" (control gap {control_gap}%)"
    if not passed and noise is not None and noise > cfg.latency_regression * 100.0:
        detail += f"; candidate is {_round(excess, 1)}% over budget"
    if not passed and noise is not None and noise > cfg.latency_regression * 100.0:
        detail += (
            " -- INCONCLUSIVE: three series of the same graph spread wider than the "
            "allowed band, so this host cannot resolve that band and a difference "
            "of this size is not evidence about the candidate. Re-run on a quieter "
            "runner, raise --latency-runs, or widen --latency-regression "
            "deliberately. The candidate is NOT being cleared by this."
        )
    return GateResult(
        "latency_p95_within_budget", passed, got, _round(budget, 3), detail
    )


def gate_size(candidate: dict, production: dict, cfg: Config) -> GateResult:
    limit = cfg.max_size_mb if cfg.max_size_mb is not None else production["size_mb"]
    passed = candidate["size_mb"] <= limit
    return GateResult(
        "model_size_within_limit",
        passed,
        candidate["size_mb"],
        _round(limit, 2),
        "defaults to the production graph's size: no growth without a reason",
    )


def gate_parity(cfg: Config) -> GateResult:
    section = _parity_section(read_parity(cfg))
    agreement = section.get("label_agreement")
    max_diff = float(section.get("max_abs_diff", float("inf")))
    # Dynamic INT8 quantization moves logits a long way (0.46 on the recorded
    # run) while keeping every argmax, so the *prediction* agreement is the
    # meaningful check and the logit bound is a backstop for a graph that has
    # not been quantized at all.
    agreement_ok = agreement is None or agreement >= cfg.min_label_agreement
    diff_ok = max_diff <= cfg.max_parity_diff
    passed = agreement_ok and diff_ok
    detail = f"{cfg.parity_report} max_abs_diff={_round(max_diff)}"
    if agreement is not None:
        detail += f" label_agreement={agreement}"
    return GateResult(
        "onnx_parity", passed, _round(max_diff), cfg.max_parity_diff, detail
    )


def gate_full_split(cfg: Config, candidate: dict) -> GateResult:
    """A truncated evaluation can never authorize a promotion.

    ``--eval-limit`` is a local affordance for a quick look, but the floor in
    ``eval_baseline.json`` was measured on the whole 7209-row split, so a
    head() of it is not a comparable measurement: the first 800 rows happen to
    hold fewer Neutral examples, which moves macro F1 and Neutral recall enough
    to flip a gate. Failing this gate makes that impossible to act on.
    """
    full = cfg.eval_limit is None
    return GateResult(
        "full_split_evaluated",
        full,
        candidate.get("sample_size"),
        "all rows",
        "a partial evaluation is a rehearsal, not a promotion decision"
        if not full
        else "the whole frozen split was scored",
    )


def gate_label_order(candidate: dict, production: dict) -> GateResult:
    """The two graphs must agree on what index 0 means.

    Comparing metrics from graphs with different label orders produces a
    confident, meaningless verdict, so it is checked explicitly rather than
    trusted.
    """
    same = candidate["label_names"] == production["label_names"]
    return GateResult(
        "label_order_matches_production",
        same,
        ",".join(candidate["label_names"]),
        ",".join(production["label_names"]),
        "the graphs disagree on which id means which class"
        if not same
        else "same label order",
    )


def evaluate_all_gates(
    cfg: Config,
    candidate: dict,
    production: dict,
    baseline: dict,
) -> list[GateResult]:
    gates: list[GateResult] = [
        check_frozen_split(cfg),
        gate_full_split(cfg, candidate),
        gate_label_order(candidate, production),
        gate_f1_macro(candidate, production, cfg),
        gate_f1_macro_floor(candidate, baseline, cfg),
        gate_accuracy(candidate, production, cfg),
        *gate_recall(candidate, production, cfg),
        gate_latency(candidate, production, cfg),
        gate_size(candidate, production, cfg),
        gate_parity(cfg),
    ]
    return gates


# --- reporting ---------------------------------------------------------------


def trigger_report_path(cfg: Config) -> Path | None:
    """Newest Phase 6 step 3 trigger report, or ``None`` when there is none.

    The trigger reports are git-tracked (unlike the per-run data-refresh
    reports), so on a fresh checkout the promoted version can be stamped with
    the reason it was retrained. Absent is normal: a promotion driven by hand,
    or one predating this step, simply carries no trigger tag.
    """
    reports = cfg.report_dir / "retrain_trigger"
    if not reports.is_dir():
        return None
    candidates = sorted(reports.glob("*.json"))
    return candidates[-1] if candidates else None


def trigger_version_tags(cfg: Config) -> dict[str, str]:
    """Registry tags recording *why* a version was retrained.

    Deliberately best-effort and total: an unreadable or half-written trigger
    report yields no tags rather than raising, because losing the audit trail
    is not a reason to block a promotion that already passed 14 gates.
    """
    path = trigger_report_path(cfg)
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    # Only a firing report says anything about *why* this version exists. A
    # clean night's report (`reason: none`) is a non-trigger: tagging the
    # version with it would assert a drift-motivated promotion that did not
    # happen, which is worse than no tag at all. Absent reports take the same
    # path, so an untagged promotion means exactly one thing.
    if not payload.get("triggered"):
        return {}
    tags: dict[str, str] = {}
    reason = payload.get("reason")
    if reason:
        tags["trigger_reason"] = str(reason)
    when = payload.get("date")
    if when:
        tags["trigger_date"] = str(when)
    worst = (payload.get("psi") or {}).get("worst") or {}
    if worst.get("drift_score") is not None:
        tags["trigger_psi"] = str(worst["drift_score"])
    if worst.get("column"):
        tags["trigger_psi_column"] = str(worst["column"])
    return tags


def apply_trigger_tags(
    client: RegistryClient, version: str, tags: dict[str, str]
) -> list[str]:
    """Stamp the trigger provenance onto a registered version.

    Returns the keys that stuck. Each tag is attempted independently: a single
    rejection must not abort the rest, and the caller records what landed.
    """
    applied: list[str] = []
    for key, value in tags.items():
        try:
            client.set_model_version_tag(_MODEL, version, key, value)
            applied.append(key)
        except Exception as error:  # noqa: BLE001 - best-effort provenance
            logger.warning(f"Could not set {key} on version {version}: {error}")
    return applied


def report_path(cfg: Config, version: str) -> Path:
    return cfg.report_dir / f"promotion_{version}.json"


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )


def format_table(gates: list[GateResult]) -> str:
    header = f"{'gate':<34} {'verdict':<8} {'observed':>12} {'threshold':>12}"
    lines = [header, "-" * len(header)]
    for gate in gates:
        observed = "-" if gate.observed is None else str(gate.observed)
        threshold = "-" if gate.threshold is None else str(gate.threshold)
        lines.append(
            f"{gate.name:<34} {'PASS' if gate.passed else 'FAIL':<8} "
            f"{observed:>12} {threshold:>12}"
        )
    return "\n".join(lines)


# --- the flow ----------------------------------------------------------------


def build_payload(
    cfg: Config,
    version: str,
    candidate: dict,
    production: dict,
    baseline: dict,
    gates: list[GateResult],
    promoted: bool,
    previous_production: str | None,
    dry_run: bool = False,
    trigger_tags: list[str] | None = None,
) -> dict[str, Any]:
    failed = [gate.name for gate in gates if not gate.passed]
    if promoted:
        decision = "promoted"
    elif failed:
        decision = "rejected"
    else:
        # Every gate passed but nothing was flipped -- a --skip-registry or
        # --dry-run rehearsal. Calling that "rejected" would be a lie about
        # what the measurements said.
        decision = "passed_not_promoted"
    if decision == "passed_not_promoted":
        # Distinguish "measured, nobody could flip it" from "measured, and the
        # Promotion was deliberately withheld", which are very different things
        # to find in a report attached to an approval.
        blocked_reason = "dry_run" if dry_run else "no_registry_client"
    else:
        blocked_reason = None
    return {
        "tool": _TOOL_TAG,
        "registered_model": _MODEL,
        "candidate_version": version,
        "previous_production_version": previous_production,
        "promoted": promoted,
        "decision": decision,
        "promotion_blocked_reason": blocked_reason,
        # Which Phase 6 step 3 trigger tags actually landed on the version. Empty
        # is normal (no trigger report, or a dry run) and is why this records
        # what stuck rather than what was attempted.
        "trigger_tags_applied": trigger_tags or [],
        "dry_run": dry_run,
        "failed_gates": failed,
        "gates": [gate.as_dict() for gate in gates],
        "candidate": candidate,
        "production_baseline": production,
        "floor_baseline": {
            "source": str(cfg.floor_report),
            "f1_macro": baseline["f1_macro"],
            "accuracy": baseline["accuracy"],
            "per_class": {
                name: values.get("recall")
                for name, values in baseline.get("per_class", {}).items()
            },
        },
        "test_split": {
            "path": str(cfg.test_split),
            "md5": file_md5(cfg.test_split) if cfg.test_split.exists() else None,
            "source": "dvc repro output, pinned by dvc.lock",
            "used_for_training": False,
        },
        "thresholds": {
            "f1_tolerance": cfg.f1_tolerance,
            "floor_tolerance": cfg.floor_tolerance,
            "accuracy_tolerance": cfg.accuracy_tolerance,
            "recall_tolerance": cfg.recall_tolerance,
            "min_recall": cfg.min_recall,
            "latency_regression": cfg.latency_regression,
            "max_size_mb": cfg.max_size_mb,
            "min_label_agreement": cfg.min_label_agreement,
            "max_parity_diff": cfg.max_parity_diff,
        },
        "latency_noise": {
            "same_graph_noise_pct": production.get("latency", {}).get("noise_pct"),
            "control_gap_pct": production.get("latency", {}).get("control_gap_pct"),
            "series_p95_ms": {
                "candidate": candidate.get("latency", {}).get("p95_ms"),
                "production": production.get("latency", {}).get("p95_ms"),
                "control": production.get("latency", {}).get("control_p95_ms"),
            },
            "allowed_regression_pct": _round(cfg.latency_regression * 100.0, 2),
            "resolvable_on_this_host": (
                None
                if production.get("latency", {}).get("noise_pct") is None
                else production["latency"]["noise_pct"]
                <= cfg.latency_regression * 100.0
            ),
            "note": (
                "Spread across the three interleaved series (candidate, production, "
                "and a control that re-times production) versus the gap between the "
                "two production series alone. The spread is the bound used for the "
                "diagnosis. "
                "Above allowed_regression_pct the host cannot resolve the band, so "
                "a latency failure is a measurement problem rather than evidence "
                "about the candidate."
            ),
        },
        "caveat": (
            "Latency is measured on the machine that ran this gate and compared "
            "against the production graph measured in the same process; it is not "
            "comparable with reports/benchmark_table.csv, which was measured "
            "elsewhere."
        ),
    }


@dataclass(frozen=True)
class Decision:
    """What the gate concluded, and on what evidence."""

    promoted: bool
    gates: list[GateResult]
    payload: dict[str, Any]

    @property
    def failed(self) -> list[GateResult]:
        return [gate for gate in self.gates if not gate.passed]


def promote(
    cfg: Config,
    version: str,
    candidate_onnx: str,
    production_onnx: str,
    client: RegistryClient | None = None,
    dry_run: bool = False,
) -> Decision:
    """Register the candidate, gate it, and promote only if everything passed.

    The ``Production`` alias is the last thing touched, and only on a clean
    sweep, so a failure anywhere above leaves the registry serving what it was
    serving before. ``dry_run`` measures everything and sets ``Candidate`` but
    stops there, which is what the measuring half of the workflow runs.
    """
    previous_production = None
    if client is not None:
        try:
            previous_production = client.get_model_version_by_alias(
                _MODEL, _ALIAS_PRODUCTION
            )
        except Exception:  # noqa: BLE001 - see below
            # No Production alias yet means this is the first promotion, not a
            # reason to abort. There is also nothing to roll back to, which the
            # report records as a null previous_production_version.
            #
            # The catch is deliberately broad: the registry sits behind a
            # Protocol, and each implementation signals "this alias does not
            # exist" its own way (MlflowClient raises MlflowException, a stub
            # may raise anything). Only this one lookup is guarded, so a real
            # error later in the flow still surfaces.
            previous_production = None
        # The candidate is registered under its own alias, never directly as
        # Production, so a version is always inspectable before it can serve.
        client.set_registered_model_alias(_MODEL, _ALIAS_CANDIDATE, version)

    # Verified before anything is scored. The frozen-split gate also appears in
    # the report, but by the time evaluate_all_gates reaches it both graphs have
    # already been scored against a split the gate was about to reject -- twenty
    # minutes of CPU, and on a contended runner enough scheduling noise to
    # produce a latency verdict nobody should act on.
    #
    # Only a *mismatch* short-circuits. A split with no dvc.lock entry at all is
    # a known, tolerated configuration that still produces a truthful report
    # with a failing pin gate, and turning that into an exception would take
    # away the report that explains it.
    frozen = check_frozen_split(cfg)
    if frozen.observed is not None and not frozen.passed:
        raise PromotionError(
            f"the test split does not match {cfg.dvc_lock} "
            f"(observed md5 {frozen.observed}, locked {frozen.threshold}); "
            "re-run the gate against the frozen split"
        )

    frame = load_split(cfg)
    candidate_graph = load_graph(cfg, candidate_onnx, frame)
    production_graph = load_graph(cfg, production_onnx, frame)
    measure_latency_pair(
        candidate_graph, production_graph, frame["text"].tolist()[:64], cfg
    )
    candidate = candidate_graph.metrics
    production = production_graph.metrics
    baseline = json.loads(cfg.floor_report.read_text())

    gates = evaluate_all_gates(cfg, candidate, production, baseline)
    passed = all(gate.passed for gate in gates)
    promoted = False
    trigger_tags: list[str] = []
    if passed and client is not None and not dry_run:
        client.set_registered_model_alias(_MODEL, _ALIAS_PRODUCTION, version)
        promoted = True
        # Provenance for *why* this version exists, stamped only once the alias
        # has actually moved -- a rejected candidate is not part of the served
        # history, so tagging it would imply a promotion that did not happen.
        trigger_tags = apply_trigger_tags(client, version, trigger_version_tags(cfg))
    elif passed and dry_run:
        # The measuring job must not be able to promote. This is the whole
        # reason --dry-run is a separate flag from --skip-registry: the
        # workflow's gate job needs a real registry connection to resolve and
        # stage the candidate graph, so "no registry" cannot be how it is kept
        # from flipping Production before a human approves.
        print("dry run: Candidate alias set, Production alias not touched")

    payload = build_payload(
        cfg,
        version,
        candidate,
        production,
        baseline,
        gates,
        promoted,
        previous_production,
        dry_run=dry_run,
        trigger_tags=trigger_tags,
    )
    return Decision(promoted=promoted, gates=gates, payload=payload)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidate-version", required=True)
    parser.add_argument(
        "--candidate-onnx",
        default="models/onnx/model_int8.onnx",
        help="graph to gate (measured, not trusted)",
    )
    parser.add_argument(
        "--production-onnx",
        default="models/onnx/model_int8.onnx",
        help="the graph currently in Production, for the comparison",
    )
    parser.add_argument("--tokenizer-dir", default=DefaultPaths.BASELINE_MODEL.value)
    parser.add_argument("--floor-report", default=DefaultPaths.EVAL_BASELINE.value)
    parser.add_argument("--parity-report", default="reports/onnx_int8_parity.json")
    parser.add_argument("--test-split", default=DefaultPaths.TEST_SPLIT.value)
    parser.add_argument("--report-dir", default="reports")
    # Kept as a flag rather than hard-coded so the lock the split is checked
    # against is always explicit at the call site.
    parser.add_argument("--dvc-lock", default="dvc.lock")
    parser.add_argument("--max-size-mb", type=float, default=None)
    parser.add_argument("--f1-tolerance", type=float, default=0.005)
    parser.add_argument("--floor-tolerance", type=float, default=0.01)
    parser.add_argument("--recall-tolerance", type=float, default=0.02)
    parser.add_argument("--latency-regression", type=float, default=0.10)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument(
        "--dialect-breakdown",
        action="store_true",
        help="also report per-dialect metrics (roughly doubles the runtime)",
    )
    parser.add_argument(
        "--skip-registry",
        action="store_true",
        help="evaluate and gate only, touching no alias (local rehearsal)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="evaluate and gate, then stop before moving Production",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = Config(
        test_split=Path(args.test_split),
        tokenizer_dir=args.tokenizer_dir,
        floor_report=Path(args.floor_report),
        parity_report=Path(args.parity_report),
        max_size_mb=args.max_size_mb,
        f1_tolerance=args.f1_tolerance,
        floor_tolerance=args.floor_tolerance,
        recall_tolerance=args.recall_tolerance,
        latency_regression=args.latency_regression,
        eval_limit=args.eval_limit,
        dialect_breakdown=args.dialect_breakdown,
        report_dir=Path(args.report_dir),
        dvc_lock=Path(args.dvc_lock),
    )

    client = None
    if not args.skip_registry:
        try:
            import mlflow

            from raay.config.env import load_environment, mlflow_tracking_uri

            load_environment()
            if mlflow_tracking_uri().startswith("file:"):
                import os

                os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
            client = mlflow.tracking.MlflowClient()
        except Exception as exc:  # pragma: no cover - environment dependent
            raise PromotionError(f"could not reach the MLflow registry: {exc}") from exc

    try:
        decision = promote(
            cfg,
            args.candidate_version,
            args.candidate_onnx,
            args.production_onnx,
            client=client,
            dry_run=args.dry_run,
        )
    except PromotionError as exc:
        print(f"promotion gate could not run: {exc}")
        return 2

    path = report_path(cfg, args.candidate_version)
    write_report(path, decision.payload)

    print(format_table(decision.gates))
    print()
    if decision.promoted:
        verdict = "PROMOTED"
    elif decision.payload["decision"] == "passed_not_promoted":
        verdict = "PASSED (not promoted)"
    else:
        verdict = "REJECTED"
    print(f"candidate v{args.candidate_version}: {verdict}")
    print(f"report: {path}")
    if decision.failed:
        for gate in decision.failed:
            print(
                f"  FAIL {gate.name}: observed={gate.observed} "
                f"threshold={gate.threshold} ({gate.detail})"
            )
        print("the Production alias was not touched")
    return 0 if not decision.failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
