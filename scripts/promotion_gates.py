"""Gates that compare two metrics blocks, plus the gate list itself.

Accuracy is in here as a deliberate redundancy, not by accident: predicting
``positive`` for everything scores 0.85 on this split, the same as the real
model, with macro F1 at 0.42. Accuracy alone cannot tell those apart, which is
why per-class recall and an absolute Neutral floor are gated too.
"""

from __future__ import annotations

from typing import Any

from promotion_split import check_frozen_split
from promotion_types import _WATCHED_CLASSES, Config, GateResult, _round


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
    candidate: dict[str, Any],
    production: dict[str, Any],
    baseline: dict[str, Any],
) -> list[GateResult]:
    """All fourteen gates, in the order they appear in the report."""
    from promotion_gates_host import gate_latency, gate_parity, gate_size

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
