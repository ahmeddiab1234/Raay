"""Gates that depend on something other than the two metrics blocks.

Latency depends on the host, size on the file on disk, and parity on a committed
report from the export step. Keeping them apart from the pure metric comparisons
makes the one place where the host can be the answer obvious.
"""

from __future__ import annotations

from promotion_graph import _parity_section, read_parity
from promotion_types import Config, GateResult, _round


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
