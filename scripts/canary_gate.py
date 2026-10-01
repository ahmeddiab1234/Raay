"""The online stage gate.

Three outcomes, and the difference matters: a threshold failure is exit 1, a
hold-timeout is exit 1 (a *wait*, not a failure), and missing data is exit 2
INCONCLUSIVE. Missing data must never pass -- an empty canary with no metrics
would otherwise advance on the strength of not having produced evidence against
itself.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass
class GateOutcome:
    passed: bool
    inconclusive: bool
    reason: str
    observed: dict[str, float | int | None]


def evaluate_gate(
    phase: str,
    *,
    query: Callable[[str], float | None],
    entered_at: str | None = None,
    window: str = "5m",
    min_requests: int = 500,
    hold_seconds: int = 600,
    agreement_min: float = 0.99,
    error_rate_max: float = 0.005,
    p95_ratio_max: float = 1.10,
    now: Callable[[], datetime] | None = None,
) -> GateOutcome:
    """Run the stage gates over the accumulated window.

    Semantics (all over the trailing ``window``):

    * candidate must have seen a real request stream (``--min-requests``) and
      the current stage must have been held for ``--hold-seconds`` -- a wait,
      not a fail-if-data-missing, so a short window exits 1 (operator waits).
    * candidate error rate <= ``--error-rate-max``.
    * candidate p95 <= ``--p95-ratio-max`` x stable p95 (same box, so it is a
      fair A/B).
    * shadow stage only: label agreement >= ``--agreement-min``.

    ``None`` from any query (missing series, Prometheus down, zero traffic in a
    window) is INCONCLUSIVE, never a pass: an empty canary must not advance.
    """
    observed: dict[str, float | int | None] = {}

    def _q(name: str, promql: str) -> float | None:
        value = query(promql)
        observed[name] = value
        return value

    def inconclusive(reason: str) -> GateOutcome:
        return GateOutcome(
            passed=False, inconclusive=True, reason=reason, observed=observed
        )

    def fail(reason: str) -> GateOutcome:
        return GateOutcome(
            passed=False, inconclusive=False, reason=reason, observed=observed
        )

    requests_c = _q(
        "candidate_requests",
        f'sum(increase(raay_ingest_total{{worker="candidate"}}[{window}]))',
    )
    errors_c = _q(
        "candidate_errors",
        f'sum(increase(raay_errors_total{{worker="candidate"}}[{window}]))',
    )
    p95_c = _q(
        "candidate_p95_ms",
        f'histogram_quantile(0.95, sum(rate(raay_latency_seconds_bucket{{worker="candidate"}}[{window}])) by (le))',
    )
    p95_s = _q(
        "stable_p95_ms",
        f'histogram_quantile(0.95, sum(rate(raay_latency_seconds_bucket{{worker="stable"}}[{window}])) by (le))',
    )

    if requests_c is None or requests_c <= 0:
        return inconclusive(
            "no candidate requests in the window (metric missing or scrape broken)"
        )
    error_rate = (errors_c or 0.0) / requests_c
    if error_rate > error_rate_max:
        return fail(f"candidate error rate {error_rate:.4f} > max {error_rate_max}")

    if p95_c is None or p95_s is None or p95_c <= 0 or p95_s <= 0:
        return inconclusive("latency percentile missing (no samples in window)")

    if entered_at is not None:
        held_for = (
            now() if now is not None else datetime.now(UTC)
        ) - datetime.fromisoformat(entered_at)
        held_s = held_for.total_seconds()
        if held_s < hold_seconds:
            return fail(
                f"stage held only {held_s:.0f}s < {hold_seconds}s",
            )

    if requests_c < min_requests:
        return fail(
            f"candidate saw {requests_c:.0f} requests < --min-requests {min_requests}"
        )

    ratio = p95_c / p95_s
    if ratio > p95_ratio_max:
        return fail(
            f"candidate p95 {p95_c:.1f}ms is {ratio:.2f}x stable p95 {p95_s:.1f}ms "
            f"(> {p95_ratio_max}x)"
        )

    if phase == "shadow":
        agree = _q(
            "agreement_agree",
            f'sum(increase(raay_agreement_total{{status="agree"}}[{window}]))',
        )
        disagree = _q(
            "agreement_disagree",
            f'sum(increase(raay_agreement_total{{status="disagree"}}[{window}]))',
        )
        total = (agree or 0.0) + (disagree or 0.0)
        if total <= 0:
            return inconclusive("no paired shadow traffic in the window")
        rate = (agree or 0.0) / total
        if rate < agreement_min:
            return fail(
                f"shadow agreement {rate:.4f} < --agreement-min {agreement_min}"
            )

    return GateOutcome(
        passed=True,
        inconclusive=False,
        reason="all gates passed",
        observed=observed,
    )
