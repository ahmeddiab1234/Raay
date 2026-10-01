"""The agent's Prometheus metric set, on a private registry.

A private ``CollectorRegistry`` (not the process-global default one) is
deliberate: the module-level ``app`` below is constructed at import time, and
registering these collectors in the default registry would raise
``Duplicated timeseries`` the second anything else in the process touches
prometheus_client.
"""

from __future__ import annotations

from dataclasses import dataclass

from prometheus_client import CollectorRegistry, Counter, Histogram


@dataclass
class AgentMetrics:
    """Counters + histogram the agent updates, bound to one registry."""

    registry: CollectorRegistry
    ingest: Counter
    errors: Counter
    latency: Histogram
    agreement: Counter
    pair_gap: Counter


def build_metrics() -> AgentMetrics:
    registry = CollectorRegistry()
    return AgentMetrics(
        registry=registry,
        ingest=Counter(
            "raay_ingest_total",
            "Prediction events received, per worker.",
            ("worker",),
            registry=registry,
        ),
        errors=Counter(
            "raay_errors_total",
            "Prediction events that errored, per worker.",
            ("worker",),
            registry=registry,
        ),
        latency=Histogram(
            "raay_latency_seconds",
            "Worker-side /predict latency.",
            ("worker",),
            registry=registry,
        ),
        agreement=Counter(
            "raay_agreement_total",
            "Paired shadow requests by agreement status.",
            ("status",),
            registry=registry,
        ),
        pair_gap=Counter(
            "raay_pair_gap_total",
            "Shadowed request ids that expired with one side missing, by the "
            "worker that *did not* arrive.",
            ("worker",),
            registry=registry,
        ),
    )
