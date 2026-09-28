"""Hermetic tests for the canary/shadow comparison agent.

The agent only does three things -- pair events by request id, count
agreement/gaps, and render Prometheus text -- so these tests cover all of it
with plain objects and a fake clock. No network, no nginx, no model.
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from raay.serving.canary_agent import CanaryAgent, PredictionEvent


class _Clock:
    def __init__(self, start: float = 1000.0) -> None:
        self._t = start

    def advance(self, seconds: float) -> None:
        self._t += seconds

    def __call__(self) -> float:
        return self._t


def _ev(
    request_id: str,
    worker: str,
    label: str = "positive",
    *,
    shadow: bool = False,
    status: int = 200,
    error: str | None = None,
    latency_ms: float | None = 10.0,
    version: str = "v1",
    texts: list[str] | None = None,
    labels: list[str] | None = None,
) -> PredictionEvent:
    preds = [{"label": lab, "score": 0.9} for lab in (labels or [label])]
    return PredictionEvent(
        request_id=request_id,
        worker=worker,
        model_version=version,
        status=status,
        error=error,
        latency_ms=latency_ms,
        shadow=shadow,
        predictions=preds,
        texts=texts or [],
    )


def _agent(
    ttl: float = 30.0, max_pairs: int = 100, clock: _Clock | None = None
) -> tuple[CanaryAgent, _Clock]:
    injected = clock or _Clock()
    return CanaryAgent(pair_ttl_s=ttl, max_pairs=max_pairs, clock=injected), injected


def _val(agent: CanaryAgent, name: str, **labels: str) -> float:
    """Read one sample value straight from the rendered Prometheus text."""

    text = agent.render_metrics().decode()
    q = '"'
    if labels:
        needle = f"{name}{{{','.join(f'{k}={q}{v}{q}' for k, v in labels.items())}}} "
    else:
        needle = f"{name} "
    for line in text.splitlines():
        if line.startswith(needle):
            return float(line.split()[-1])
    # An uninstantiated label series is absent from the exposition, which is
    # exactly zero observations -- match Prometheus missing-series semantics.
    return 0.0


# ---------------------------------------------------------------- ingestion


def test_ingest_counts_per_worker():
    agent, _ = _agent()
    agent.ingest(_ev("a", "stable"))
    agent.ingest(_ev("b", "candidate"))
    assert _val(agent, "raay_ingest_total", worker="stable") == 1
    assert _val(agent, "raay_ingest_total", worker="candidate") == 1


def test_ingest_counts_errors_per_worker():
    agent, _ = _agent()
    agent.ingest(_ev("a", "stable", status=500, error="boom"))
    agent.ingest(_ev("b", "candidate", status=422))
    assert _val(agent, "raay_errors_total", worker="stable") == 1
    assert _val(agent, "raay_errors_total", worker="candidate") == 1


def test_ingest_observes_latency_histogram():
    agent, _ = _agent()
    agent.ingest(_ev("a", "stable", latency_ms=50.0))
    agent.ingest(_ev("b", "stable", latency_ms=100.0, label="negative"))
    # Histogram summary lines for the worker label equal the observations.
    assert _val(agent, "raay_latency_seconds_sum", worker="stable") == pytest.approx(
        0.15
    )
    assert _val(agent, "raay_latency_seconds_count", worker="stable") == 2


def test_ingest_without_request_id_never_pairs():
    agent, _ = _agent()
    agent.ingest(PredictionEvent(worker="stable", predictions=[{"label": "positive"}]))
    agent.ingest(
        PredictionEvent(worker="candidate", predictions=[{"label": "positive"}])
    )
    assert _val(agent, "raay_agreement_total", status="agree") == 0
    assert _val(agent, "raay_pair_gap_total", worker="candidate") == 0


# ------------------------------------------------- pairing key (shadow mirror)


def test_shadowed_events_pair_by_text_without_request_id():
    # nginx mirror subrequests get no X-Request-ID, so content is the key.
    agent, _ = _agent()
    texts = ["مراجعة", "مثيرة"]
    agent.ingest(_ev("", "stable", label="positive", shadow=True, texts=texts))
    agent.ingest(_ev("", "candidate", label="positive", shadow=True, texts=texts))
    assert _val(agent, "raay_agreement_total", status="agree") == 1
    assert _val(agent, "raay_pair_gap_total", worker="candidate") == 0


def test_text_paired_agreement_compares_full_label_sequence():
    agent, _ = _agent()
    texts = ["أ", "ب"]
    agent.ingest(
        _ev("", "stable", shadow=True, texts=texts, labels=["positive", "negative"])
    )
    agent.ingest(
        _ev("", "candidate", shadow=True, texts=texts, labels=["positive", "neutral"])
    )
    assert _val(agent, "raay_agreement_total", status="disagree") == 1
    assert _val(agent, "raay_agreement_total", status="agree") == 0


def test_shadowed_text_pair_gap_counts_when_stable_never_arrives():
    agent, clock = _agent(ttl=5.0)
    agent.ingest(_ev("", "candidate", shadow=True, texts=["منفرد"]))
    clock.advance(6.0)
    agent.ingest(_ev("other", "stable", shadow=False))
    assert _val(agent, "raay_pair_gap_total", worker="stable") == 1


def test_request_id_key_wins_over_texts():
    agent, _ = _agent()
    agent.ingest(_ev("shared", "stable", label="positive", shadow=True, texts=["x"]))
    agent.ingest(_ev("shared", "candidate", label="positive", shadow=True, texts=["y"]))
    assert _val(agent, "raay_agreement_total", status="agree") == 1


# ---------------------------------------------------------------- agreement


def test_pair_with_matching_labels_counts_agreement():
    agent, _ = _agent()
    agent.ingest(_ev("r1", "stable", label="positive", shadow=True))
    agent.ingest(_ev("r1", "candidate", label="positive", shadow=True))
    assert _val(agent, "raay_agreement_total", status="agree") == 1
    assert _val(agent, "raay_agreement_total", status="disagree") == 0


def test_pair_with_differing_labels_counts_disagreement():
    agent, _ = _agent()
    agent.ingest(_ev("r1", "candidate", label="negative", shadow=True))
    agent.ingest(_ev("r1", "stable", label="positive", shadow=True))
    assert _val(agent, "raay_agreement_total", status="disagree") == 1
    assert _val(agent, "raay_agreement_total", status="agree") == 0


def test_pair_resolution_does_not_depend_on_arrival_order():
    agent, _ = _agent()
    agent.ingest(_ev("r1", "candidate", label="neutral", shadow=True))
    agent.ingest(_ev("r1", "stable", label="neutral", shadow=True))
    assert _val(agent, "raay_agreement_total", status="agree") == 1


def test_unshadowed_events_never_count_agreement():
    # Weighted canary: each request hits exactly one worker, never a pair.
    agent, _ = _agent()
    agent.ingest(_ev("r1", "stable", label="positive"))
    assert _val(agent, "raay_agreement_total", status="agree") == 0
    assert _val(agent, "raay_pair_gap_total", worker="candidate") == 0


def test_pair_is_closed_after_resolution():
    agent, _ = _agent()
    agent.ingest(_ev("r1", "stable", shadow=True))
    agent.ingest(_ev("r1", "candidate", shadow=True))
    # Same id again starts a fresh, unrelated pair.
    agent.ingest(_ev("r1", "stable", shadow=True))
    assert _val(agent, "raay_agreement_total", status="agree") == 1
    assert _val(agent, "raay_ingest_total", worker="stable") == 2


# ------------------------------------------------------------------- gaps


def test_expired_shadowed_pair_missing_candidate_counts_candidate_gap():
    agent, clock = _agent(ttl=5.0)
    agent.ingest(_ev("r1", "stable", shadow=True))
    clock.advance(6.0)
    agent.ingest(_ev("other", "stable", shadow=False))
    assert _val(agent, "raay_pair_gap_total", worker="candidate") == 1
    assert _val(agent, "raay_pair_gap_total", worker="stable") == 0


def test_expired_shadowed_pair_missing_stable_counts_stable_gap():
    agent, clock = _agent(ttl=5.0)
    agent.ingest(_ev("r1", "candidate", shadow=True))
    clock.advance(6.0)
    agent.ingest(_ev("other", "stable"))
    assert _val(agent, "raay_pair_gap_total", worker="stable") == 1


def test_expired_unshadowed_pair_never_counts_a_gap():
    agent, clock = _agent(ttl=5.0)
    agent.ingest(_ev("r1", "stable"))
    clock.advance(6.0)
    agent.ingest(_ev("other", "stable"))
    assert _val(agent, "raay_pair_gap_total", worker="candidate") == 0
    assert _val(agent, "raay_pair_gap_total", worker="stable") == 0


def test_resolved_pair_is_not_a_gap():
    agent, clock = _agent(ttl=5.0)
    agent.ingest(_ev("r1", "stable", shadow=True))
    agent.ingest(_ev("r1", "candidate", shadow=True))
    clock.advance(6.0)
    agent.ingest(_ev("other", "stable"))
    assert _val(agent, "raay_pair_gap_total", worker="candidate") == 0
    assert _val(agent, "raay_pair_gap_total", worker="stable") == 0


def test_overflow_eviction_does_not_count_gaps():
    agent, _ = _agent(ttl=1000.0, max_pairs=2)
    agent.ingest(_ev("r1", "stable", shadow=True))
    agent.ingest(_ev("r2", "stable", shadow=True))
    agent.ingest(_ev("r3", "stable", shadow=True))  # evicts r1
    assert len(agent._pairs) == 2
    assert _val(agent, "raay_pair_gap_total", worker="candidate") == 0


# -------------------------------------------------------------- prometheus


def test_render_metrics_exposes_all_series():
    agent, _ = _agent()
    agent.ingest(_ev("r1", "stable", shadow=True))
    agent.ingest(_ev("r1", "candidate", shadow=True))
    text = agent.render_metrics().decode()
    assert "raay_ingest_total" in text
    assert "raay_errors_total" in text
    assert "raay_latency_seconds" in text
    assert "raay_agreement_total" in text
    assert "raay_pair_gap_total" in text


def test_render_metrics_is_isolated_per_agent():
    a, _ = _agent()
    b, _ = _agent()
    a.ingest(_ev("r1", "stable"))
    assert 'worker="stable"' in a.render_metrics().decode()
    assert 'worker="stable"' not in b.render_metrics().decode()


# ------------------------------------------------------------- HTTP surface


def test_http_ingest_and_metrics_and_health():
    agent, _ = _agent()
    with TestClient(agent.asgi) as client:
        resp = client.post(
            "/ingest",
            json={
                "request_id": "r1",
                "worker": "stable",
                "shadow": True,
                "predictions": [{"label": "positive", "score": 0.9}],
            },
        )
        assert resp.status_code == 200
        assert client.get("/health").json() == {"status": "healthy"}
        metrics = client.get("/metrics").text
        assert 'worker="stable"' in metrics


def test_http_rejects_bad_json_and_bad_event():
    agent, _ = _agent()
    with TestClient(agent.asgi) as client:
        assert (
            client.post(
                "/ingest",
                content=b"{not json",
                headers={"Content-Type": "application/json"},
            ).status_code
            == 400
        )
        assert client.post("/ingest", json={"worker": 42}).status_code == 422
