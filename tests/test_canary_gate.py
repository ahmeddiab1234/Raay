"""Hermetic tests for the gate math and PromQL parsing.

Covers:
* the gate math itself: error rate, p95 ratio, agreement, hold, empty-series
  INCONCLUSIVE handling.
"""

from __future__ import annotations

from canary_helpers import _good_query, _past

# ----------------------------------------------------------------- gate math


def test_gate_passes_shadow_stage(canary):
    outcome = canary.evaluate_gate(
        "shadow",
        query=_good_query(),
        entered_at=_past(),
        hold_seconds=600,
    )
    assert outcome.passed
    assert not outcome.inconclusive
    assert outcome.observed["agreement_agree"] == 595.0


def test_gate_skips_agreement_outside_shadow(canary):
    outcome = canary.evaluate_gate(
        "canary-5",
        query=_good_query(agree=None, disagree=None),
        entered_at=_past(),
        hold_seconds=600,
    )
    # Weighted canaries have no mirror, so agreement is undefined by design and
    # NOT an inconclusive failure.
    assert outcome.passed


def test_gate_fails_on_error_rate(canary):
    outcome = canary.evaluate_gate(
        "canary-5",
        query=_good_query(requests=600),
        entered_at=_past(),
        hold_seconds=600,
    )
    assert outcome.passed  # 0 errors

    def _q(promql):
        if "raay_errors_total" in promql:
            return 30.0
        return _good_query()(promql)

    outcome_bad = canary.evaluate_gate(
        "canary-5",
        query=_q,
        entered_at=_past(),
        hold_seconds=600,
    )
    assert not outcome_bad.passed
    assert not outcome_bad.inconclusive
    assert "error rate" in outcome_bad.reason


def test_gate_fails_on_p95_ratio(canary):
    def _q(promql):
        if "raay_latency_seconds_bucket" in promql:
            return 220.0 if '"candidate"' in promql else 40.0
        return _good_query()(promql)

    outcome = canary.evaluate_gate(
        "canary-5",
        query=_q,
        entered_at=_past(),
        hold_seconds=600,
    )
    assert not outcome.passed
    assert "p95" in outcome.reason


def test_gate_fails_on_low_agreement(canary):
    outcome = canary.evaluate_gate(
        "shadow",
        query=_good_query(agree=300.0, disagree=300.0),
        entered_at=_past(),
        hold_seconds=600,
    )
    assert not outcome.passed
    assert "agreement" in outcome.reason


def test_gate_inconclusive_with_zero_requests(canary):
    outcome = canary.evaluate_gate(
        "canary-5",
        query=_good_query(requests=0),
        entered_at=_past(),
        hold_seconds=600,
    )
    assert outcome.inconclusive
    assert not outcome.passed


def test_gate_inconclusive_with_missing_p95(canary):
    outcome = canary.evaluate_gate(
        "canary-5",
        query=_good_query(p95_c=None),
        entered_at=_past(),
        hold_seconds=600,
    )
    assert outcome.inconclusive
    assert not outcome.passed


def test_gate_inconclusive_with_zero_paired_traffic(canary):
    outcome = canary.evaluate_gate(
        "shadow",
        query=_good_query(agree=None, disagree=None),
        entered_at=_past(),
        hold_seconds=600,
    )
    assert outcome.inconclusive


def test_gate_fails_below_min_requests(canary):
    outcome = canary.evaluate_gate(
        "canary-5",
        query=_good_query(requests=10),
        entered_at=_past(),
        hold_seconds=600,
        min_requests=500,
    )
    assert not outcome.passed
    assert not outcome.inconclusive
    assert "--min-requests" in outcome.reason


# ------------------------------------------------------------ promql parsing


def test_promql_query_parses_instant_vector_value(canary, monkeypatch):
    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b'{"status":"success","data":{"result":[{"metric":{},"value":[1,"42.5"]}]}}'

    monkeypatch.setattr(canary.urllib.request, "urlopen", lambda url, timeout: _Resp())
    assert canary._promql_query("http://127.0.0.1:9090", "up") == 42.5


def test_promql_query_missing_series_returns_none(canary, monkeypatch):
    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b'{"status":"success","data":{"result":[]}}'

    monkeypatch.setattr(canary.urllib.request, "urlopen", lambda url, timeout: _Resp())
    assert canary._promql_query("http://127.0.0.1:9090", "up") is None


def test_promql_query_network_error_returns_none(canary, monkeypatch):
    def boom(url, timeout):
        raise OSError("refused")

    monkeypatch.setattr(canary.urllib.request, "urlopen", boom)
    assert canary._promql_query("http://127.0.0.1:9090", "up") is None
