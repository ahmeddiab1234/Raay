"""Hermetic tests for the customer-service feedback capture endpoint.

The service does four things -- validate, authenticate, hash for idempotency,
append -- so it is tested with plain objects, a fixed clock and a temp directory.
No network, no ONNX graph, no ``data/``.
"""

from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from raay.serving.feedback_service import (
    RAW_COLUMNS,
    FeedbackOverride,
    FeedbackService,
    FeedbackSink,
    resolve_token,
)

TOKEN = "s3cret-token-value"

#: Long enough to clear the default min_char_length of 10.
TEXT = "المنتج ممتاز لكن التوصيل كان متأخر جدا"


def _fixed_clock() -> datetime:
    return datetime(2026, 10, 1, 12, 30, tzinfo=UTC)


def _service(tmp_path: Path, **kwargs) -> FeedbackService:
    sink = FeedbackSink(base_dir=str(tmp_path / "raw"), clock=_fixed_clock)
    kwargs.setdefault("token", TOKEN)
    return FeedbackService(sink=sink, **kwargs)


def _payload(**overrides) -> dict:
    payload = {
        "text": TEXT,
        "model_label": "positive",
        "corrected_label": "negative",
        "agent_id": "agent.01",
        "captured_at": "2026-10-01T12:00:00+00:00",
        "model_score": 0.42,
    }
    payload.update(overrides)
    return payload


def _auth() -> dict[str, str]:
    return {"authorization": f"Bearer {TOKEN}"}


def _rows(tmp_path: Path) -> list[dict[str, str]]:
    """The captured rows, as the sink actually wrote them to disk."""
    with open(tmp_path / "raw" / "2026-10-01.csv", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# --------------------------------------------------------------------- model


def test_label_must_be_a_known_class():
    for label in ("positive", "negative", "neutral"):
        assert (
            FeedbackOverride(**_payload(corrected_label=label)).corrected_label == label
        )


def test_unknown_label_is_rejected():
    """Guards the label-encoding trap in docs/labeling_guidelines.md section 1.

    The doc states positive=2/negative=0/neutral=1; every graph in the repo uses
    positive=0/negative=1/neutral=2. Accepting an integer here would accept a
    value that silently means a different class on the serving side.
    """
    with pytest.raises(ValueError, match="label must be one of"):
        FeedbackOverride(**_payload(model_label="2"))


def test_agent_id_cannot_forge_a_csv_row():
    """A comma or newline in agent_id would write a second record into the sink."""
    for bad in ("agent,01", "agent\n01", "a" * 65, "agent 01"):
        with pytest.raises(ValueError, match="agent_id must match"):
            FeedbackOverride(**_payload(agent_id=bad))


def test_blank_text_is_rejected():
    with pytest.raises(ValueError, match="text must not be empty"):
        FeedbackOverride(**_payload(text="   "))


def test_is_correction_is_false_when_the_agent_agrees():
    assert FeedbackOverride(**_payload()).is_correction is True
    same = FeedbackOverride(**_payload(corrected_label="positive"))
    assert same.is_correction is False


def test_override_id_ignores_the_timestamp():
    """A retried POST must not become a second row."""
    first = FeedbackOverride(**_payload(captured_at="2026-10-01T12:00:00+00:00"))
    retry = FeedbackOverride(**_payload(captured_at="2026-10-01T12:00:07+00:00"))
    assert first.override_id() == retry.override_id()


def test_override_id_differs_per_agent_and_per_label():
    base = FeedbackOverride(**_payload())
    other_agent = FeedbackOverride(**_payload(agent_id="agent.02"))
    other_label = FeedbackOverride(**_payload(corrected_label="neutral"))
    ids = {base.override_id(), other_agent.override_id(), other_label.override_id()}
    assert len(ids) == 3


def test_distinct_agents_who_agree_get_distinct_ids():
    """Corroboration depends on two rows not colliding into one."""
    a = FeedbackOverride(**_payload(agent_id="agent.01"))
    b = FeedbackOverride(**_payload(agent_id="agent.02"))
    assert a.override_id() != b.override_id()


# ----------------------------------------------------------------- the sink


def test_sink_writes_one_header_and_appends(tmp_path: Path):
    sink = FeedbackSink(base_dir=str(tmp_path), clock=_fixed_clock)
    sink.append(FeedbackOverride(**_payload()), "aaa")
    sink.append(FeedbackOverride(**_payload(agent_id="agent.02")), "bbb")

    with open(sink.path_for("2026-10-01"), encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["override_id"] for row in rows] == ["aaa", "bbb"]
    assert list(rows[0]) == list(RAW_COLUMNS)


def test_sink_routes_rows_to_their_utc_day(tmp_path: Path):
    sink = FeedbackSink(base_dir=str(tmp_path), clock=_fixed_clock)
    override = FeedbackOverride(**_payload(captured_at="2026-10-02T23:30:00+00:00"))
    sink.append(override, "aaa")
    assert sink.path_for("2026-10-02").exists()
    assert not sink.path_for("2026-10-01").exists()


def test_sink_converts_a_naive_timestamp_to_utc(tmp_path: Path):
    sink = FeedbackSink(base_dir=str(tmp_path), clock=_fixed_clock)
    override = FeedbackOverride(**_payload(captured_at="2026-10-01T12:00:00"))
    sink.append(override, "aaa")
    with open(sink.path_for("2026-10-01"), encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["captured_at"].endswith("+00:00")


def test_sink_counts_rows_written(tmp_path: Path):
    sink = FeedbackSink(base_dir=str(tmp_path), clock=_fixed_clock)
    for i in range(3):
        sink.append(FeedbackOverride(**_payload(agent_id=f"agent.0{i}")), str(i))
    assert sink.rows_written == 3


def test_ids_for_is_empty_when_the_day_file_is_absent(tmp_path: Path):
    assert FeedbackSink(base_dir=str(tmp_path)).ids_for("2026-10-01") == set()


# -------------------------------------------------------------------- auth


def test_service_refuses_to_build_without_a_token(tmp_path: Path):
    """An unauthenticated write path into a training set is a poisoning vector."""
    with pytest.raises(RuntimeError, match="refuses|will not serve"):
        FeedbackService(sink=FeedbackSink(base_dir=str(tmp_path)))


def test_allow_anon_is_the_escape_hatch(tmp_path: Path):
    service = FeedbackService(
        sink=FeedbackSink(base_dir=str(tmp_path)), allow_anon=True
    )
    assert service.authorize(None) is True
    assert service.stats["auth_required"] is False


def test_authorize_rejects_missing_and_malformed_headers(tmp_path: Path):
    service = _service(tmp_path)
    assert service.authorize(None) is False
    assert service.authorize("") is False
    assert service.authorize(TOKEN) is False
    assert service.authorize(f"Basic {TOKEN}") is False
    assert service.authorize("Bearer ") is False


def test_authorize_accepts_the_right_token(tmp_path: Path):
    service = _service(tmp_path)
    assert service.authorize(f"Bearer {TOKEN}") is True
    assert service.authorize(f"bearer {TOKEN}") is True


def test_authorize_rejects_a_prefix_of_the_token(tmp_path: Path):
    """compare_digest is exact, not prefix-tolerant."""
    service = _service(tmp_path)
    assert service.authorize(f"Bearer {TOKEN[:-1]}") is False


def test_resolve_token_prefers_an_explicit_value(tmp_path: Path):
    assert resolve_token("inline", env={}) == "inline"


def test_resolve_token_reads_a_file(tmp_path: Path):
    secret = tmp_path / "tok"
    secret.write_text(f"  {TOKEN}\n")
    assert resolve_token(env={"RAAY_FEEDBACK_TOKEN_FILE": str(secret)}) == TOKEN


def test_resolve_token_falls_back_to_the_environment():
    assert resolve_token(env={"RAAY_FEEDBACK_TOKEN": TOKEN}) is TOKEN


def test_resolve_token_returns_none_when_unprovisioned():
    assert resolve_token(env={}) is None


# ---------------------------------------------------------------- endpoints


def test_post_feedback_persists_a_row(tmp_path: Path):
    service = _service(tmp_path)
    with TestClient(service.asgi) as client:
        resp = client.post("/feedback", json=_payload(), headers=_auth())
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["is_correction"] is True
    assert body["duplicate"] is False
    assert (tmp_path / "raw" / "2026-10-01.csv").exists()


def test_post_feedback_without_a_token_is_401_and_writes_nothing(tmp_path: Path):
    service = _service(tmp_path)
    with TestClient(service.asgi) as client:
        resp = client.post("/feedback", json=_payload())
    assert resp.status_code == 401
    assert not (tmp_path / "raw").exists() or not list((tmp_path / "raw").glob("*.csv"))
    assert service.rejected_unauthorized == 1


def test_post_feedback_with_a_wrong_token_is_401(tmp_path: Path):
    service = _service(tmp_path)
    with TestClient(service.asgi) as client:
        resp = client.post(
            "/feedback", json=_payload(), headers={"authorization": "Bearer no"}
        )
    assert resp.status_code == 401


def test_a_retry_of_the_same_override_is_idempotent(tmp_path: Path):
    service = _service(tmp_path)
    with TestClient(service.asgi) as client:
        first = client.post("/feedback", json=_payload(), headers=_auth())
        second = client.post(
            "/feedback",
            json=_payload(captured_at="2026-10-01T12:00:09+00:00"),
            headers=_auth(),
        )
    assert first.json()["override_id"] == second.json()["override_id"]
    assert second.json()["duplicate"] is True
    assert len(_rows(tmp_path)) == 1
    assert service.duplicate_hits == 1


def test_a_confirmation_is_accepted_not_rejected(tmp_path: Path):
    """A confirmation is the denominator of production_error_rate.

    If this 422'd, the error rate would only ever see disputes and could not be
    computed at all.
    """
    service = _service(tmp_path)
    with TestClient(service.asgi) as client:
        resp = client.post(
            "/feedback",
            json=_payload(corrected_label="positive"),
            headers=_auth(),
        )
    assert resp.status_code == 200
    assert resp.json()["is_correction"] is False
    assert _rows(tmp_path)[0]["corrected_label"] == "positive"


def test_an_invalid_override_is_422(tmp_path: Path):
    service = _service(tmp_path)
    with TestClient(service.asgi) as client:
        resp = client.post(
            "/feedback", json=_payload(corrected_label="excellent"), headers=_auth()
        )
    assert resp.status_code == 422
    assert resp.json()["error"] == "invalid override"


def test_malformed_json_is_400_not_422(tmp_path: Path):
    service = _service(tmp_path)
    with TestClient(service.asgi) as client:
        resp = client.post(
            "/feedback",
            content=b"{not json",
            headers={**_auth(), "content-type": "application/json"},
        )
    assert resp.status_code == 400


def test_health_needs_no_token_and_no_graph(tmp_path: Path):
    service = _service(tmp_path)
    with TestClient(service.asgi) as client:
        assert client.get("/health").json() == {"status": "healthy"}


def test_stats_reports_the_capture_counters(tmp_path: Path):
    service = _service(tmp_path)
    with TestClient(service.asgi) as client:
        client.post("/feedback", json=_payload(), headers=_auth())
        client.post("/feedback", json=_payload(), headers=_auth())
        client.post("/feedback", json=_payload())
        stats = client.get("/stats").json()
    assert stats["rows_written"] == 1
    assert stats["duplicate_hits"] == 1
    assert stats["rejected_unauthorized"] == 1
    assert stats["auth_required"] is True


def test_model_score_and_provenance_are_persisted(tmp_path: Path):
    service = _service(tmp_path)
    with TestClient(service.asgi) as client:
        client.post(
            "/feedback",
            json=_payload(model_score=0.77, model_version="int8-687d587004c6"),
            headers=_auth(),
        )
    row = _rows(tmp_path)[0]
    assert row["model_score"] == "0.77"
    assert row["model_version"] == "int8-687d587004c6"
    assert row["guideline_version"] == "v1.0"
