"""Retrain trigger: token resolution, GitHub dispatch, and the CLI main()."""

from __future__ import annotations

import io
import json
import urllib.error
from typing import Self

import pytest
from retrain_helpers import breach_input_report

from raay.inference import retrain_cli
from raay.inference.retrain_trigger import dispatch_retrain, main, resolve_token


def test_resolve_token_reads_file(tmp_path):
    path = tmp_path / "tok"
    path.write_text("ghp_secret\n")
    assert resolve_token(str(path)) == "ghp_secret"


def test_resolve_token_absent_returns_none(monkeypatch):
    monkeypatch.delenv("RAAY_GITHUB_DISPATCH_TOKEN", raising=False)
    monkeypatch.delenv("RAAY_GITHUB_DISPATCH_TOKEN_FILE", raising=False)
    assert resolve_token() is None


def test_resolve_token_missing_file_is_none_not_an_error(tmp_path):
    missing = tmp_path / "nope"
    assert resolve_token(str(missing)) is None


class _FakeResponse:
    def __init__(self, status: int) -> None:
        self.status = status

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def test_dispatch_posts_the_event_with_a_bearer_header():
    captured: dict = {}

    def fake_opener(request, timeout=None):
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["headers"] = {k.lower(): v for k, v in request.header_items()}
        captured["body"] = json.loads(request.data)
        return _FakeResponse(204)

    result = dispatch_retrain(
        {"reason": "psi_breach", "trigger_date": "2026-10-01"},
        "github_pat_secret",
        repository="ahmeddiab1234/Raay",
        opener=fake_opener,
    )

    assert result["ok"] is True
    assert result["status"] == 204
    assert captured["method"] == "POST"
    assert captured["url"] == (
        "https://api.github.com/repos/ahmeddiab1234/Raay/dispatches"
    )
    assert captured["headers"]["authorization"] == "Bearer github_pat_secret"
    assert captured["body"]["event_type"] == "retrain"
    assert captured["body"]["client_payload"]["reason"] == "psi_breach"


def test_dispatch_reports_http_error_body_and_does_not_raise():
    def fake_opener(request, timeout=None):
        raise urllib.error.HTTPError(
            request.full_url,
            401,
            "Unauthorized",
            {},
            io.BytesIO(b'{"message":"Bad credentials"}'),
        )

    result = dispatch_retrain({}, "bad", opener=fake_opener)

    assert result["attempted"] is True
    assert result["ok"] is False
    assert result["status"] == 401
    assert "Bad credentials" in result["error"]


def test_dispatch_reports_network_error_and_does_not_raise():
    def fake_opener(request, timeout=None):
        raise urllib.error.URLError("name resolution failed")

    result = dispatch_retrain({}, "tok", opener=fake_opener)

    assert result["ok"] is False
    assert "name resolution failed" in result["error"]


def test_main_skips_dispatch_without_token(tmp_path, monkeypatch):
    monkeypatch.delenv("RAAY_GITHUB_DISPATCH_TOKEN", raising=False)
    monkeypatch.delenv("RAAY_GITHUB_DISPATCH_TOKEN_FILE", raising=False)
    inp = tmp_path / "in.json"
    pred = tmp_path / "pred.json"
    inp.write_text(json.dumps(breach_input_report()))
    pred.write_text(json.dumps({"triage": "world_changed"}))
    out = tmp_path / "trigger.json"

    code = main(
        [
            "--date",
            "2026-10-01",
            "--input-drift-report",
            str(inp),
            "--prediction-drift-report",
            str(pred),
            "--report-out",
            str(out),
            "--no-mlflow",
        ]
    )

    report = json.loads(out.read_text())
    assert report["triggered"] is True
    assert report["dispatch"]["attempted"] is False
    assert "no dispatch token" in report["dispatch"]["reason_not_attempted"]
    assert code == 0


def test_main_returns_nonzero_when_a_dispatch_attempt_fails(tmp_path, monkeypatch):
    tok = tmp_path / "tok"
    tok.write_text("github_pat_secret\n")
    inp = tmp_path / "in.json"
    pred = tmp_path / "pred.json"
    inp.write_text(json.dumps(breach_input_report()))
    pred.write_text(json.dumps({"triage": "world_changed"}))
    out = tmp_path / "trigger.json"
    monkeypatch.setattr(
        retrain_cli,
        "dispatch_retrain",
        lambda *a, **k: {"attempted": True, "ok": False, "error": "HTTP 403"},
    )

    code = main(
        [
            "--date",
            "2026-10-01",
            "--input-drift-report",
            str(inp),
            "--prediction-drift-report",
            str(pred),
            "--report-out",
            str(out),
            "--token-file",
            str(tok),
            "--no-mlflow",
        ]
    )

    assert code == 1
    assert json.loads(out.read_text())["dispatch"]["ok"] is False


def test_main_no_dispatch_flag_never_reads_the_token(tmp_path, monkeypatch):
    inp = tmp_path / "in.json"
    pred = tmp_path / "pred.json"
    inp.write_text(json.dumps(breach_input_report()))
    pred.write_text(json.dumps({"triage": "world_changed"}))
    out = tmp_path / "trigger.json"
    monkeypatch.setattr(
        retrain_cli,
        "resolve_token",
        lambda *a, **k: pytest.fail("token must not be read"),
    )

    code = main(
        [
            "--date",
            "2026-10-01",
            "--input-drift-report",
            str(inp),
            "--prediction-drift-report",
            str(pred),
            "--report-out",
            str(out),
            "--no-dispatch",
            "--no-mlflow",
        ]
    )

    assert code == 0
