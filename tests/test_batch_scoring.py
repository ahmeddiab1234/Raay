"""Unit tests for the batch scoring stage: seeded sampling, scoring, reference.

Hermetic by design (AGENTS.md rule: unit tests only): no ONNX graph, no MLflow,
no Airflow, no encoder weights. The duck-typed fakes in ``conftest.py`` stand in
for the INT8 session.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from conftest import FakeScorer
from drift_helpers import pool_csv

from raay.inference.batch_reference import init_reference
from raay.inference.batch_scoring import _date_seed, make_input, score_input


class TestDateSeed:
    def test_date_seed_is_deterministic(self):
        assert _date_seed("2026-09-24") == _date_seed("2026-09-24")

    def test_date_seed_is_uint32(self):
        assert 0 <= _date_seed("2026-09-24") <= 2**32 - 1


class TestMakeInput:
    def test_same_date_same_sample(self, tmp_path):
        pool = pool_csv(tmp_path)
        a = make_input(pool, "2026-09-24", 10, str(tmp_path / "a.csv"))
        b = make_input(pool, "2026-09-24", 10, str(tmp_path / "b.csv"))
        assert list(a["text"]) == list(b["text"])

    def test_different_date_different_sample(self, tmp_path):
        pool = pool_csv(tmp_path)
        a = make_input(pool, "2026-09-24", 10, str(tmp_path / "a.csv"))
        c = make_input(pool, "2026-09-25", 10, str(tmp_path / "c.csv"))
        assert list(a["text"]) != list(c["text"])

    def test_adds_day_column(self, tmp_path):
        pool = pool_csv(tmp_path)
        out = str(tmp_path / "in.csv")
        sample = make_input(pool, "2026-09-24", 10, out)
        assert next(iter(sample.columns)) == "day"
        assert (sample["day"] == "2026-09-24").all()

    def test_raises_when_samples_exceed_pool(self, tmp_path):
        pool = pool_csv(tmp_path, n=5)
        try:
            make_input(pool, "2026-09-24", 10, str(tmp_path / "x.csv"))
        except ValueError as exc:
            assert "need at least 10" in str(exc)
        else:
            raise AssertionError("expected ValueError")


class TestScoreInput:
    def test_output_schema_and_argmax(self, tmp_path):
        inp = str(tmp_path / "in.csv")
        pd.DataFrame({"text": ["جيد جدا", "أعجبني", "سيء"]}).to_csv(inp, index=False)
        out = str(tmp_path / "out.csv")
        stats = score_input(inp, out, FakeScorer(), min_samples=1)
        df = pd.read_csv(out)
        assert {
            "text",
            "positive",
            "negative",
            "neutral",
            "predicted_label",
            "predicted_score",
        } <= set(df.columns)
        assert (df["predicted_label"] == "positive").all()
        assert np.allclose(df["predicted_score"], 0.8)
        assert stats["n_reviews"] == 3

    def test_min_samples_floor_enforced(self, tmp_path):
        inp = str(tmp_path / "in.csv")
        pd.DataFrame({"text": ["x"]}).to_csv(inp, index=False)
        try:
            score_input(inp, str(tmp_path / "o.csv"), FakeScorer(), min_samples=10)
        except ValueError as exc:
            assert "at least 10" in str(exc)
        else:
            raise AssertionError("expected ValueError")

    def test_missing_text_column_rejected(self, tmp_path):
        inp = str(tmp_path / "in.csv")
        pd.DataFrame({"label": [1, 2]}).to_csv(inp, index=False)
        try:
            score_input(inp, str(tmp_path / "o.csv"), FakeScorer(), min_samples=1)
        except ValueError as exc:
            assert "'text'" in str(exc)
        else:
            raise AssertionError("expected ValueError")


class TestInitReference:
    def test_reference_is_scored(self, tmp_path):
        pool = pool_csv(tmp_path, n=30)
        ref = str(tmp_path / "reference.csv")
        out = init_reference(pool, 20, ref, FakeScorer())
        assert len(out) == 20
        assert {"text", "predicted_label", "positive"} <= set(out.columns)
