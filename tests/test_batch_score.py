"""Unit tests for the nightly batch re-scoring job + Evidently PSI drift.

Hermetic by design (AGENTS.md rule: unit tests only): no ONNX graph, no
MLflow, no Airflow. A duck-typed fake scorer stands in for the INT8 session
and drift checks run on tiny synthetic DataFrames.
"""

import json
from typing import ClassVar

import numpy as np
import pandas as pd

from raay.inference.batch_score import (
    _date_seed,
    drift_check,
    init_reference,
    make_input,
    score_input,
)


class FakeScorer:
    """Matches the ``Scorer`` interface (label_columns + score) without ORT."""

    label_columns: ClassVar[list[str]] = ["positive", "negative", "neutral"]
    batch_size = 64

    def score(self, texts: list[str]) -> np.ndarray:
        n = len(texts)
        probs = np.tile([0.8, 0.15, 0.05], (n, 1))
        probs[:, 1] += np.arange(n) * 0.001
        probs[:, 2] -= np.arange(n) * 0.001
        return probs


def _pool_csv(tmp_path, n: int = 60) -> str:
    df = pd.DataFrame({"text": [f"مراجعة {i}" for i in range(n)]})
    path = tmp_path / "pool.csv"
    df.to_csv(path, index=False)
    return str(path)


class TestDateSeed:
    def test_date_seed_is_deterministic(self):
        assert _date_seed("2026-09-24") == _date_seed("2026-09-24")

    def test_date_seed_is_uint32(self):
        assert 0 <= _date_seed("2026-09-24") <= 2**32 - 1


class TestMakeInput:
    def test_same_date_same_sample(self, tmp_path):
        pool = _pool_csv(tmp_path)
        a = make_input(pool, "2026-09-24", 10, str(tmp_path / "a.csv"))
        b = make_input(pool, "2026-09-24", 10, str(tmp_path / "b.csv"))
        assert list(a["text"]) == list(b["text"])

    def test_different_date_different_sample(self, tmp_path):
        pool = _pool_csv(tmp_path)
        a = make_input(pool, "2026-09-24", 10, str(tmp_path / "a.csv"))
        c = make_input(pool, "2026-09-25", 10, str(tmp_path / "c.csv"))
        assert list(a["text"]) != list(c["text"])

    def test_adds_day_column(self, tmp_path):
        pool = _pool_csv(tmp_path)
        out = str(tmp_path / "in.csv")
        sample = make_input(pool, "2026-09-24", 10, out)
        assert next(iter(sample.columns)) == "day"
        assert (sample["day"] == "2026-09-24").all()

    def test_raises_when_samples_exceed_pool(self, tmp_path):
        pool = _pool_csv(tmp_path, n=5)
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
        pool = _pool_csv(tmp_path, n=30)
        ref = str(tmp_path / "reference.csv")
        out = init_reference(pool, 20, ref, FakeScorer())
        assert len(out) == 20
        assert {"text", "predicted_label", "positive"} <= set(out.columns)


class TestDriftCheck:
    def _write(self, tmp_path, frame: pd.DataFrame, name: str) -> str:
        path = tmp_path / name
        frame.to_csv(path, index=False)
        return str(path)

    def test_identical_distributions_pass(self, tmp_path):
        ref = pd.DataFrame(
            {
                "predicted_label": ["positive"] * 30
                + ["negative"] * 30
                + ["neutral"] * 30,
                "positive": np.random.default_rng(1).random(90),
            }
        )
        cur = ref.copy()
        ref_csv = self._write(tmp_path, ref, "ref.csv")
        cur_csv = self._write(tmp_path, cur, "cur.csv")
        out = str(tmp_path / "drift.json")
        verdict = drift_check(ref_csv, cur_csv, out, "2026-09-24")
        assert verdict["overall"] == "PASS"
        assert all(c["decision"] == "PASS" for c in verdict["columns"].values())
        assert verdict["date"] == "2026-09-24"

    def test_shifted_distributions_fail(self, tmp_path):
        ref = pd.DataFrame(
            {
                "predicted_label": ["positive"] * 30
                + ["negative"] * 30
                + ["neutral"] * 30,
                "positive": np.random.default_rng(2).random(90),
            }
        )
        cur = pd.DataFrame(
            {
                "predicted_label": ["positive"] * 80
                + ["negative"] * 5
                + ["neutral"] * 5,
                "positive": np.random.default_rng(3).beta(10, 1, 90),
            }
        )
        ref_csv = self._write(tmp_path, ref, "ref.csv")
        cur_csv = self._write(tmp_path, cur, "cur.csv")
        verdict = drift_check(
            ref_csv, cur_csv, str(tmp_path / "drift.json"), "2026-09-24"
        )
        assert all(c["decision"] == "FAIL" for c in verdict["columns"].values())
        assert verdict["overall"] == "FAIL"

    def test_mild_shift_does_not_fail(self, tmp_path):
        ref = pd.DataFrame(
            {
                "predicted_label": ["positive"] * 40
                + ["negative"] * 30
                + ["neutral"] * 20,
                "positive": np.random.default_rng(4).random(90),
            }
        )
        cur = pd.DataFrame(
            {
                "predicted_label": ["positive"] * 34
                + ["negative"] * 31
                + ["neutral"] * 25,
                "positive": np.random.default_rng(5).random(90),
            }
        )
        ref_csv = self._write(tmp_path, ref, "ref.csv")
        cur_csv = self._write(tmp_path, cur, "cur.csv")
        verdict = drift_check(
            ref_csv, cur_csv, str(tmp_path / "drift.json"), "2026-09-24"
        )
        assert all(
            c["decision"] in ("PASS", "WARN") for c in verdict["columns"].values()
        )

    def test_verdict_json_is_written(self, tmp_path):
        ref = pd.DataFrame(
            {
                "predicted_label": ["positive"] * 50 + ["negative"] * 50,
                "positive": np.random.default_rng(6).random(100),
            }
        )
        ref_csv = self._write(tmp_path, ref, "ref.csv")
        out = str(tmp_path / "d.json")
        verdict = drift_check(ref_csv, ref_csv, out, "2026-09-24")
        with open(out) as f:
            assert json.load(f)["overall"] == verdict["overall"]
