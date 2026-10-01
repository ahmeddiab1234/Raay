"""End-to-end ``drift_check`` behaviour and engineered ``init-reference``."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from conftest import FakeScorer
from drift_helpers import engineered_spec, pool_csv, write_frame

from raay.inference.batch_reference import init_reference
from raay.inference.drift_engine import drift_check
from raay.inference.drift_features import COL_OOV_RATE, embedding_pc_columns


class TestDriftCheck:
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
        ref_csv = write_frame(tmp_path, ref, "ref.csv")
        cur_csv = write_frame(tmp_path, cur, "cur.csv")
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
        ref_csv = write_frame(tmp_path, ref, "ref.csv")
        cur_csv = write_frame(tmp_path, cur, "cur.csv")
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
        ref_csv = write_frame(tmp_path, ref, "ref.csv")
        cur_csv = write_frame(tmp_path, cur, "cur.csv")
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
        ref_csv = write_frame(tmp_path, ref, "ref.csv")
        out = str(tmp_path / "d.json")
        verdict = drift_check(ref_csv, ref_csv, out, "2026-09-24")
        with open(out) as f:
            assert json.load(f)["overall"] == verdict["overall"]


class TestEngineeredInitReference:
    def test_writes_the_basis_and_the_engineered_panel(self, tmp_path):
        spec, builder = engineered_spec(tmp_path)
        pool = pool_csv(tmp_path, n=40)
        out = str(tmp_path / "reference.csv")
        frame = init_reference(pool, 30, out, FakeScorer(), engineered=spec)
        assert (tmp_path / "pca_basis.joblib").exists()
        assert (tmp_path / "reference_engineered.csv").exists()
        assert set(embedding_pc_columns(3)) <= set(frame.columns)
        assert COL_OOV_RATE in frame.columns
        assert builder.embedder.calls == 2, (
            "engineered_spec() embedded once to fit the basis; init_reference must "
            "embed exactly once more and reuse those embeddings for the transform"
        )

    def test_the_basis_is_usable_by_the_drift_check(self, tmp_path):
        """init-reference and drift have to agree on the projection, or the
        nightly gate would be comparing against a different basis than it fitted."""
        spec, _ = engineered_spec(tmp_path)
        pool = pool_csv(tmp_path, n=40)
        out = str(tmp_path / "reference.csv")
        init_reference(pool, 30, out, FakeScorer(), engineered=spec)
        verdict = drift_check(
            out, out, str(tmp_path / "d.json"), "2026-09-24", engineered=spec
        )
        assert verdict["overall"] == "PASS"
        assert verdict["reference_from_cache"] is True

    def test_engineering_is_opt_in(self, tmp_path):
        """Passing no spec must leave the original behaviour untouched."""
        pool = pool_csv(tmp_path, n=40)
        out = str(tmp_path / "reference.csv")
        frame = init_reference(pool, 30, out, FakeScorer())
        assert COL_OOV_RATE not in frame.columns
        assert not (tmp_path / "pca_basis.joblib").exists()
