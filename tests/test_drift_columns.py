"""Engineered drift columns: which features are gated and what they detect."""

from __future__ import annotations

import pandas as pd
import pytest
from conftest import scored_frame
from drift_helpers import engineered_spec, write_frame

from raay.inference.drift_engine import drift_check
from raay.inference.drift_features import (
    COL_DIALECT_LABEL,
    COL_OOV_RATE,
    COL_TEXT_LENGTH,
    default_drift_columns,
    embedding_pc_columns,
)


class TestEngineeredDriftColumns:
    def test_default_drift_columns_include_the_engineered_features(self):
        cols = default_drift_columns(10)
        for required in (
            "predicted_label",
            "positive",
            COL_TEXT_LENGTH,
            COL_OOV_RATE,
            COL_DIALECT_LABEL,
            *embedding_pc_columns(10),
        ):
            assert required in cols, f"{required} missing from the gated columns"

    def test_identical_panels_pass_on_every_engineered_column(self, tmp_path):
        ref = scored_frame(60, seed=11)
        spec, _ = engineered_spec(tmp_path, reference=ref)
        ref_csv = write_frame(tmp_path, ref, "ref.csv")
        verdict = drift_check(
            ref_csv, ref_csv, str(tmp_path / "d.json"), "2026-09-24", engineered=spec
        )
        assert verdict["engineered"] is True
        assert set(verdict["columns"]) == set(default_drift_columns(3))
        assert verdict["overall"] == "PASS"
        assert verdict["skipped_columns"] == []

    def test_shorter_reviews_are_detected_on_text_length(self, tmp_path):
        """A panel of far shorter reviews must move text_length, not pass quietly."""
        ref = scored_frame(60, seed=12, filler=" filler words to make them long")
        spec, _ = engineered_spec(tmp_path, reference=ref)
        cur = scored_frame(60, seed=12, filler=" x")
        ref_csv = write_frame(tmp_path, ref, "ref.csv")
        cur_csv = write_frame(tmp_path, cur, "cur.csv")
        verdict = drift_check(
            ref_csv, cur_csv, str(tmp_path / "d.json"), "2026-09-24", engineered=spec
        )
        assert verdict["columns"][COL_TEXT_LENGTH]["drift_detected"] is True
        assert verdict["overall"] in ("WARN", "FAIL")

    def test_rising_oov_rate_is_detected(self, tmp_path):
        ref = scored_frame(60, seed=13)
        spec, _ = engineered_spec(tmp_path, reference=ref)
        cur = scored_frame(60, seed=13)
        # Half the current panel is unspellable to the fake tokenizer; the
        # reference is all known characters.
        cur["text"] = [f"zz {'review number ' + str(i)}" for i in range(60)]
        ref_csv = write_frame(tmp_path, ref, "ref.csv")
        cur_csv = write_frame(tmp_path, cur, "cur.csv")
        verdict = drift_check(
            ref_csv, cur_csv, str(tmp_path / "d.json"), "2026-09-24", engineered=spec
        )
        assert verdict["columns"][COL_OOV_RATE]["drift_detected"] is True

    def test_engineered_verdict_reports_feature_means_and_dialect_mix(self, tmp_path):
        ref = scored_frame(60, seed=15)
        spec, _ = engineered_spec(tmp_path, reference=ref)
        ref_csv = write_frame(tmp_path, ref, "ref.csv")
        verdict = drift_check(
            ref_csv, ref_csv, str(tmp_path / "d.json"), "2026-09-24", engineered=spec
        )
        assert COL_TEXT_LENGTH in verdict["feature_means"]
        assert COL_OOV_RATE in verdict["feature_means"]
        assert verdict["dialect_mix"]["total_variation"] == 0.0
        assert sum(verdict["dialect_mix"]["current"].values()) == pytest.approx(1.0)

    def test_engineered_current_panel_is_persisted(self, tmp_path):
        """A failing gate needs an inspectable panel, not just a score."""
        ref = scored_frame(60, seed=16)
        spec, _ = engineered_spec(tmp_path, reference=ref)
        ref_csv = write_frame(tmp_path, ref, "ref.csv")
        drift_check(
            ref_csv, ref_csv, str(tmp_path / "d.json"), "2026-09-24", engineered=spec
        )
        written = pd.read_csv(spec.current_engineered)
        assert set(embedding_pc_columns(3)) <= set(written.columns)
        assert COL_DIALECT_LABEL in written.columns
