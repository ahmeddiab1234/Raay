"""Frozen PCA basis handling: caching, mismatch refusal, skips, fallback."""

from __future__ import annotations

import numpy as np
import pandas as pd
from conftest import scored_frame
from drift_helpers import engineered_spec, write_frame

from raay.inference.drift_engine import EngineeredDriftSpec, drift_check
from raay.inference.drift_features import default_drift_columns


class TestEngineeredDriftProjection:
    def test_engineered_verdict_reports_the_projection(self, tmp_path):
        ref = scored_frame(60, seed=14)
        spec, _ = engineered_spec(tmp_path, n_components=3, reference=ref)
        ref_csv = write_frame(tmp_path, ref, "ref.csv")
        verdict = drift_check(
            ref_csv, ref_csv, str(tmp_path / "d.json"), "2026-09-24", engineered=spec
        )
        projection = verdict["projection"]
        assert projection["n_components"] == 3
        assert projection["pooling"] == "mean"
        assert projection["max_length"] == 128
        assert projection["n_reference"] == 60

    def test_reference_is_reused_from_the_cache_when_present(self, tmp_path):
        ref = scored_frame(60, seed=17)
        spec, _ = engineered_spec(tmp_path, reference=ref)
        ref_csv = write_frame(tmp_path, ref, "ref.csv")
        spec.builder.engineer(
            ref, spec.builder.fit_basis(spec.builder.embed(ref))
        ).to_csv(spec.reference_engineered, index=False)
        verdict = drift_check(
            ref_csv, ref_csv, str(tmp_path / "d.json"), "2026-09-24", engineered=spec
        )
        assert verdict["reference_from_cache"] is True
        assert verdict["reference"] == spec.reference_engineered

    def test_missing_reference_cache_is_engineered_on_the_fly(self, tmp_path):
        ref = scored_frame(60, seed=18)
        spec, _ = engineered_spec(tmp_path, reference=ref)
        ref_csv = write_frame(tmp_path, ref, "ref.csv")
        verdict = drift_check(
            ref_csv, ref_csv, str(tmp_path / "d.json"), "2026-09-24", engineered=spec
        )
        assert verdict["reference_from_cache"] is False
        assert verdict["overall"] == "PASS"

    def test_missing_pca_basis_names_the_command(self, tmp_path):
        spec, _ = engineered_spec(tmp_path)
        ref = scored_frame(60, seed=19)
        ref_csv = write_frame(tmp_path, ref, "ref.csv")
        spec = EngineeredDriftSpec(
            builder=spec.builder,
            pca_path=str(tmp_path / "absent.joblib"),
            reference_engineered=spec.reference_engineered,
            current_engineered=spec.current_engineered,
        )
        try:
            drift_check(
                ref_csv,
                ref_csv,
                str(tmp_path / "d.json"),
                "2026-09-24",
                engineered=spec,
            )
        except FileNotFoundError as exc:
            assert "init-reference" in str(exc)
        else:
            raise AssertionError("expected FileNotFoundError")

    def test_basis_with_mismatched_preprocessing_is_refused(self, tmp_path):
        ref = scored_frame(60, seed=20)
        spec, _ = engineered_spec(tmp_path, reference=ref)
        ref_csv = write_frame(tmp_path, ref, "ref.csv")
        spec.builder.max_length = 64  # basis was fitted at 128
        try:
            drift_check(
                ref_csv,
                ref_csv,
                str(tmp_path / "d.json"),
                "2026-09-24",
                engineered=spec,
            )
        except ValueError as exc:
            assert "max_length" in str(exc)
        else:
            raise AssertionError("expected ValueError")

    def test_absent_columns_are_skipped_and_named(self, tmp_path):
        """A basis clamped to fewer components than requested leaves columns that
        do not exist. The gate must compare what is there and name what it
        dropped, not crash and not pad the missing ones with zeros."""
        ref = scored_frame(60, seed=21)
        spec, builder = engineered_spec(tmp_path, n_components=3, reference=ref)
        engineered_ref = builder.engineer(ref, builder.fit_basis(builder.embed(ref)))
        engineered_ref.drop(columns=["embedding_pc3"]).to_csv(
            spec.reference_engineered, index=False
        )
        ref_csv = write_frame(tmp_path, ref, "ref.csv")
        verdict = drift_check(
            ref_csv,
            ref_csv,
            str(tmp_path / "d.json"),
            "2026-09-24",
            engineered=spec,
            drift_columns=default_drift_columns(3),
        )
        assert "embedding_pc3" not in verdict["columns"]
        assert verdict["skipped_columns"] == ["embedding_pc3"]

    def test_no_comparable_columns_is_an_error_not_an_empty_report(self, tmp_path):
        ref = scored_frame(60, seed=22)
        spec, _ = engineered_spec(tmp_path, reference=ref)
        ref_csv = write_frame(tmp_path, ref, "ref.csv")
        try:
            drift_check(
                ref_csv,
                ref_csv,
                str(tmp_path / "d.json"),
                "2026-09-24",
                engineered=spec,
                drift_columns=("embedding_zzz",),
            )
        except ValueError as exc:
            assert "none of the requested drift columns" in str(exc)
        else:
            raise AssertionError("expected ValueError")

    def test_frames_without_text_fall_back_to_the_output_columns(self, tmp_path):
        """The legacy path must survive: synthetic frames have no text, so the
        engineered columns simply are not available."""
        spec, _ = engineered_spec(tmp_path)
        ref = pd.DataFrame(
            {
                "predicted_label": ["positive"] * 30 + ["negative"] * 30,
                "positive": np.random.default_rng(23).random(60),
            }
        )
        ref_csv = write_frame(tmp_path, ref, "ref.csv")
        verdict = drift_check(
            ref_csv, ref_csv, str(tmp_path / "d.json"), "2026-09-24", engineered=spec
        )
        assert set(verdict["columns"]) == {"predicted_label", "positive"}
        assert "engineered" not in verdict
        assert verdict["overall"] == "PASS"
