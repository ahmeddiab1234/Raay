"""PSI primitive behaviour: dead columns, skipped columns, binning failures."""

from __future__ import annotations

import pandas as pd
from conftest import scored_frame
from drift_helpers import engineered_spec, write_frame

from raay.inference.drift_engine import drift_check
from raay.inference.drift_features import COL_DIALECT_LABEL, embedding_pc_columns
from raay.inference.drift_psi import _psi_per_column


class TestUncomparableColumns:
    """A column with no reference spread cannot drift, and cannot be binned.

    This is not hypothetical: the tail PCs of a real PCA basis sit at float
    noise, and Evidently's sturges binning then raises. Gating those would fail
    the report for a component that cannot move; omitting them silently would
    read as "checked, no drift". The verdict has to say SKIPPED and show the std.
    """

    def test_constant_reference_column_is_skipped(self, tmp_path):
        ref = scored_frame(60, seed=30)
        spec, _ = engineered_spec(tmp_path, reference=ref)
        ref["flat"] = 1.0
        ref_csv = write_frame(tmp_path, ref, "ref.csv")
        verdict = drift_check(
            ref_csv,
            ref_csv,
            str(tmp_path / "d.json"),
            "2026-09-24",
            engineered=spec,
            drift_columns=("flat", "text_length"),
        )
        assert verdict["columns"]["flat"]["decision"] == "SKIPPED"
        assert "no spread" in verdict["columns"]["flat"]["reason"]
        assert verdict["uncomparable_columns"] == ["flat"]
        assert verdict["columns"]["text_length"]["decision"] == "PASS"
        assert verdict["overall"] == "PASS", "a dead column must not fail the gate"

    def test_skipped_columns_are_excluded_from_drift_share(self, tmp_path):
        ref = scored_frame(60, seed=31)
        spec, _ = engineered_spec(tmp_path, reference=ref)
        ref["flat"] = 0.0
        ref_csv = write_frame(tmp_path, ref, "ref.csv")
        verdict = drift_check(
            ref_csv,
            ref_csv,
            str(tmp_path / "d.json"),
            "2026-09-24",
            engineered=spec,
            drift_columns=("flat",),
        )
        assert verdict["n_columns_checked"] == 0
        assert verdict["drift_share"] == 0.0

    def test_a_gate_that_checked_nothing_is_not_a_pass(self, tmp_path):
        ref = scored_frame(60, seed=32)
        spec, _ = engineered_spec(tmp_path, reference=ref)
        ref["flat"] = 0.0
        ref_csv = write_frame(tmp_path, ref, "ref.csv")
        verdict = drift_check(
            ref_csv,
            ref_csv,
            str(tmp_path / "d.json"),
            "2026-09-24",
            engineered=spec,
            drift_columns=("flat",),
        )
        assert verdict["overall"] == "SKIPPED"

    def test_real_degenerate_pcs_are_skipped_not_errored(self, tmp_path):
        """The FakeEmbedder's 3rd+ components are pure float noise (std ~1e-16),
        which is exactly the production shape of a PCA basis's tail."""
        ref = scored_frame(60, seed=33)
        spec, builder = engineered_spec(tmp_path, n_components=3, reference=ref)
        basis = builder.fit_basis(builder.embed(ref))
        ref_csv = write_frame(tmp_path, ref, "ref.csv")
        verdict = drift_check(
            ref_csv,
            ref_csv,
            str(tmp_path / "d.json"),
            "2026-09-24",
            engineered=spec,
            drift_columns=embedding_pc_columns(basis.n_components),
        )
        noisy = [
            col
            for col in embedding_pc_columns(basis.n_components)
            if verdict["columns"][col]["decision"] == "SKIPPED"
        ]
        assert noisy, "expected the noise-floor components to be skipped"
        assert verdict["errored_columns"] == []
        for col in noisy:
            assert verdict["columns"][col]["reference_std"] < 1e-8

    def test_constant_categorical_column_is_still_compared(self, tmp_path):
        """Every row the same dialect is a real distribution, not a dead column:
        Evidently bins categories rather than a numeric range."""
        ref = scored_frame(60, seed=34)
        spec, _ = engineered_spec(tmp_path, reference=ref)
        ref[COL_DIALECT_LABEL] = "msa"
        ref_csv = write_frame(tmp_path, ref, "ref.csv")
        verdict = drift_check(
            ref_csv,
            ref_csv,
            str(tmp_path / "d.json"),
            "2026-09-24",
            engineered=spec,
            drift_columns=(COL_DIALECT_LABEL,),
        )
        assert verdict["columns"][COL_DIALECT_LABEL]["decision"] == "PASS"

    def test_an_unexpected_binning_failure_escalates(self, tmp_path, monkeypatch):
        """A binning failure on a column that *does* have spread is a bug, not a
        dead column, so it must be ERROR (which lifts the verdict) and never a
        silent omission."""
        import evidently.legacy.report as ev_report

        class ExplodingReport:
            def __init__(self, metrics=None) -> None:
                self.metrics = metrics or []

            def run(self, reference_data, current_data):
                raise RuntimeError("simulated binning failure")

        monkeypatch.setattr(ev_report, "Report", ExplodingReport)
        # _psi_per_column imports Report inside its body, so patching the
        # evidently module attribute is what actually takes effect.
        frame = pd.DataFrame({"boom": [float(i) for i in range(50)]})
        verdicts, errored, uncomparable = _psi_per_column(
            ["boom"], frame, frame, (0.1, 0.2)
        )
        assert verdicts["boom"]["decision"] == "ERROR"
        assert verdicts["boom"]["drift_score"] is None
        assert "simulated binning failure" in verdicts["boom"]["error"]
        assert errored == ["boom"]
        assert uncomparable == []
