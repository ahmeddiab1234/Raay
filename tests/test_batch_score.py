"""Unit tests for the nightly batch re-scoring job + Evidently PSI drift.

Hermetic by design (AGENTS.md rule: unit tests only): no ONNX graph, no
MLflow, no Airflow, no encoder weights. The duck-typed fakes in ``conftest.py``
stand in for the INT8 session and the AraBERT encoder; drift checks run on tiny
synthetic DataFrames.
"""

import json

import numpy as np
import pandas as pd
import pytest
from conftest import (
    FakeEmbedder,
    FakeScorer,
    FakeTokenizer,
    ascii_vocab,
    identity,
    scored_frame,
)

from raay.inference.batch_score import (
    EngineeredDriftSpec,
    _date_seed,
    _psi_per_column,
    drift_check,
    init_reference,
    make_input,
    score_input,
)
from raay.inference.drift_features import (
    COL_DIALECT_LABEL,
    COL_OOV_RATE,
    COL_TEXT_LENGTH,
    DriftFeatureBuilder,
    default_drift_columns,
    embedding_pc_columns,
    save_basis,
)


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


def _write(tmp_path, frame: pd.DataFrame, name: str) -> str:
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return str(path)


class TestDriftCheck:
    _write = staticmethod(_write)

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
        ref_csv = _write(tmp_path, ref, "ref.csv")
        cur_csv = _write(tmp_path, cur, "cur.csv")
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
        ref_csv = _write(tmp_path, ref, "ref.csv")
        cur_csv = _write(tmp_path, cur, "cur.csv")
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
        ref_csv = _write(tmp_path, ref, "ref.csv")
        cur_csv = _write(tmp_path, cur, "cur.csv")
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
        ref_csv = _write(tmp_path, ref, "ref.csv")
        out = str(tmp_path / "d.json")
        verdict = drift_check(ref_csv, ref_csv, out, "2026-09-24")
        with open(out) as f:
            assert json.load(f)["overall"] == verdict["overall"]


def _spec(tmp_path, n_components: int = 3, reference: pd.DataFrame | None = None):
    """An ``EngineeredDriftSpec`` wired to the fakes, with a fitted basis on disk.

    Returns ``(spec, builder)``. The basis is fitted on ``reference`` (or a
    default panel) so ``drift_check`` has a frozen projection to load, exactly
    as ``init-reference`` would have written.
    """
    builder = DriftFeatureBuilder(
        FakeEmbedder(),
        FakeTokenizer(ascii_vocab()),
        n_components=n_components,
        model_dir="models/baseline/final",
        max_length=128,
        preprocess=identity,
    )
    ref = reference if reference is not None else scored_frame(60)
    basis = builder.fit_basis(builder.embed(ref))
    pca_path = str(tmp_path / "pca_basis.joblib")
    save_basis(basis, pca_path)
    spec = EngineeredDriftSpec(
        builder=builder,
        pca_path=pca_path,
        reference_engineered=str(tmp_path / "reference_engineered.csv"),
        current_engineered=str(tmp_path / "current_engineered.csv"),
    )
    return spec, builder


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
        spec, _ = _spec(tmp_path, reference=ref)
        ref_csv = _write(tmp_path, ref, "ref.csv")
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
        spec, _ = _spec(tmp_path, reference=ref)
        cur = scored_frame(60, seed=12, filler=" x")
        ref_csv = _write(tmp_path, ref, "ref.csv")
        cur_csv = _write(tmp_path, cur, "cur.csv")
        verdict = drift_check(
            ref_csv, cur_csv, str(tmp_path / "d.json"), "2026-09-24", engineered=spec
        )
        assert verdict["columns"][COL_TEXT_LENGTH]["drift_detected"] is True
        assert verdict["overall"] in ("WARN", "FAIL")

    def test_rising_oov_rate_is_detected(self, tmp_path):
        ref = scored_frame(60, seed=13)
        spec, _ = _spec(tmp_path, reference=ref)
        cur = scored_frame(60, seed=13)
        # Half the current panel is unspellable to the fake tokenizer; the
        # reference is all known characters.
        cur["text"] = [f"zz {'review number ' + str(i)}" for i in range(60)]
        ref_csv = _write(tmp_path, ref, "ref.csv")
        cur_csv = _write(tmp_path, cur, "cur.csv")
        verdict = drift_check(
            ref_csv, cur_csv, str(tmp_path / "d.json"), "2026-09-24", engineered=spec
        )
        assert verdict["columns"][COL_OOV_RATE]["drift_detected"] is True

    def test_engineered_verdict_reports_the_projection(self, tmp_path):
        ref = scored_frame(60, seed=14)
        spec, _ = _spec(tmp_path, n_components=3, reference=ref)
        ref_csv = _write(tmp_path, ref, "ref.csv")
        verdict = drift_check(
            ref_csv, ref_csv, str(tmp_path / "d.json"), "2026-09-24", engineered=spec
        )
        projection = verdict["projection"]
        assert projection["n_components"] == 3
        assert projection["pooling"] == "mean"
        assert projection["max_length"] == 128
        assert projection["n_reference"] == 60

    def test_engineered_verdict_reports_feature_means_and_dialect_mix(self, tmp_path):
        ref = scored_frame(60, seed=15)
        spec, _ = _spec(tmp_path, reference=ref)
        ref_csv = _write(tmp_path, ref, "ref.csv")
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
        spec, _ = _spec(tmp_path, reference=ref)
        ref_csv = _write(tmp_path, ref, "ref.csv")
        drift_check(
            ref_csv, ref_csv, str(tmp_path / "d.json"), "2026-09-24", engineered=spec
        )
        written = pd.read_csv(spec.current_engineered)
        assert set(embedding_pc_columns(3)) <= set(written.columns)
        assert COL_DIALECT_LABEL in written.columns

    def test_reference_is_reused_from_the_cache_when_present(self, tmp_path):
        ref = scored_frame(60, seed=17)
        spec, _ = _spec(tmp_path, reference=ref)
        ref_csv = _write(tmp_path, ref, "ref.csv")
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
        spec, _ = _spec(tmp_path, reference=ref)
        ref_csv = _write(tmp_path, ref, "ref.csv")
        verdict = drift_check(
            ref_csv, ref_csv, str(tmp_path / "d.json"), "2026-09-24", engineered=spec
        )
        assert verdict["reference_from_cache"] is False
        assert verdict["overall"] == "PASS"

    def test_missing_pca_basis_names_the_command(self, tmp_path):
        spec, _ = _spec(tmp_path)
        ref = scored_frame(60, seed=19)
        ref_csv = _write(tmp_path, ref, "ref.csv")
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
        spec, _ = _spec(tmp_path, reference=ref)
        ref_csv = _write(tmp_path, ref, "ref.csv")
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
        spec, builder = _spec(tmp_path, n_components=3, reference=ref)
        engineered_ref = builder.engineer(ref, builder.fit_basis(builder.embed(ref)))
        engineered_ref.drop(columns=["embedding_pc3"]).to_csv(
            spec.reference_engineered, index=False
        )
        ref_csv = _write(tmp_path, ref, "ref.csv")
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
        spec, _ = _spec(tmp_path, reference=ref)
        ref_csv = _write(tmp_path, ref, "ref.csv")
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
        spec, _ = _spec(tmp_path)
        ref = pd.DataFrame(
            {
                "predicted_label": ["positive"] * 30 + ["negative"] * 30,
                "positive": np.random.default_rng(23).random(60),
            }
        )
        ref_csv = _write(tmp_path, ref, "ref.csv")
        verdict = drift_check(
            ref_csv, ref_csv, str(tmp_path / "d.json"), "2026-09-24", engineered=spec
        )
        assert set(verdict["columns"]) == {"predicted_label", "positive"}
        assert "engineered" not in verdict
        assert verdict["overall"] == "PASS"


class TestUncomparableColumns:
    """A column with no reference spread cannot drift, and cannot be binned.

    This is not hypothetical: the tail PCs of a real PCA basis sit at float
    noise, and Evidently's sturges binning then raises. Gating those would fail
    the report for a component that cannot move; omitting them silently would
    read as "checked, no drift". The verdict has to say SKIPPED and show the std.
    """

    def test_constant_reference_column_is_skipped(self, tmp_path):
        ref = scored_frame(60, seed=30)
        spec, _ = _spec(tmp_path, reference=ref)
        ref["flat"] = 1.0
        ref_csv = _write(tmp_path, ref, "ref.csv")
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
        spec, _ = _spec(tmp_path, reference=ref)
        ref["flat"] = 0.0
        ref_csv = _write(tmp_path, ref, "ref.csv")
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
        spec, _ = _spec(tmp_path, reference=ref)
        ref["flat"] = 0.0
        ref_csv = _write(tmp_path, ref, "ref.csv")
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
        spec, builder = _spec(tmp_path, n_components=3, reference=ref)
        basis = builder.fit_basis(builder.embed(ref))
        ref_csv = _write(tmp_path, ref, "ref.csv")
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
        spec, _ = _spec(tmp_path, reference=ref)
        ref[COL_DIALECT_LABEL] = "msa"
        ref_csv = _write(tmp_path, ref, "ref.csv")
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
        verdicts, errored, uncomparable = _psi_per_column(
            ["boom"],
            pd.DataFrame({"boom": [float(i) for i in range(50)]}),
            pd.DataFrame({"boom": [float(i) for i in range(50)]}),
            (0.1, 0.2),
        )
        assert verdicts["boom"]["decision"] == "ERROR"
        assert verdicts["boom"]["drift_score"] is None
        assert "simulated binning failure" in verdicts["boom"]["error"]
        assert errored == ["boom"]
        assert uncomparable == []


class TestEngineeredInitReference:
    def test_writes_the_basis_and_the_engineered_panel(self, tmp_path):
        spec, builder = _spec(tmp_path)
        pool = _pool_csv(tmp_path, n=40)
        out = str(tmp_path / "reference.csv")
        frame = init_reference(pool, 30, out, FakeScorer(), engineered=spec)
        assert (tmp_path / "pca_basis.joblib").exists()
        assert (tmp_path / "reference_engineered.csv").exists()
        assert set(embedding_pc_columns(3)) <= set(frame.columns)
        assert COL_OOV_RATE in frame.columns
        assert builder.embedder.calls == 2, (
            "_spec() embedded once to fit the basis; init_reference must embed "
            "exactly once more and reuse those embeddings for the transform"
        )

    def test_the_basis_is_usable_by_the_drift_check(self, tmp_path):
        """init-reference and drift have to agree on the projection, or the
        nightly gate would be comparing against a different basis than it fitted."""
        spec, _ = _spec(tmp_path)
        pool = _pool_csv(tmp_path, n=40)
        out = str(tmp_path / "reference.csv")
        init_reference(pool, 30, out, FakeScorer(), engineered=spec)
        verdict = drift_check(
            out, out, str(tmp_path / "d.json"), "2026-09-24", engineered=spec
        )
        assert verdict["overall"] == "PASS"
        assert verdict["reference_from_cache"] is True

    def test_engineering_is_opt_in(self, tmp_path):
        """Passing no spec must leave the original behaviour untouched."""
        pool = _pool_csv(tmp_path, n=40)
        out = str(tmp_path / "reference.csv")
        frame = init_reference(pool, 30, out, FakeScorer())
        assert COL_OOV_RATE not in frame.columns
        assert not (tmp_path / "pca_basis.joblib").exists()
