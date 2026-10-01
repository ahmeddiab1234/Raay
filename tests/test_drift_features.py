"""Unit tests for the Phase-6 engineered drift features.

Hermetic (AGENTS.md rule): a fake embedder and a fake tokenizer stand in for the
AraBERT encoder, so no model weights, no ``data/`` and no ONNX graph are needed.
What is under test is the parts that can be wrong silently -- the frozen-basis
contract, the ``[UNK]`` arithmetic, the dialect label canonicalisation, and the
skip/report behaviour of missing columns.
"""

import numpy as np
import pandas as pd
import pytest
from conftest import (
    HIDDEN,
    FakeEmbedder,
    FakeTokenizer,
    ascii_vocab,
    scored_frame,
)
from conftest import (
    identity as _identity,
)

from raay.inference.drift_features import (
    COL_CONFIDENCE,
    COL_DIALECT_LABEL,
    COL_OOV_RATE,
    COL_TEXT_LENGTH,
    DriftFeatureBuilder,
    ProjectionBasis,
    confidence_scores,
    default_drift_columns,
    embedding_pc_columns,
    encoder_weights_present,
    fit_basis,
    load_basis,
    save_basis,
    text_lengths,
)


def _builder(hidden_size: int = HIDDEN, n_components: int = 3) -> DriftFeatureBuilder:
    """A builder wired to the fakes: no encoder weights, no pyarabic."""
    return DriftFeatureBuilder(
        FakeEmbedder(hidden_size),
        FakeTokenizer(ascii_vocab()),
        n_components=n_components,
        max_length=128,
        preprocess=_identity,
    )


class TestColumnNames:
    def test_embedding_pc_columns_are_ordered(self):
        assert embedding_pc_columns(3) == (
            "embedding_pc1",
            "embedding_pc2",
            "embedding_pc3",
        )

    def test_default_columns_cover_every_engineered_feature(self):
        cols = default_drift_columns(10)
        assert COL_CONFIDENCE in cols
        assert COL_TEXT_LENGTH in cols
        assert COL_OOV_RATE in cols
        assert COL_DIALECT_LABEL in cols
        assert "predicted_label" in cols
        assert all(f"embedding_pc{i}" in cols for i in range(1, 11))

    def test_default_columns_track_n_components(self):
        assert len(default_drift_columns(10)) == 17
        assert len(default_drift_columns(5)) == 12


class TestEngineer:
    def test_adds_every_engineered_column(self):
        builder = _builder()
        frame = scored_frame(30)
        basis = builder.fit_basis(builder.embed(frame))
        out = builder.engineer(frame, basis)
        expected = {
            COL_TEXT_LENGTH,
            COL_CONFIDENCE,
            COL_OOV_RATE,
            COL_DIALECT_LABEL,
            *embedding_pc_columns(3),
        }
        assert expected <= set(out.columns)

    def test_does_not_mutate_the_input_frame(self):
        builder = _builder()
        frame = scored_frame(20)
        basis = builder.fit_basis(builder.embed(frame))
        builder.engineer(frame, basis)
        assert COL_OOV_RATE not in frame.columns

    def test_reuses_supplied_embeddings_instead_of_reembedding(self):
        builder = _builder()
        frame = scored_frame(20)
        embeddings = builder.embed(frame)
        assert builder.embedder.calls == 1
        builder.engineer(frame, builder.fit_basis(embeddings), embeddings)
        assert builder.embedder.calls == 1, "engineer() should not re-embed"

    def test_text_length_is_characters_not_words(self):
        assert list(text_lengths(["ab", "abcd", ""])) == [2.0, 4.0, 0.0]

    def test_row_count_mismatch_is_rejected(self):
        builder = _builder()
        frame = scored_frame(20)
        basis = builder.fit_basis(builder.embed(frame))
        with pytest.raises(ValueError, match="row-aligned"):
            builder.engineer(frame, basis, embeddings=np.zeros((5, HIDDEN)))

    def test_embed_requires_a_text_column(self):
        builder = _builder()
        with pytest.raises(ValueError, match="'text'"):
            builder.embed(pd.DataFrame({"label": [1, 2]}))

    def test_projection_is_frozen_so_a_shifted_panel_stays_shifted(self):
        """The whole point of the frozen basis, stated as a contrast.

        The current panel is much shorter than the reference, so under the
        *frozen* basis its PC1 mean is far from zero -- that offset IS the
        signal. A basis refitted on the current panel would centre it to exactly
        zero and make the shift invisible. So the test asserts both halves: the
        frozen basis preserves the offset, and a refit destroys it.
        """
        builder = _builder()
        ref = scored_frame(40)
        basis = builder.fit_basis(builder.embed(ref))
        cur = scored_frame(40)
        cur["text"] = [f"review {i}" for i in range(40)]

        frozen_mean = builder.engineer(cur, basis)["embedding_pc1"].mean()
        refitted_mean = builder.engineer(cur, builder.fit_basis(builder.embed(cur)))[
            "embedding_pc1"
        ].mean()
        assert abs(frozen_mean) > 5.0, "the frozen basis hid the length shift"
        assert refitted_mean == pytest.approx(0.0, abs=1e-9), (
            "a per-day refit would re-centre the panel and report no drift at all"
        )


class TestConfidenceScores:
    def test_prefers_predicted_score(self):
        frame = pd.DataFrame(
            {
                "predicted_score": [0.7],
                "positive": [0.1],
                "negative": [0.2],
                "neutral": [0.7],
            }
        )
        assert list(confidence_scores(frame)) == [0.7]

    def test_falls_back_to_max_class_probability(self):
        frame = pd.DataFrame({"positive": [0.1], "negative": [0.6], "neutral": [0.3]})
        assert list(confidence_scores(frame)) == [0.6]

    def test_refuses_to_invent_zeros(self):
        """A constant 0.0 column would pass PSI forever and read as 'no drift'."""
        with pytest.raises(ValueError, match="cannot derive"):
            confidence_scores(pd.DataFrame({"predicted_label": ["positive"]}))


class TestProjectionBasis:
    def test_components_are_clamped_to_the_reference_size(self):
        """A short smoke-test panel must not crash on n_components > n_samples."""
        basis = fit_basis(
            np.random.default_rng(0).random((4, HIDDEN)), requested_components=10
        )
        assert basis.n_components == 4
        assert basis.requested_components == 10

    def test_clamps_to_the_embedding_width_too(self):
        basis = fit_basis(
            np.random.default_rng(0).random((40, 3)), requested_components=10
        )
        assert basis.n_components == 3

    def test_rejects_non_2d_embeddings(self):
        with pytest.raises(ValueError, match="2-D"):
            fit_basis(np.zeros(10))

    def test_explained_variance_is_recorded(self):
        basis = fit_basis(np.random.default_rng(0).random((50, HIDDEN)))
        assert len(basis.explained_variance_ratio) == basis.n_components
        assert all(0.0 <= v <= 1.0 for v in basis.explained_variance_ratio)
        # Components here span the whole space, so the ratios sum to 1 up to
        # floating point; the assertion is on the ratio, not the exact total.
        assert sum(basis.explained_variance_ratio) == pytest.approx(1.0)

    def test_explained_variance_is_ordered(self):
        basis = fit_basis(np.random.default_rng(0).random((60, HIDDEN)))
        ratios = basis.explained_variance_ratio
        assert ratios == sorted(ratios, reverse=True)

    def test_round_trips_through_disk(self, tmp_path):
        basis = fit_basis(np.random.default_rng(0).random((40, HIDDEN)))
        path = str(tmp_path / "pca.joblib")
        save_basis(basis, path)
        loaded = load_basis(path)
        assert isinstance(loaded, ProjectionBasis)
        assert loaded.n_components == basis.n_components
        assert loaded.max_length == basis.max_length

    def test_missing_basis_names_the_command_to_run(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="init-reference"):
            load_basis(str(tmp_path / "nope.joblib"))

    def test_wrong_file_type_is_refused(self, tmp_path):
        import joblib

        path = str(tmp_path / "bogus.joblib")
        joblib.dump({"not": "a basis"}, path)
        with pytest.raises(TypeError, match="ProjectionBasis"):
            load_basis(path)

    def test_summary_is_json_friendly(self):
        basis = fit_basis(np.random.default_rng(0).random((40, HIDDEN)))
        summary = basis.summary()
        assert summary["n_components"] == basis.n_components
        assert isinstance(summary["explained_variance_total"], float)

    def test_mismatched_max_length_is_refused(self):
        """A basis fitted at 128 and applied to 64-token embeddings looks valid
        and means nothing."""
        basis = ProjectionBasis(
            pca=fit_basis(np.random.default_rng(0).random((40, HIDDEN))).pca,
            n_components=3,
            requested_components=3,
            explained_variance_ratio=[0.5, 0.3, 0.2],
            encoder_dir="models/baseline/final",
            model_name="m",
            max_length=128,
            pooling="mean",
            n_reference=40,
        )
        with pytest.raises(ValueError, match="max_length"):
            basis.check_compatible(
                model_dir="models/baseline/final", max_length=64, pooling="mean"
            )

    def test_mismatched_pooling_is_refused(self):
        basis = fit_basis(np.random.default_rng(0).random((40, HIDDEN)), pooling="mean")
        with pytest.raises(ValueError, match="pooling"):
            basis.check_compatible(
                model_dir=basis.encoder_dir, max_length=basis.max_length, pooling="cls"
            )

    def test_matching_projection_settings_pass(self):
        basis = fit_basis(np.random.default_rng(0).random((40, HIDDEN)))
        basis.check_compatible(
            model_dir=basis.encoder_dir,
            max_length=basis.max_length,
            pooling=basis.pooling,
        )


class TestEncoderWeightsPresent:
    def test_false_for_a_missing_dir(self, tmp_path):
        assert not encoder_weights_present(str(tmp_path / "nope"))

    def test_false_for_a_tokenizer_only_dir(self, tmp_path):
        """The real models/baseline/final on a fresh clone: JSON but no weights."""
        (tmp_path / "tokenizer.json").write_text("{}")
        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "model.safetensors.index.json").write_text("{}")
        assert not encoder_weights_present(str(tmp_path))

    def test_true_when_safetensors_are_present(self, tmp_path):
        (tmp_path / "model.safetensors").write_bytes(b"")
        assert encoder_weights_present(str(tmp_path))

    def test_true_for_sharded_safetensors(self, tmp_path):
        (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"")
        assert encoder_weights_present(str(tmp_path))
