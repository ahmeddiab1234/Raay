"""Unit tests for the Phase-6 engineered drift text features."""

import pandas as pd
import pytest
from conftest import (
    FakeTokenizer,
)
from conftest import (
    identity as _identity,
)

from raay.inference.drift_features import (
    COL_DIALECT_LABEL,
    COL_OOV_RATE,
    dialect_labels,
    dialect_mix,
    dialect_total_variation,
    feature_means,
    oov_bucket,
    oov_rates,
)


class TestOovRate:
    """Normalization is injected as identity: pyarabic rewrites text in ways
    unrelated to OOV (it even drops runs of repeated letters), so leaving it in
    would make these arithmetic assertions about the preprocessor instead."""

    def test_unknown_characters_are_counted(self):
        tok = FakeTokenizer({"a": 1, "b": 2})
        # 'a' and 'b' are known, 'Z' and '?' are not; digits are filtered above 128.
        assert list(oov_rates(["abZ?"], tok, preprocess=_identity)) == [0.5]

    def test_fully_known_text_is_zero(self):
        tok = FakeTokenizer({"a": 1, "b": 2})
        assert list(oov_rates(["abab"], tok, preprocess=_identity)) == [0.0]

    def test_fully_novel_text_is_one(self):
        tok = FakeTokenizer({"a": 1})
        assert list(oov_rates(["ZZZZ"], tok, preprocess=_identity)) == [1.0]

    def test_empty_token_list_is_zero_not_nan(self):
        """Division by zero here would put NaN in the panel and quietly drop a column."""
        tok = FakeTokenizer({"a": 1})
        assert list(oov_rates([""], tok, preprocess=_identity)) == [0.0]

    def test_rising_novelty_is_monotone(self):
        tok = FakeTokenizer({"a": 1})
        rates = oov_rates(["aaaa", "aaZZ", "aZZZ", "ZZZZ"], tok, preprocess=_identity)
        assert list(rates) == [0.0, 0.5, 0.75, 1.0]

    def test_tokenizer_without_unk_id_is_refused(self):
        class NoUnk(FakeTokenizer):
            unk_token_id = None

        with pytest.raises(ValueError, match="unk_token_id"):
            oov_rates(["abc"], NoUnk({}), preprocess=_identity)

    def test_batching_does_not_change_the_rates(self):
        tok = FakeTokenizer({"a": 1})
        texts = ["aaaa", "aaZZ", "ZZZZ", "aZZa"]
        assert list(oov_rates(texts, tok, batch_size=1, preprocess=_identity)) == list(
            oov_rates(texts, tok, batch_size=3, preprocess=_identity)
        )

    def test_normalization_is_applied_before_tokenizing(self):
        """The rate must describe what the encoder sees, not the raw string."""
        tok = FakeTokenizer({"a": 1})
        raw = oov_rates(["aZZ"], tok, preprocess=_identity)
        normalized = oov_rates(["aZZ"], tok, preprocess=lambda t: t.replace("ZZ", "aa"))
        assert list(raw) == [2 / 3]  # 1 known char, 2 unknown
        assert list(normalized) == [0.0]


class TestDialectLabels:
    def test_labels_are_canonical_enum_values(self):
        """Not 'Dialects.ARABIZI' -- the str(enum) round-trip that pollutes the
        stored dialect column after a CSV write."""
        labels = dialect_labels(["هذا النص كويس", "مد شطور يا معلم", "zabou3 5afak"])
        assert all("." not in label for label in labels)
        assert all(
            label in {"msa", "egyptian", "gulf", "levantine", "maghrebi", "arabizi"}
            for label in labels
        )

    def test_arabizi_is_detected(self):
        assert dialect_labels(["machi 3ajed very much"]) == ["arabizi"]

    def test_mix_shares_sum_to_one(self):
        frame = pd.DataFrame(
            {COL_DIALECT_LABEL: dialect_labels(["mod shator", "zabou3", "ok"])}
        )
        assert sum(dialect_mix(frame).values()) == pytest.approx(1.0)

    def test_total_variation_is_zero_for_an_identical_mix(self):
        frame = pd.DataFrame({COL_DIALECT_LABEL: ["msa", "gulf", "msa", "egyptian"]})
        result = dialect_total_variation(frame, frame)
        assert result["total_variation"] == 0.0

    def test_total_variation_measures_a_share_shift(self):
        ref = pd.DataFrame({COL_DIALECT_LABEL: ["msa"] * 80 + ["gulf"] * 20})
        cur = pd.DataFrame({COL_DIALECT_LABEL: ["gulf"] * 80 + ["msa"] * 20})
        # 60 points of share moved, all of it between two labels -> TVD 0.6.
        result = dialect_total_variation(ref, cur)
        assert result["total_variation"] == pytest.approx(0.6)
        assert result["reference"]["msa"] == pytest.approx(0.8)
        assert result["current"]["msa"] == pytest.approx(0.2)

    def test_total_variation_names_a_label_missing_from_one_side(self):
        """A dialect appearing for the first time is a mix shift, not an absence."""
        ref = pd.DataFrame({COL_DIALECT_LABEL: ["msa"] * 4})
        cur = pd.DataFrame({COL_DIALECT_LABEL: ["arabizi"] * 4})
        result = dialect_total_variation(ref, cur)
        assert "arabizi" in result["current"] and "arabizi" not in result["reference"]
        assert result["shares"]["arabizi"] == {
            "reference": 0.0,
            "current": 1.0,
        }
        assert result["total_variation"] == pytest.approx(1.0)


class TestOovBucket:
    def test_zero_is_none(self):
        assert oov_bucket(0.0) == "none"

    def test_just_above_zero_is_low(self):
        assert oov_bucket(0.01) == "low"

    def test_boundaries_land_in_the_lower_bucket(self):
        assert oov_bucket(0.02) == "low"
        assert oov_bucket(0.05) == "moderate"
        assert oov_bucket(0.10) == "high"

    def test_large_rates_are_high(self):
        assert oov_bucket(0.5) == "high"
        assert oov_bucket(1.0) == "high"

    def test_buckets_are_ordered(self):
        rates = [0.0, 0.01, 0.03, 0.07, 0.4]
        labels = [oov_bucket(r) for r in rates]
        order = ["none", "low", "moderate", "high"]
        assert [order.index(label) for label in labels] == sorted(
            order.index(label) for label in labels
        )

    def test_few_distinct_values_is_the_point(self):
        """Why the bucket exists: at most four distinct values keeps Evidently on
        its 'few unique values' path, where it never builds a histogram."""
        assert len({oov_bucket(r / 100) for r in range(101)}) <= 4


class TestFeatureMeans:
    def test_averages_numeric_columns_only(self):
        frame = pd.DataFrame(
            {COL_OOV_RATE: [0.0, 0.5], COL_DIALECT_LABEL: ["msa", "gulf"]}
        )
        assert feature_means(frame, (COL_OOV_RATE, COL_DIALECT_LABEL)) == {
            COL_OOV_RATE: 0.25
        }

    def test_absent_columns_are_skipped(self):
        assert feature_means(pd.DataFrame({"x": [1.0]}), (COL_OOV_RATE,)) == {}
