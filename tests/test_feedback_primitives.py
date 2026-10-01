"""Pure feedback text and label helper tests."""

from __future__ import annotations

import pandas as pd

from raay.data.feedback import coerce_label, label_proportions, normalize_text
from raay.enums.constants import LABELS


def test_normalize_text_matches_the_preprocessing_pipeline():
    """The merged file must not mix normalized and raw text.

    train.csv texts went through remove_diacritics -> collapse_elongation ->
    normalize_casing. A feedback row that skipped those steps would be a visible
    distribution artifact in the same training file.
    """
    assert normalize_text("لللل custtom PRODUCT") == "لل custtom product"


def test_normalize_text_strips_tashkeel():
    assert normalize_text("مُحَمَّد") == "محمد"


def test_normalize_text_is_idempotent():
    once = normalize_text("كــرisement PRODUCT")
    assert normalize_text(once) == once


def test_coerce_label_rejects_anything_outside_labels():
    assert coerce_label("negative") == "negative"
    assert coerce_label("excellent") == ""
    assert coerce_label("0") == ""


def test_label_proportions_is_zero_filled_over_labels():
    """An absent class is a fact about the batch, not a missing key."""
    series = pd.Series(["positive", "positive", "negative"])
    props = label_proportions(series)
    assert set(props) == set(LABELS)
    assert props["neutral"] == 0.0
    assert props["positive"] == round(2 / 3, 5)


def test_label_proportions_of_an_empty_series_is_all_zero():
    assert label_proportions(pd.Series([], dtype=str)) == dict.fromkeys(LABELS, 0.0)
