import pandas as pd

from raay.data.preprocess import (
    collapse_elongation,
    flag_near_empty,
    fuzzy_deduplicate,
    has_diacritics,
    has_elongation,
    has_emoji,
    has_latin,
    normalize_casing,
    remove_diacritics,
)


def test_flag_near_empty():
    assert flag_near_empty("hello", min_length=10) is True
    assert flag_near_empty("hello world", min_length=10) is False
    assert flag_near_empty("   ", min_length=10) is True


def test_remove_diacritics():
    # Arabic with tashkeel
    text_with_tashkeel = "مَرْحَبًا"
    text_without_tashkeel = "مرحبا"
    assert remove_diacritics(text_with_tashkeel) == text_without_tashkeel


def test_collapse_elongation():
    assert collapse_elongation("ببببب", max_repeat=2) == "بب"
    assert collapse_elongation("بب", max_repeat=2) == "بب"
    assert collapse_elongation("ب", max_repeat=2) == "ب"
    assert collapse_elongation("ببب", max_repeat=1) == "ب"


def test_normalize_casing():
    assert normalize_casing("Hello WORLD") == "hello world"
    assert normalize_casing("مرحبا") == "مرحبا"


def test_has_latin():
    assert has_latin("مرحبا Hello") is True
    assert has_latin("مرحبا") is False


def test_has_emoji():
    assert has_emoji("مرحبا 😊") is True
    assert has_emoji("مرحبا") is False


def test_has_diacritics():
    assert has_diacritics("مَرْحَبًا") is True
    assert has_diacritics("مرحبا") is False


def test_has_elongation():
    assert has_elongation("ببببب") is True
    assert has_elongation("بب") is False


def test_fuzzy_deduplicate():
    # Creating a sample dataframe with near duplicates
    df = pd.DataFrame(
        {
            "text": [
                "This is a test review for the product",
                "This is a test reveiw for the product",  # Typo, should be dropped if threshold allows
                "Completely different review here",
            ],
            "label": ["positive", "positive", "negative"],
        }
    )

    deduped_df = fuzzy_deduplicate(df, "text", threshold=0.9)
    # The second one should be removed as it is highly similar
    assert len(deduped_df) == 2
    assert "Completely different review here" in deduped_df["text"].values
    assert "This is a test review for the product" in deduped_df["text"].values
