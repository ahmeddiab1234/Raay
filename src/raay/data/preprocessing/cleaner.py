import pandas as pd
from raay.config.data_config import PreprocessingConfig


class TextCleaner:
    """Model/Component responsible for cleaning text data."""

    def __init__(self, config: PreprocessingConfig):
        self.config = config

    def _remove_punctuation(self, text: str) -> str:
        # TODO: Implement punctuation removal logic
        return text

    def _normalize_arabic(self, text: str) -> str:
        # TODO: Implement Arabic normalization logic
        return text

    def clean(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Applies configured cleaning steps to the DataFrame."""
        # This is a structural placeholder
        df_cleaned = df.copy()

        # Example processing pipeline based on config
        if self.config.normalize_arabic:
            df_cleaned[text_column] = df_cleaned[text_column].apply(
                self._normalize_arabic
            )

        return df_cleaned
