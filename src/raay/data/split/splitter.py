import pandas as pd
from typing import Tuple
from raay.config.data_config import SplitConfig


class DataSplitter:
    """Model/Component responsible for splitting data into train, val, test sets."""

    def __init__(self, config: SplitConfig):
        self.config = config

    def split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits the dataframe into train, validation, and test sets.

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # TODO: Implement actual splitting logic using scikit-learn
        # This is a structural placeholder

        train_df = pd.DataFrame()
        val_df = pd.DataFrame()
        test_df = pd.DataFrame()

        return train_df, val_df, test_df
