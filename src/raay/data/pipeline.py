import pandas as pd
from typing import Tuple
from raay.config.data_config import PipelineConfig
from raay.enums.constants import DataColumns
from raay.data.preprocessing.cleaner import TextCleaner
from raay.data.split.splitter import DataSplitter


class DataPipeline:
    """
    Controller responsible for orchestrating the data processing pipeline.
    It coordinates the loading, cleaning, and splitting of data.
    """

    def __init__(self, config: PipelineConfig = PipelineConfig()):
        self.config = config
        self.cleaner = TextCleaner(config.preprocessing)
        self.splitter = DataSplitter(config.split)

    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Executes the full data pipeline.
        """
        # 1. Preprocessing (Clean Data)
        print("Starting data preprocessing...")
        df_cleaned = self.cleaner.clean(df, text_column=DataColumns.TEXT.value)

        # 2. Splitting (Train/Val/Test Split)
        print("Starting data splitting...")
        train_df, val_df, test_df = self.splitter.split(df_cleaned)

        print("Data pipeline completed successfully.")
        return train_df, val_df, test_df
