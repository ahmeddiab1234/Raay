from dataclasses import dataclass


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing steps."""

    remove_punctuation: bool = True
    remove_numbers: bool = True
    remove_stopwords: bool = True
    normalize_arabic: bool = True


@dataclass
class SplitConfig:
    """Configuration for data splitting."""

    test_size: float = 0.2
    val_size: float = 0.1  # Percentage of train set used for validation
    random_state: int = 42
    stratify_by_column: str = "label"


@dataclass
class PipelineConfig:
    """Main configuration for the data pipeline."""

    preprocessing: PreprocessingConfig = PreprocessingConfig()
    split: SplitConfig = SplitConfig()
