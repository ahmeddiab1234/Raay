from enum import Enum


class DataColumns(str, Enum):
    """Enum for dataset column names."""

    TEXT = "text"
    LABEL = "label"
    DIALECT = "dialect"
    TEXT_LENGTH = "text_length"


class SplitNames(str, Enum):
    """Enum for split names."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
