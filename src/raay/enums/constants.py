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


class Dialects(str, Enum):
    """Coarse Arabic dialect buckets used by the heuristic detector."""

    MSA = "msa"
    EGYPTIAN = "egyptian"
    GULF = "gulf"
    LEVANTINE = "levantine"
    MAGHREBI = "maghrebi"
    ARABIZI = "arabizi"
