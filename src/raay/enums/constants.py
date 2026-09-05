from enum import Enum


class EnvVar(str, Enum):
    """Environment variable names (resolved from the repo ``.env`` or the process env)."""

    MLFLOW_TRACKING_URI = "MLFLOW_TRACKING_URI"
    MLFLOW_ALLOW_FILE_STORE = "MLFLOW_ALLOW_FILE_STORE"
    DATA_ROOT = "DATA_ROOT"
    KAGGLE_DATA_DIR = "KAGGLE_DATA_DIR"
    KAGGLE_TEACHER_DIR = "KAGGLE_TEACHER_DIR"


class Experiments(str, Enum):
    """MLflow experiment names."""

    PREPROCESSING = "raay_preprocessing"
    TRAINING = "raay_training"


class Models(str, Enum):
    """Model identifiers (HF repo id or MLflow registry name)."""

    TEACHER = "aubmindlab/bert-base-arabertv02"
    REGISTERED_BASELINE = "ArabicSentiment"
    REGISTERED_DISTILLED = "ArabicSentimentDistilled"


class DefaultPaths(str, Enum):
    """Repo-relative default paths for data, models, reports and outputs.

    Entries are relative to the repo root. Kaggle sessions override the data /
    teacher locations through ``DATA_ROOT`` and ``KAGGLE_TEACHER_DIR``.
    """

    RAW_DATA = "data/raw/Final_Data.csv"
    INTERIM_DATA = "data/interim/normalized.csv"
    PROCESSED_DATA = "data/processed"
    TRAIN_SPLIT = "data/processed/train.csv"
    VAL_SPLIT = "data/processed/val.csv"
    TEST_SPLIT = "data/processed/test.csv"
    PREPROCESS_METRICS = "reports/preprocess_metrics.json"
    EVAL_BASELINE = "reports/eval_baseline.json"
    EVAL_DISTILLED = "reports/eval_distilled.json"
    BASELINE_MODEL = "models/baseline/final"
    DISTILLED_MODEL = "models/distilled/final"
    TEACHER_LOGITS_CACHE = "models/distilled/teacher_logits"
    LOCAL_MLRUNS = "mlflow/mlruns"
    OUTPUT_DIR_TRAIN = "outputs/train"
    OUTPUT_DIR_DISTILL = "outputs/distill"
    CONFIG_TRAIN = "configs/train.yaml"
    CONFIG_DISTILL = "configs/distill.yaml"
    PARAMS = "params.yaml"
    LOGS = "logs"


class SplitFileNames(str, Enum):
    """Output filenames written under ``data/processed``."""

    TRAIN = "train.csv"
    VALIDATION = "val.csv"
    TEST = "test.csv"


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
