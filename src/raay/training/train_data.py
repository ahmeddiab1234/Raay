"""Dataset loading and tokenization for the baseline training job."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset
from loguru import logger
from omegaconf import DictConfig

from raay.training.training_common import _preprocess_fn


def load_data(cfg: DictConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the three splits and concatenate validated feedback onto train only.

    ``extra_train_file`` holds the QA-approved customer-service overrides from
    Phase 6 step 4 (``raay.data.feedback --mode merge``). It goes on **train
    only, deliberately**: ``val_file`` drives ``load_best_model_at_end``, so
    letting overrides influence model selection would corrupt every metric
    downstream, and ``test_file`` is the frozen split that
    ``scripts/promote_model.py`` hashes against ``dvc.lock`` -- the ``leak`` check
    in the feedback merge is what keeps an overlapping row out of the eval.

    A missing or empty file is the normal state until a CS tool has captured
    anything, so it is not an error: the base splits are simply trained on alone.
    """
    train = pd.read_csv(cfg.train_file)
    val = pd.read_csv(cfg.val_file)
    test = pd.read_csv(cfg.test_file)

    extra_path = cfg.get("extra_train_file", None)
    if extra_path and Path(extra_path).exists():
        extra = pd.read_csv(extra_path)
        if not extra.empty:
            before = len(train)
            train = pd.concat([train, extra], ignore_index=True)
            logger.info(
                f"Added {len(extra)} validated customer-service overrides to train "
                f"({before} -> {len(train)}); these are the production hard negatives"
            )
    return train, val, test


def tokenize_and_encode(
    df: pd.DataFrame, tokenizer: Any, label_map: dict[str, int], cfg: DictConfig
) -> Dataset:
    texts = df["text"].map(_preprocess_fn).tolist()
    labels = df["label"].map(label_map).tolist()

    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=cfg.max_length,
    )
    return Dataset.from_dict(
        {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": labels,
        }
    )
