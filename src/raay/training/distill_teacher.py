"""Teacher logits: batched forward pass plus the on-disk cache."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig

from raay.training.training_common import _preprocess_fn


def _preprocess_batch(texts: list[str]) -> list[str]:
    return [_preprocess_fn(t) for t in texts]


def _resolve_cache_path(split: str, cfg: DictConfig) -> Path | None:
    cache_dir = str(getattr(cfg, "teacher_logits_cache", ""))
    if not cache_dir:
        return None
    relative = Path(cache_dir)
    if not relative.is_absolute():
        relative = Path(os.getcwd()) / relative
    return relative / f"{split}_logits.npy"


def _teacher_logits(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    cfg: DictConfig,
) -> np.ndarray:
    """Run the frozen teacher over ``texts`` and return its logits.

    Feeds the preprocessed text through the teacher in batches so the whole
    train/val split can be distilled without keeping the teacher graph in memory.
    """
    model.eval()
    device = next(model.parameters()).device
    logits_list: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(texts), cfg.batch_size):
            batch = _preprocess_batch(texts[i : i + cfg.batch_size])
            enc = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=cfg.max_length,
                return_tensors="pt",
            ).to(device)
            out = model(**enc).logits
            logits_list.append(out.detach().cpu().numpy())
    return np.concatenate(logits_list, axis=0)


def _cached_teacher_logits(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    split: str,
    cfg: DictConfig,
) -> np.ndarray:
    """Teacher logits for ``split``, cached on disk when cfg.teacher_logits_cache is set.

    Logits depend only on the teacher + data, not on the distillation knobs
    (``alpha``, ``temperature``, ``learning_rate``...), so a hyperparameter sweep
    computes them once and reuses the file across runs. A cache entry is reused
    only when its row count matches ``texts`` (i.e. the split hasn't changed).
    """
    cache_path = _resolve_cache_path(split, cfg)
    if cache_path is not None and cache_path.is_file():
        cached = np.load(cache_path)
        if cached.shape[0] == len(texts):
            logger.info(f"Using cached teacher logits: {cache_path}")
            return cached
        logger.warning(
            f"Stale teacher logits cache ({cached.shape[0]} rows, "
            f"expected {len(texts)}); recomputing."
        )

    logits = _teacher_logits(model, tokenizer, texts, cfg)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, logits)
        logger.info(f"Saved teacher logits cache: {cache_path}")
    return logits
