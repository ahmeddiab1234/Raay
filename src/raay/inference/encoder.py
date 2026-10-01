"""Pooled sentence embeddings from the fine-tuned AraBERT encoder.

The ONNX graph cannot do this job: it takes ``input_ids``/``attention_mask``
and emits 3 logits, nothing else. Drift features need the pooled hidden state,
so this loads the encoder the classification head sits on top of
(``AutoModel``, not ``BertForSequenceClassification`` -- the ``classifier.*``
weights come back as unexpected keys and are dropped).

This is the ~2 GB / ~2 min part of the nightly job, which is why the builder
that owns it is only constructed for the two modes that read it.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from transformers import AutoTokenizer

from raay.enums.constants import DefaultPaths, Models
from raay.inference.text_features import _preprocess

warnings.filterwarnings("ignore", category=SyntaxWarning)

# Files whose presence means "this directory holds real encoder weights".
# ``model.safetensors.index.json`` ends in ``.json`` and is correctly excluded.
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")
POOLERS = ("mean", "cls")


def encoder_weights_present(model_dir: str) -> bool:
    """Whether ``model_dir`` actually holds fine-tuned encoder weights.

    Only the three JSON files under ``models/baseline/final/`` are DVC-tracked,
    so a fresh clone has a config and a tokenizer but no weights -- and
    ``AutoModel.from_pretrained`` on that directory raises an error about
    missing weights rather than about the thing the operator actually needs to
    know. Checked up front so the failure is actionable.
    """
    directory = Path(model_dir)
    if not directory.is_dir():
        return False
    return any(p.is_file() and p.suffix in WEIGHT_SUFFIXES for p in directory.iterdir())


class AraBertEmbedder:
    """Pooled sentence embeddings from the fine-tuned AraBERT encoder.

    Pooling defaults to **mean over non-padding tokens**. ``cls`` is available
    and is what the classifier itself reads, but the mean is far less
    sensitive to truncation at ``max_length``, and a review truncated to 128
    subwords would otherwise be represented mostly by its first tokens.
    """

    def __init__(
        self,
        model_dir: str = DefaultPaths.BASELINE_MODEL.value,
        model_name: str = Models.TEACHER.value,
        max_length: int = 128,
        batch_size: int = 32,
        pooling: str = "mean",
    ) -> None:
        if pooling not in POOLERS:
            raise ValueError(f"pooling must be one of {POOLERS}, got {pooling!r}")
        self.model_dir = model_dir
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.pooling = pooling
        self._model: Any = None
        self._tokenizer: Any = None
        self._torch: Any = None
        self._hidden_size: int | None = None

    @property
    def tokenizer(self) -> Any:
        self._ensure_loaded()
        return self._tokenizer

    @property
    def hidden_size(self) -> int:
        self._ensure_loaded()
        assert self._hidden_size is not None
        return self._hidden_size

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if not encoder_weights_present(self.model_dir):
            raise FileNotFoundError(
                f"no encoder weights in {self.model_dir} "
                f"(expected a *{', *'.join(WEIGHT_SUFFIXES)} file). The embedding "
                f"drift features need the fine-tuned AraBERT encoder, not just the "
                f"tokenizer; only the JSON files there are DVC-tracked, so a fresh "
                f"clone cannot compute them. Restore the weights, or point "
                f"--tokenizer-dir at a directory that has them."
            )
        import torch
        from transformers import AutoConfig, AutoModel

        self._torch = torch
        self._model = AutoModel.from_pretrained(self.model_dir, local_files_only=True)
        self._model.eval()
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir, local_files_only=True
        )
        self._hidden_size = int(AutoConfig.from_pretrained(self.model_dir).hidden_size)
        logger.info(
            f"Embedding encoder loaded from {self.model_dir} "
            f"hidden_size={self._hidden_size} pooling={self.pooling} "
            f"max_length={self.max_length}"
        )

    def _pool(self, last_hidden_state: Any, attention_mask: Any) -> np.ndarray:
        if self.pooling == "cls":
            return last_hidden_state[:, 0, :].float().cpu().numpy()
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        # A fully padded row cannot happen (the tokenizer always emits at least
        # one real token), but dividing by 0 would poison the panel silently.
        counts = mask.sum(dim=1).clamp(min=1.0)
        return (summed / counts).float().cpu().numpy()

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """``(len(texts), hidden_size)`` float64 embeddings, preprocessed first."""
        self._ensure_loaded()
        torch = self._torch
        if len(texts) == 0:
            return np.zeros((0, self.hidden_size), dtype=np.float64)
        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = [
                    _preprocess(t, self.model_name)
                    for t in texts[i : i + self.batch_size]
                ]
                enc = self._tokenizer(
                    batch,
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                out = self._model(**enc)
                chunks.append(self._pool(out.last_hidden_state, enc["attention_mask"]))
        return np.concatenate(chunks, axis=0).astype(np.float64)
