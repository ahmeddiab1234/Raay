"""Shared hermetic test doubles.

Lives here rather than in one test module because both ``test_drift_features.py``
and ``test_batch_score.py`` need a fake tokenizer/encoder: the engineered
drift features are tested at the feature level and at the ``drift_check``
integration level, and duplicating the doubles would let the two drift apart.

Nothing here touches ``data/``, ``models/`` or a real graph (AGENTS.md rule).
"""

import sys
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

pytest_plugins = ("canary_helpers", "cd_helpers", "deploy_staging_helpers")

HIDDEN = 8


def identity(text: str) -> str:
    """Preprocessor stand-in.

    pyarabic normalizes in ways unrelated to OOV -- it even collapses runs of
    repeated letters -- so leaving it in would make the OOV arithmetic tests
    assertions about pyarabic.
    """
    return text


class FakeTokenizer:
    """Character-level tokenizer with a real ``unk_token_id``.

    ``vocab`` maps characters to ids; anything not in it becomes
    ``unk_token_id``, which lets a test state "this review is half
    unspellable" without a real WordPiece vocabulary. Characters above
    ``ascii_high`` are skipped, standing in for what a subword tokenizer does
    with multi-byte text.
    """

    unk_token_id = 0

    def __init__(self, vocab: dict[str, int], ascii_high: int = 128) -> None:
        self.vocab = vocab
        self.ascii_high = ascii_high

    def __call__(
        self,
        batch: list[str],
        add_special_tokens: bool = False,
        truncation: bool = False,
    ) -> dict[str, list[list[int]]]:
        return {
            "input_ids": [
                [
                    self.vocab.get(ch, self.unk_token_id)
                    for ch in text
                    if ord(ch) < self.ascii_high
                ]
                for text in batch
            ]
        }


class FakeEmbedder:
    """Deterministic embeddings whose leading dims are readable by hand.

    PC1 of this fake is review length and PC2 is the non-ASCII share, so a test
    can state the expected drift direction instead of "some number moved".
    """

    pooling = "mean"

    def __init__(self, hidden_size: int = HIDDEN) -> None:
        self.hidden_size = hidden_size
        self.calls = 0

    def embed(self, texts: Any) -> np.ndarray:
        texts = list(texts)
        self.calls += 1
        out = np.zeros((len(texts), self.hidden_size), dtype=np.float64)
        for i, text in enumerate(texts):
            text = str(text)
            non_ascii = sum(1 for ch in text if ord(ch) > 127) / max(len(text), 1)
            out[i, 0] = len(text)
            out[i, 1] = non_ascii
            out[i, 2:] = (len(text) % 3) + np.arange(self.hidden_size - 2) / 10
        return out


class FakeScorer:
    """Matches the ``Scorer`` interface (label_columns + score) without ORT."""

    label_columns: ClassVar[list[str]] = ["positive", "negative", "neutral"]
    batch_size = 64

    def score(self, texts: list[str]) -> np.ndarray:
        n = len(texts)
        probs = np.tile([0.8, 0.15, 0.05], (n, 1))
        probs[:, 1] += np.arange(n) * 0.001
        probs[:, 2] -= np.arange(n) * 0.001
        return probs


def ascii_vocab(size: int = 26) -> dict[str, int]:
    return {chr(ord("a") + i): i + 1 for i in range(size)}


def scored_frame(n: int = 60, seed: int = 7, filler: str = "") -> pd.DataFrame:
    """A frame shaped like a real scored output: text plus probs and argmax."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "text": [f"review number {i} with some latin{filler}" for i in range(n)],
            "positive": rng.random(n),
            "negative": rng.random(n),
            "neutral": rng.random(n),
            "predicted_label": ["positive"] * n,
            "predicted_score": rng.random(n),
        }
    )
