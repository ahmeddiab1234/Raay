"""Hermetic doubles for the micro-batch consumer tests.

No Redis server and no ONNX graph: the fake scorer records call sizes (which is
what the batch-vs-serial win is measured on) and the fake Redis client is a
minimal ``blpop``/``rpush``/``llen``.
"""

from __future__ import annotations


class FakeScorer:
    """Scorer stub recording call sizes; returns labels matching input count."""

    def __init__(self, n_labels: int = 3) -> None:
        self.calls: list[int] = []
        self.n_labels = n_labels

    def score(self, texts: list[str]) -> list[dict]:
        self.calls.append(len(texts))
        return [
            {"label": f"label{i % self.n_labels}", "score": 0.9 + i * 0.01}
            for i in range(len(texts))
        ]


class FakeRedis:
    """Minimal ``blpop``/``rpush``/``llen`` client for RedisQueue unit tests."""

    def __init__(self) -> None:
        self._lists: dict[str, list[str]] = {"reviews": [], "reviews-results": []}

    def blpop(self, key: str, timeout: int) -> tuple | None:
        if key not in self._lists or not self._lists[key]:
            return None
        return (key, self._lists[key].pop(0))

    def rpush(self, key: str, *values: str) -> int:
        self._lists[key].extend(values)
        return len(values)

    def llen(self, key: str) -> int:
        return len(self._lists[key])
