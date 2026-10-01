"""Accumulated statistics for one consumer run."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

import numpy as np


@dataclass
class BatchStats:
    """Accumulated consumer statistics across one ``run``."""

    batches: int = 0
    items_processed: int = 0
    drain_time_sec: float = 0.0
    score_time_sec: float = 0.0
    total_time_sec: float = 0.0
    fill_ratios: list[float] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)

    @property
    def avg_batch_size(self) -> float:
        return self.items_processed / self.batches if self.batches else 0.0

    @property
    def fill_p50(self) -> float:
        return float(np.percentile(self.fill_ratios, 50)) if self.fill_ratios else 0.0

    @property
    def fill_ratio_avg(self) -> float:
        return statistics.mean(self.fill_ratios) if self.fill_ratios else 0.0

    @property
    def inference_work_sec(self) -> float:
        return self.drain_time_sec + self.score_time_sec

    @property
    def work_reviews_per_sec(self) -> float:
        return (
            self.items_processed / self.inference_work_sec
            if self.inference_work_sec
            else 0.0
        )

    @property
    def reviews_per_sec(self) -> float:
        return (
            self.items_processed / self.total_time_sec if self.total_time_sec else 0.0
        )
