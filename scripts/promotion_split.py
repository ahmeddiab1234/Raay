"""The frozen-split guard: is the test set still the one ``dvc.lock`` pins?

Checked before anything is scored, and checked again as the first entry of the
gate list. The early check turns a drifted split into seconds of work instead of
a full evaluation -- twenty minutes of CPU, and on a contended runner enough
scheduling noise to produce a latency verdict nobody should act on.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
from promotion_types import Config, GateResult, PromotionError, file_md5


def locked_test_split_md5(lock_path: Path) -> str | None:
    """The md5 ``dvc.lock`` pins for the test split, or None if not recorded."""
    if not lock_path.exists():
        return None
    lock = yaml.safe_load(lock_path.read_text()) or {}
    for stage in (lock.get("stages") or {}).values():
        for out in stage.get("outs") or []:
            if str(out.get("path", "")).endswith("test.csv"):
                return out.get("md5")
    return None


def check_frozen_split(cfg: Config) -> GateResult:
    """Refuse to gate against a test set that has drifted from ``dvc.lock``.

    Without this the gate would happily evaluate on whatever happens to be on
    disk, and a "the candidate regressed" verdict could really mean "someone
    regenerated the split".
    """
    locked = locked_test_split_md5(cfg.dvc_lock)
    if locked is None:
        return GateResult(
            "frozen_test_split",
            not cfg.require_frozen_split,
            None,
            cfg.dvc_lock.name,
            "dvc.lock does not pin the test split",
        )
    if not cfg.test_split.exists():
        raise PromotionError(f"{cfg.test_split} is missing; run `dvc repro` first")
    actual = file_md5(cfg.test_split)
    return GateResult(
        "frozen_test_split",
        actual == locked,
        actual,
        locked,
        f"{cfg.test_split} matches dvc.lock"
        if actual == locked
        else "test split drifted",
    )


def load_split(cfg: Config) -> pd.DataFrame:
    if not cfg.test_split.exists():
        raise PromotionError(
            f"{cfg.test_split} is missing; the gate never evaluates on a split it "
            "cannot verify against dvc.lock"
        )
    frame = pd.read_csv(cfg.test_split)
    if cfg.eval_limit:
        frame = frame.head(cfg.eval_limit)
    return frame
