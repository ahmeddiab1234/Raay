"""Kaggle sweep definitions and the Hydra overrides they need.

Runs live on Kaggle GPUs; the processed splits arrive as a Dataset snapshot under
``/kaggle/input/<dataset>/``, so ``DATA_ROOT`` (legacy ``KAGGLE_DATA_DIR``) points
the runs at it instead of the repo-relative defaults.

``extra_train_file`` (Phase 6 step 4's merged CS overrides) is included **only**
when the snapshot actually carries it. Requiring it would make a snapshot without
feedback unbuildable for no reason -- and unlike ``train.csv`` it is not something
every run can regenerate.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from raay.config.env import env_str
from raay.enums.constants import DefaultPaths, EnvVar

# (learning_rate, batch_size)
SWEEP = [
    (2e-5, 16),
    (3e-5, 16),
    (5e-5, 16),
    (3e-5, 32),
    (2e-5, 32),
]

# (learning_rate, batch_size, alpha, temperature)
DISTILL_SWEEP = [
    (2e-5, 16, 0.4, 4.0),
    (3e-5, 16, 0.4, 4.0),
    (5e-5, 16, 0.4, 4.0),
    (3e-5, 16, 0.3, 3.0),
    (3e-5, 16, 0.5, 4.0),
    (3e-5, 32, 0.4, 4.0),
]


def data_overrides() -> list[str]:
    """Hydra overrides pointing at the processed-DataFrame snapshot."""
    root = env_str(EnvVar.KAGGLE_DATA_DIR) or env_str(EnvVar.DATA_ROOT)
    if not root:
        return []
    root_path = Path(root)
    for name in ("train", "val", "test"):
        cand = root_path / f"{name}.csv"
        if not cand.exists():
            raise FileNotFoundError(
                f"{cand} not found. Point {EnvVar.DATA_ROOT.value} at the dir "
                "containing the processed train/val/test.csv snapshot."
            )
    overrides = [
        f"train_file={root_path / 'train.csv'}",
        f"val_file={root_path / 'val.csv'}",
        f"test_file={root_path / 'test.csv'}",
    ]
    # `.name` on a str-Enum member is the member name ("FEEDBACK_MERGED"), so the
    # filename comes off the value instead.
    feedback = root_path / Path(DefaultPaths.FEEDBACK_MERGED.value).name
    if feedback.exists():
        overrides.append(f"extra_train_file={feedback}")
    else:
        logger.info(
            f"No {feedback} in the snapshot; training on the base splits alone "
            "(no validated customer-service overrides yet)."
        )
    return overrides


def teacher_dir(teacher_dir_arg: str | None) -> str | None:
    """Resolve the teacher checkpoint dir (distill only).

    ``--teacher-dir`` wins over ``KAGGLE_TEACHER_DIR``; if neither is set the
    Hydra config default (``models/baseline/final``) is used, which is the
    correct value for standalone/local runs.
    """
    return teacher_dir_arg or env_str(EnvVar.KAGGLE_TEACHER_DIR) or None


def train_command(lr: float, bs: int, *, save_model: bool, run_tag: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "raay.training.train",
        f"learning_rate={lr}",
        f"batch_size={bs}",
        f"save_model={'true' if save_model else 'false'}",
        f"run_tag={run_tag}",
        f"hydra.run.dir=./{DefaultPaths.OUTPUT_DIR_TRAIN.value}",
        *data_overrides(),
    ]


def distill_command(
    lr: float,
    bs: int,
    alpha: float,
    temperature: float,
    *,
    save_model: bool,
    run_tag: str,
    teacher: str | None,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "raay.training.distill",
        f"learning_rate={lr}",
        f"batch_size={bs}",
        f"alpha={alpha}",
        f"temperature={temperature}",
        f"save_model={'true' if save_model else 'false'}",
        f"run_tag={run_tag}",
        f"hydra.run.dir=./{DefaultPaths.OUTPUT_DIR_DISTILL.value}",
        *data_overrides(),
    ]
    if teacher:
        cmd.append(f"teacher_model={teacher}")
    return cmd
