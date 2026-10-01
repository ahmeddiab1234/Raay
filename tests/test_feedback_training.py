"""Hermetic tests for feedback integration into training data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def test_load_data_concatenates_feedback_onto_train_only(tmp_path: Path):
    """The load-bearing integration: feedback reaches train, and *only* train.

    Adding it to val would corrupt model selection (val drives
    load_best_model_at_end, so every metric downstream would be fitted on the
    overrides); adding it to test would break the frozen split that
    scripts/promote_model.py hashes.
    """
    from omegaconf import OmegaConf

    from raay.training.train import load_data

    for name in ("train", "val", "test"):
        pd.DataFrame(
            {"text": [f"{name} row"], "label": ["positive"], "company": ["acme"]}
        ).to_csv(tmp_path / f"{name}.csv", index=False)
    pd.DataFrame(
        {
            "text": ["validated override"],
            "label": ["negative"],
            "company": ["acme"],
            "source": "customer_service_feedback",
        }
    ).to_csv(tmp_path / "train_feedback.csv", index=False)

    cfg = OmegaConf.create(
        {
            "train_file": str(tmp_path / "train.csv"),
            "val_file": str(tmp_path / "val.csv"),
            "test_file": str(tmp_path / "test.csv"),
            "extra_train_file": str(tmp_path / "train_feedback.csv"),
        }
    )
    train, val, test = load_data(cfg)
    assert len(train) == 2
    assert len(val) == 1
    assert len(test) == 1
    assert "validated override" in set(train["text"])


def test_load_data_without_the_key_is_unchanged(tmp_path: Path):
    """`extra_train_file` is optional; the config may predate the feedback loop."""
    from omegaconf import OmegaConf

    from raay.training.train import load_data

    for name in ("train", "val", "test"):
        pd.DataFrame({"text": ["a row"], "label": ["positive"]}).to_csv(
            tmp_path / f"{name}.csv", index=False
        )
    cfg = OmegaConf.create(
        {
            "train_file": str(tmp_path / "train.csv"),
            "val_file": str(tmp_path / "val.csv"),
            "test_file": str(tmp_path / "test.csv"),
        }
    )
    train, _, _ = load_data(cfg)
    assert len(train) == 1


def test_load_data_tolerates_a_missing_feedback_file(tmp_path: Path):
    """The normal state until a CS tool exists: nothing captured yet."""
    from omegaconf import OmegaConf

    from raay.training.train import load_data

    for name in ("train", "val", "test"):
        pd.DataFrame({"text": ["a row"], "label": ["positive"]}).to_csv(
            tmp_path / f"{name}.csv", index=False
        )
    cfg = OmegaConf.create(
        {
            "train_file": str(tmp_path / "train.csv"),
            "val_file": str(tmp_path / "val.csv"),
            "test_file": str(tmp_path / "test.csv"),
            "extra_train_file": str(tmp_path / "absent.csv"),
        }
    )
    train, _, _ = load_data(cfg)
    assert len(train) == 1


def test_load_data_tolerates_an_empty_feedback_file(tmp_path: Path):
    """`--mode merge` writes a header-only file before anything is captured."""
    from omegaconf import OmegaConf

    from raay.training.train import load_data

    for name in ("train", "val", "test"):
        pd.DataFrame({"text": ["a row"], "label": ["positive"]}).to_csv(
            tmp_path / f"{name}.csv", index=False
        )
    pd.DataFrame(columns=["text", "label"]).to_csv(
        tmp_path / "train_feedback.csv", index=False
    )
    cfg = OmegaConf.create(
        {
            "train_file": str(tmp_path / "train.csv"),
            "val_file": str(tmp_path / "val.csv"),
            "test_file": str(tmp_path / "test.csv"),
            "extra_train_file": str(tmp_path / "train_feedback.csv"),
        }
    )
    train, _, _ = load_data(cfg)
    assert len(train) == 1


def test_the_extra_train_columns_do_not_collide_with_train(tmp_path: Path):
    """The merge carries train.csv's schema first, so a concat cannot double up."""
    from omegaconf import OmegaConf

    from raay.training.train import load_data

    pd.DataFrame(
        {"text": ["a row"], "label": ["positive"], "company": ["acme"]}
    ).to_csv(tmp_path / "train.csv", index=False)
    for name in ("val", "test"):
        pd.DataFrame({"text": ["a row"], "label": ["positive"]}).to_csv(
            tmp_path / f"{name}.csv", index=False
        )
    pd.DataFrame(
        {
            "text": ["override"],
            "label": ["negative"],
            "company": ["acme"],
            "source": "customer_service_feedback",
            "model_label": "positive",
        }
    ).to_csv(tmp_path / "train_feedback.csv", index=False)
    cfg = OmegaConf.create(
        {
            "train_file": str(tmp_path / "train.csv"),
            "val_file": str(tmp_path / "val.csv"),
            "test_file": str(tmp_path / "test.csv"),
            "extra_train_file": str(tmp_path / "train_feedback.csv"),
        }
    )
    train, _, _ = load_data(cfg)
    # Every merged column the model reads must still be single-valued.
    assert not train["text"].duplicated().any()
    assert list(train["label"]) == ["positive", "negative"]
