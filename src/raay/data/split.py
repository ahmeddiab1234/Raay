from pathlib import Path

import pandas as pd
import yaml
from loguru import logger
from sklearn.model_selection import train_test_split

import mlflow


def load_config(config_path: str = "params.yaml") -> dict:
    """Load split configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def perform_stratified_split(
    df: pd.DataFrame, test_size: float, val_size: float, random_state: int
):
    """
    Splits DataFrame into train, val, test datasets stratified by 'label'.
    """
    logger.info(f"Total dataset size: {len(df)}")

    # First split: Train+Val vs Test
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=random_state
    )

    # Calculate the proportion of the validation set relative to the training+val set
    # val_size is relative to the *entire* dataset according to standard convention,
    # but since train_test_split is applied to train_val_df, we adjust the proportion.
    # actual_val_fraction = val_size / (1.0 - test_size)
    actual_val_fraction = val_size / (1.0 - test_size)

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=actual_val_fraction,
        stratify=train_val_df["label"],
        random_state=random_state,
    )

    return train_df, val_df, test_df


def main():
    config = load_config()
    split_config = config.get("split", {})
    test_size = split_config.get("test_size", 0.2)
    val_size = split_config.get("val_size", 0.1)
    random_state = split_config.get("random_state", 42)

    input_path = Path("data/interim/normalized.csv")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading normalized data from {input_path}")
    df = pd.read_csv(input_path)

    mlflow.set_experiment("raay_preprocessing")
    with mlflow.start_run(run_name="data_splitting"):
        mlflow.log_params(split_config)

        logger.info(
            f"Splitting data (test_size={test_size}, val_size={val_size}, random_state={random_state})..."
        )
        train_df, val_df, test_df = perform_stratified_split(
            df, test_size, val_size, random_state
        )

        logger.info(f"Train size: {len(train_df)}")
        logger.info(f"Val size: {len(val_df)}")
        logger.info(f"Test size: {len(test_df)}")

        train_path = output_dir / "train.csv"
        val_path = output_dir / "val.csv"
        test_path = output_dir / "test.csv"

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)

        logger.info(f"Saved splits to {output_dir}")

        # Log metrics to mlflow
        mlflow.log_metrics(
            {
                "train_size": len(train_df),
                "val_size": len(val_df),
                "test_size": len(test_df),
            }
        )


if __name__ == "__main__":
    main()
