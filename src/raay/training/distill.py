"""Distill the fine-tuned AraBERT baseline into a compact student.

Run from the repo root (DVC / Kaggle launch it there and every path below is
relative to the repo root):

    uv run python -m raay.training.distill                        # Hydra defaults
    uv run python -m raay.training.distill alpha=0.4 temperature=4.0

Behaviour
    - Loads ``data/processed/{train,val,test}.csv`` (same splits as the
      baseline so results are comparable in MLflow).
    - Applies ``ArabertPreprocessor`` to ``text`` before tokenization.
    - Precomputes teacher logits over the train/val splits.
    - Initialises a ``student_layers``-layer student from the teacher's
      AutoConfig (same tokenizer/vocab) and trains it against a weighted
      ``alpha * CE(student, labels) + (1 - alpha) * T^2 * KL`` objective using
      the HuggingFace Trainer (multiclass sentiment).
    - Logs params / metrics / model artifacts to MLflow (experiment
      ``raay_training``, same schema as the baseline) and checkpoints every
      ``save_steps``.

The GPU training happens on Kaggle; this script is GPU-agnostic (uses whatever
accelerator Trainer finds, CPU included for smoke tests).
"""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, f1_score
from torch.nn import functional as F
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

import mlflow
from raay.data.dialect import add_dialect_column

# pyarabic (a transitive dep of `arabert`) emits noisy SyntaxWarnings on import.
warnings.filterwarnings("ignore", category=SyntaxWarning)
logging.getLogger("transformers").setLevel(logging.WARNING)

try:
    from arabert.preprocess import ArabertPreprocessor
except ImportError:  # pragma: no cover - import path guard
    ArabertPreprocessor = None  # type: ignore[assignment]


def _pinned_pip_requirements() -> list[str]:
    """Pin the framework deps MLflow would infer, skipping any that are missing.

    ``mlflow.transformers.get_default_pip_requirements`` hardcodes ``torchvision``
    whenever torch is present and then imports it to read its version, which
    crashes on Kaggle (no torchvision installed). ``importlib.metadata`` reads
    versions without importing, so missing packages are simply skipped.
    """
    from importlib.metadata import PackageNotFoundError, version

    pinned: list[str] = []
    for package in ("mlflow", "transformers", "torch", "torchvision", "accelerate"):
        try:
            pinned.append(f"{package}=={version(package)}")
        except PackageNotFoundError:
            continue
    return pinned


def _label_map(cfg: DictConfig) -> dict[str, int]:
    return {label: i for i, label in enumerate(cfg.labels)}


def _inverse_label_map(cfg: DictConfig) -> dict[int, str]:
    return {i: label for i, label in enumerate(cfg.labels)}


def load_data(cfg: DictConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(cfg.train_file)
    val = pd.read_csv(cfg.val_file)
    test = pd.read_csv(cfg.test_file)
    return train, val, test


def _preprocess_fn(text: str) -> str:
    if ArabertPreprocessor is not None:
        return _PREPROCESSOR.preprocess(text)  # type: ignore[attr-defined]
    return str(text)


# Module-level slot so the preprocessor is built once after config is known.
_PREPROCESSOR: Any = None


def distill_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
    temperature: float,
) -> torch.Tensor:
    """Weighted hard-label CE + temperature-scaled soft-label KL loss.

    loss = alpha * CE(student, labels)
         + (1 - alpha) * T^2 * KL(softmax(student/T), softmax(teacher/T))

    The ``T**2`` factor keeps the soft target gradient scale equivalent to the
    hard-label CE so ``alpha`` is a meaningful weighting of the two terms.
    """
    loss_hard = F.cross_entropy(student_logits, labels)
    loss_soft = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction="batchmean",
    )
    loss_soft = loss_soft * temperature * temperature
    return alpha * loss_hard + (1.0 - alpha) * loss_soft


def compute_metrics(eval_pred: EvalPrediction, cfg: DictConfig) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    id_to_label = _inverse_label_map(cfg)
    target_names = [id_to_label[i] for i in range(cfg.num_labels)]

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro")),
        "f1_weighted": float(f1_score(labels, preds, average="weighted")),
    }

    per_class: np.ndarray = np.asarray(
        f1_score(
            labels,
            preds,
            average=None,
            labels=list(range(cfg.num_labels)),
        )
    )

    for name, value in zip(target_names, per_class):
        metrics[f"f1_{name}"] = float(value)

    return metrics


class DistillationTrainer(Trainer):
    """Trainer subclass that blends teacher logits with hard labels.

    The batch is expected to carry a ``teacher_logits`` tensor produced by a
    forward pass of the frozen teacher, added by :class:`DistillCollator`. The
    model is trained to minimise ``alpha * CE + (1 - alpha) * T^2 * KL``.
    """

    def __init__(
        self,
        alpha: float = 0.4,
        temperature: float = 4.0,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature

    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Any = None,
    ) -> Any:
        teacher_logits = inputs.pop("teacher_logits")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = distill_loss(
            logits,
            teacher_logits,
            inputs["labels"],
            self.alpha,
            self.temperature,
        )
        return (loss, outputs) if return_outputs else loss


class DistillCollator:
    """Stack ``teacher_logits`` (fixed shape) while padding the token fields.

    The default ``DataCollatorWithPadding`` pads variable-length sequences but
    treats ``teacher_logits`` (batch x num_labels) as another sequence and would
    try to pad it to the token length. We pad only the token/`labels` fields and
    stack the teacher logits as dense tensors.
    """

    def __init__(self, tokenizer: Any) -> None:
        self._base = DataCollatorWithPadding(tokenizer)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        base_features = [
            {k: v for k, v in f.items() if k != "teacher_logits"} for f in features
        ]
        batch = self._base(base_features)
        batch["teacher_logits"] = torch.stack(
            [
                torch.as_tensor(f["teacher_logits"], dtype=torch.float32)
                for f in features
            ]
        )
        return batch


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


def _resolve_cache_path(split: str, cfg: DictConfig) -> Path | None:
    cache_dir = str(getattr(cfg, "teacher_logits_cache", ""))
    if not cache_dir:
        return None
    relative = Path(cache_dir)
    if not relative.is_absolute():
        relative = Path(os.getcwd()) / relative
    return relative / f"{split}_logits.npy"


def _preprocess_batch(texts: list[str]) -> list[str]:
    return [_preprocess_fn(t) for t in texts]


def build_train_dataset(
    df: pd.DataFrame,
    tokenizer: Any,
    label_map: dict[str, int],
    cfg: DictConfig,
    teacher_logits: np.ndarray,
) -> Dataset:
    texts = df["text"].map(_preprocess_fn).tolist()
    labels = df["label"].map(label_map).tolist()

    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=cfg.max_length,
    )
    dataset = Dataset.from_dict(
        {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": labels,
            "teacher_logits": [row.tolist() for row in teacher_logits],
        }
    )
    return dataset


def _resolve_output_dir(cfg: DictConfig) -> str:
    relative_out = Path(cfg.output_dir)
    if not relative_out.is_absolute():
        relative_out = Path(os.getcwd()) / relative_out
    return str(relative_out)


def build_training_args(cfg: DictConfig) -> TrainingArguments:
    # transformers uses a *negative* max_steps as the sentinel for
    # "derive total steps from num_train_epochs". Map our "use epochs" default
    # (max_steps == 0) to -1, as in the baseline trainer.
    max_steps = cfg.max_steps if int(cfg.max_steps) > 0 else -1
    save_model = bool(cfg.save_model)
    return TrainingArguments(
        output_dir=_resolve_output_dir(cfg),
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        num_train_epochs=cfg.epochs,
        max_steps=max_steps,
        weight_decay=cfg.weight_decay,
        warmup_steps=cfg.warmup_steps,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps" if save_model else "no",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        logging_strategy="steps",
        logging_steps=cfg.logging_steps,
        load_best_model_at_end=bool(cfg.load_best_model_at_end) and save_model,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=True,
        seed=cfg.seed,
        report_to=["mlflow"],
        # Keep the precomputed `teacher_logits` column through the training
        # dataloader: with the default remove_unused_columns=True the Trainer
        # strips any column missing from the model's forward() signature, which
        # would silently drop the teacher logits our DistillCollator/loss need.
        remove_unused_columns=False,
    )


class _BestStepLogger(TrainerCallback):
    """Report the step at which the best eval checkpoint was found."""

    def on_evaluate(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if state.best_metric is not None:
            logger.info(
                f"Best step: {state.best_model_checkpoint} "
                f"(metric={state.best_metric:.4f})"
            )


@hydra.main(version_base=None, config_path="../../../configs", config_name="distill")
def main(cfg: DictConfig) -> None:
    global _PREPROCESSOR

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
    if tracking_uri.startswith("file:"):
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    if "mlflow.autolog" in str(OmegaConf.to_container(cfg, resolve=True)):
        mlflow.autolog(disable=True)

    # ---- Guard rail: never silently train on the wrong hardware ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if cfg.require_gpu and device != "cuda":
        raise RuntimeError(
            f"require_gpu=true but no CUDA GPU detected (device={device}). "
            "Distillation must run on a Kaggle GPU session; refusing to train on "
            "CPU. Set require_gpu=false ONLY for deliberate local CPU smoke tests."
        )

    OmegaConf.set_struct(cfg, False)
    cfg.output_dir = _resolve_output_dir(cfg)
    OmegaConf.set_struct(cfg, True)

    save_model = bool(cfg.save_model)

    set_seed(cfg.seed)
    logger.info(OmegaConf.to_yaml(cfg))

    label_map = _label_map(cfg)
    id_to_label = _inverse_label_map(cfg)
    logger.info(f"Label map: {label_map}")

    if ArabertPreprocessor is not None:
        _PREPROCESSOR = ArabertPreprocessor(model_name=cfg.model_name)

    train_df, val_df, test_df = load_data(cfg)
    logger.info(
        f"Loaded splits: train={len(train_df)} val={len(val_df)} test={len(test_df)}"
    )
    test_df = add_dialect_column(test_df, text_col="text")
    dialect_test_path = Path(cfg.output_dir) / "test_dialect.csv"
    dialect_test_path.parent.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(dialect_test_path, index=False)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # ---- Teacher: frozen AraBERT baseline checkpoint ----
    logger.info(f"Loading teacher model from {cfg.teacher_model}")
    teacher = AutoModelForSequenceClassification.from_pretrained(
        cfg.teacher_model, num_labels=cfg.num_labels, id2label=id_to_label
    )
    teacher.to(device)
    teacher.eval()

    teacher_config = AutoConfig.from_pretrained(cfg.teacher_model)

    # ---- Student: same tokenizer/vocab, fewer encoder layers ----
    student_config = AutoConfig.from_pretrained(
        cfg.model_name, num_labels=cfg.num_labels, id2label=id_to_label
    )
    student_config.num_hidden_layers = int(cfg.student_layers)
    student = AutoModelForSequenceClassification.from_config(student_config)
    logger.info(
        f"Student layers: {student_config.num_hidden_layers} "
        f"(teacher: {teacher_config.num_hidden_layers}) vocab: "
        f"{tokenizer.vocab_size}"
    )

    # ---- Precompute teacher logits so the student objective is a pure function ----
    logger.info("Precomputing / loading teacher logits over train/val splits...")
    train_teacher_logits = _cached_teacher_logits(
        teacher, tokenizer, train_df["text"].tolist(), "train", cfg
    )
    val_teacher_logits = _cached_teacher_logits(
        teacher, tokenizer, val_df["text"].tolist(), "val", cfg
    )

    train_ds = build_train_dataset(
        train_df, tokenizer, label_map, cfg, train_teacher_logits
    )
    val_ds = build_train_dataset(val_df, tokenizer, label_map, cfg, val_teacher_logits)

    args = build_training_args(cfg)
    collator = DistillCollator(tokenizer)

    trainer = DistillationTrainer(
        alpha=float(cfg.alpha),
        temperature=float(cfg.temperature),
        model=student,
        args=args,
        data_collator=collator,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=lambda p: compute_metrics(p, cfg),
        callbacks=[_BestStepLogger()],
    )

    mlflow.set_experiment(cfg.experiment_name)
    with mlflow.start_run(run_name="distilled") as run:
        mlflow.log_params(_loggable_params(cfg))
        mlflow.log_param("model_name", "distilled")
        mlflow.log_param("layers", int(cfg.student_layers))
        mlflow.log_param("temperature", float(cfg.temperature))
        mlflow.log_param("alpha", float(cfg.alpha))

        run_id = run.info.run_id
        logger.info(f"MLflow run ID: {run_id}")

        train_result = trainer.train()
        merged = {**train_result.metrics, **trainer.evaluate()}
        logger.info(f"Final validation metrics: {merged}")

        for key, value in merged.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, float(value))

        # Test-set accuracy / f1_macro (comparable to the baseline harness).
        test_metrics = _evaluate_test(
            student, tokenizer, test_df, cfg, save_hard_predictions=True
        )
        for key, value in test_metrics.items():
            mlflow.log_metric(key, float(value))
        logger.info(f"Test metrics: {test_metrics}")

        mlflow.log_param(
            "tokenizer_vocab_size", getattr(tokenizer, "vocab_size", "N/A")
        )
        mlflow.log_param("run_tag", str(cfg.run_tag))

        if save_model:
            final_dir = Path(cfg.output_dir) / "final"
            trainer.save_model(str(final_dir))
            mlflow.transformers.log_model(
                transformers_model={"model": student, "tokenizer": tokenizer},
                task="text-classification",
                artifact_path="model",
                pip_requirements=_pinned_pip_requirements(),
            )
            logger.info(f"Saved student to {final_dir}")


def _evaluate_test(
    student: Any,
    tokenizer: Any,
    test_df: pd.DataFrame,
    cfg: DictConfig,
    save_hard_predictions: bool = True,
) -> dict[str, float]:
    """Return test accuracy / f1 metrics for the student using the eval split.

    Uses the same preprocessing + batching path as :func:`_teacher_logits` so
    the student's test numbers are directly comparable to the baseline harness.
    """
    id_to_label = _inverse_label_map(cfg)
    label_to_id = _label_map(cfg)
    texts = test_df["text"].map(_preprocess_fn).tolist()
    labels = test_df["label"].map(label_to_id).tolist()

    student.eval()
    device = next(student.parameters()).device
    preds_list: list[np.ndarray] = []
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
            preds = torch.argmax(student(**enc).logits, dim=-1).detach().cpu().numpy()
            preds_list.append(preds)
    preds = np.concatenate(preds_list, axis=0)

    metrics: dict[str, float] = {
        "test_accuracy": float(accuracy_score(labels, preds)),
        "test_f1_macro": float(f1_score(labels, preds, average="macro")),
    }

    if save_hard_predictions:
        test_df = test_df.copy()
        test_df["predicted_label"] = [id_to_label[p] for p in preds]
        pred_path = Path(cfg.output_dir) / "test_predictions.csv"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        test_df.to_csv(pred_path, index=False)

    return metrics


def _loggable_params(cfg: DictConfig) -> dict[str, Any]:
    # Flatten the resolved config to scalar params for MLflow, skipping the
    # hydra meta keys and nested groups.
    params: dict[str, Any] = {}
    container: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]
    for key, value in container.items():
        key_str = str(key)
        if key_str.startswith("hydra"):
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            params[key_str] = value
    return params


if __name__ == "__main__":
    main()
