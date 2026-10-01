"""Teacher/student construction for the distillation run."""

from __future__ import annotations

from typing import Any

from loguru import logger
from omegaconf import DictConfig
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
)


def load_teacher(cfg: DictConfig, device: str, id_to_label: dict[int, str]) -> Any:
    """Load the frozen AraBERT baseline checkpoint used as the teacher."""
    logger.info(f"Loading teacher model from {cfg.teacher_model}")
    teacher = AutoModelForSequenceClassification.from_pretrained(
        cfg.teacher_model, num_labels=cfg.num_labels, id2label=id_to_label
    )
    teacher.to(device)
    teacher.eval()
    return teacher


def build_student(
    cfg: DictConfig,
    teacher_config: Any,
    tokenizer: Any,
    id_to_label: dict[int, str],
) -> Any:
    """Same tokenizer/vocab as the teacher, fewer encoder layers."""
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
    return student
