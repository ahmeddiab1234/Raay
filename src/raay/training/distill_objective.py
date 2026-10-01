"""Distillation objective, trainer subclass and batch collator."""

from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F
from transformers import DataCollatorWithPadding, Trainer


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
    try to pad it to the token length. We pad only the token/`labels`` fields and
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
