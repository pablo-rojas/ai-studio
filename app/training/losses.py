from __future__ import annotations

from typing import Literal

import torch.nn.functional as functional
from torch import Tensor, nn

TaskType = Literal[
    "classification",
    "anomaly_detection",
    "object_detection",
    "oriented_object_detection",
    "segmentation",
    "instance_segmentation",
    "regression",
]


class FocalLoss(nn.Module):
    """Multi-class focal loss for class-imbalanced classification."""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float | None = None,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be >= 0.")
        if alpha is not None and alpha <= 0:
            raise ValueError("alpha must be > 0 when provided.")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be one of: 'mean', 'sum', 'none'.")
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute focal loss from logits and class IDs."""
        log_probs = functional.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        targets = targets.long()
        gathered_log_probs = log_probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
        gathered_probs = probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)

        focal_weights = (1.0 - gathered_probs).pow(self.gamma)
        losses = -focal_weights * gathered_log_probs

        if self.alpha is not None:
            losses = losses * self.alpha

        if self.reduction == "mean":
            return losses.mean()
        if self.reduction == "sum":
            return losses.sum()
        return losses


def list_losses(task: TaskType) -> list[str]:
    """List available loss names for a task."""
    if task == "classification":
        return ["cross_entropy", "focal", "label_smoothing_cross_entropy"]
    if task == "object_detection":
        return ["default"]
    return []


def build_loss(
    task: TaskType,
    loss_name: str,
    *,
    label_smoothing: float = 0.0,
    focal_gamma: float = 2.0,
) -> nn.Module:
    """Instantiate a task-specific loss module."""
    if task == "object_detection":
        if loss_name != "default":
            raise ValueError(f"Unsupported loss '{loss_name}' for task '{task}'.")
        # Torchvision detection models compute their losses internally.
        return nn.Identity()

    if task != "classification":
        raise ValueError(f"Task '{task}' losses are not implemented yet.")

    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if loss_name == "focal":
        return FocalLoss(gamma=focal_gamma)
    if loss_name == "label_smoothing_cross_entropy":
        smoothed = label_smoothing if label_smoothing > 0 else 0.1
        return nn.CrossEntropyLoss(label_smoothing=smoothed)
    raise ValueError(f"Unsupported loss '{loss_name}' for task '{task}'.")
