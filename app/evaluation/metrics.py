from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    """TorchMetrics bundle used for classification evaluation."""

    accuracy: MulticlassAccuracy
    precision: MulticlassPrecision
    recall: MulticlassRecall
    f1: MulticlassF1Score
    confusion_matrix: MulticlassConfusionMatrix


def build_classification_metrics(num_classes: int) -> ClassificationMetrics:
    """Build the default metric set for multi-class classification."""
    if num_classes < 1:
        raise ValueError("num_classes must be at least 1.")

    return ClassificationMetrics(
        accuracy=MulticlassAccuracy(num_classes=num_classes),
        precision=MulticlassPrecision(num_classes=num_classes, average="macro"),
        recall=MulticlassRecall(num_classes=num_classes, average="macro"),
        f1=MulticlassF1Score(num_classes=num_classes, average="macro"),
        confusion_matrix=MulticlassConfusionMatrix(num_classes=num_classes),
    )


def compute_classification_metrics(
    predictions: Tensor,
    targets: Tensor,
    *,
    num_classes: int,
) -> dict[str, float | list[list[int]]]:
    """Compute classification aggregate metrics from logits or class IDs."""
    metrics = build_classification_metrics(num_classes)

    return {
        "accuracy": float(metrics.accuracy(predictions, targets).item()),
        "precision_macro": float(metrics.precision(predictions, targets).item()),
        "recall_macro": float(metrics.recall(predictions, targets).item()),
        "f1_macro": float(metrics.f1(predictions, targets).item()),
        "confusion_matrix": metrics.confusion_matrix(predictions, targets).int().tolist(),
    }
