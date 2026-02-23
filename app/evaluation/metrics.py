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
from torchmetrics.detection.mean_ap import MeanAveragePrecision


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    """TorchMetrics bundle used for classification evaluation."""

    accuracy: MulticlassAccuracy
    precision: MulticlassPrecision
    recall: MulticlassRecall
    f1: MulticlassF1Score
    confusion_matrix: MulticlassConfusionMatrix


@dataclass(frozen=True, slots=True)
class ObjectDetectionMAPMetrics:
    """TorchMetrics bundle used for object detection evaluation."""

    mean_average_precision: MeanAveragePrecision


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


def build_object_detection_metrics() -> ObjectDetectionMAPMetrics:
    """Build the default mAP metric set for object detection."""
    return ObjectDetectionMAPMetrics(
        mean_average_precision=MeanAveragePrecision(
            class_metrics=True,
            iou_type="bbox",
            backend="faster_coco_eval",
        )
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


def compute_object_detection_metrics(
    predictions: list[dict[str, Tensor]],
    targets: list[dict[str, Tensor]],
    *,
    class_names: list[str],
) -> dict[str, float | dict[str, float]]:
    """Compute object detection aggregate mAP metrics."""
    metrics = build_object_detection_metrics().mean_average_precision
    metrics.update(predictions, targets)
    raw = metrics.compute()

    map_50 = float(raw["map_50"].item())
    map_50_95 = float(raw["map"].item())

    per_class_ap: dict[str, float] = {}
    class_ids = raw.get("classes")
    class_ap = raw.get("map_per_class")
    if isinstance(class_ids, Tensor) and isinstance(class_ap, Tensor):
        class_ids_tensor = class_ids.reshape(-1)
        class_ap_tensor = class_ap.reshape(-1)
        for index in range(min(class_ids_tensor.numel(), class_ap_tensor.numel())):
            class_id = int(class_ids_tensor[index].item())
            if class_id < 0 or class_id >= len(class_names):
                continue
            per_class_ap[class_names[class_id]] = float(class_ap_tensor[index].item())

    return {
        "mAP_50": map_50,
        "mAP_50_95": map_50_95,
        "per_class_AP": per_class_ap,
    }
