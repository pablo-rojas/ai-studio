from __future__ import annotations

from typing import Any, Literal

import pytorch_lightning as pl
import torch
from torch import Tensor, nn
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from app.schemas.training import HyperparameterConfig
from app.training.optimizers import build_optimizer
from app.training.schedulers import build_scheduler

TaskType = Literal[
    "classification",
    "anomaly_detection",
    "object_detection",
    "oriented_object_detection",
    "segmentation",
    "instance_segmentation",
    "regression",
]

ClassificationBatch = tuple[Tensor, Tensor]
DetectionTarget = dict[str, Tensor]
DetectionBatch = tuple[list[Tensor], list[DetectionTarget]]


class AIStudioModule(pl.LightningModule):
    """Generic Lightning module used by task-specific training runs."""

    def __init__(
        self,
        *,
        task: TaskType,
        model: nn.Module,
        loss_fn: nn.Module,
        hyperparameters: HyperparameterConfig,
        num_classes: int,
    ) -> None:
        super().__init__()
        if task not in {"classification", "object_detection"}:
            raise ValueError(f"Task '{task}' is not implemented yet in AIStudioModule.")
        if num_classes < 1:
            raise ValueError("num_classes must be at least 1.")

        self.task = task
        self.model = model
        self.loss_fn = loss_fn
        self.hyperparameters_config = hyperparameters
        self.num_classes = num_classes

        self.train_metrics: MetricCollection | None = None
        self.val_metrics: MetricCollection | None = None
        self.val_detection_map: MeanAveragePrecision | None = None

        if task == "classification":
            metrics = _build_classification_metrics(num_classes=num_classes)
            self.train_metrics = metrics.clone(prefix="train_")
            self.val_metrics = metrics.clone(prefix="val_")
        else:
            self.val_detection_map = MeanAveragePrecision(
                class_metrics=True,
                iou_type="bbox",
                backend="faster_coco_eval",
            )

        self.save_hyperparameters(
            {
                "task": task,
                "num_classes": num_classes,
                **hyperparameters.model_dump(mode="json"),
            }
        )

    def forward(self, images: Tensor | list[Tensor]) -> Tensor | list[dict[str, Tensor]]:
        """Run a forward pass and return model outputs."""
        return self.model(images)

    def training_step(self, batch: ClassificationBatch | DetectionBatch, batch_idx: int) -> Tensor:
        """Compute and log training loss."""
        del batch_idx
        if self.task == "classification":
            images, targets = batch  # type: ignore[assignment]
            targets = targets.long()
            outputs = self(images)
            if not isinstance(outputs, Tensor):
                raise ValueError("Classification model output must be a tensor.")
            loss = self.loss_fn(outputs, targets)
            if self.train_metrics is None:
                raise ValueError("Classification train metrics are not initialized.")
            self.train_metrics.update(outputs, targets)
            self.log(
                "train_loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=targets.shape[0],
            )
            return loss

        images, targets = batch  # type: ignore[assignment]
        images = [image.to(self.device) for image in images]
        targets = [_move_detection_target_to_device(target, self.device) for target in targets]
        loss_dict = self.model(images, targets)
        if not isinstance(loss_dict, dict):
            raise ValueError("Detection model must return a dictionary of losses in train mode.")
        loss = _sum_loss_dict(loss_dict)
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(images),
        )
        return loss

    def validation_step(
        self,
        batch: ClassificationBatch | DetectionBatch,
        batch_idx: int,
    ) -> Tensor:
        """Compute and log validation loss."""
        del batch_idx
        if self.task == "classification":
            images, targets = batch  # type: ignore[assignment]
            targets = targets.long()
            outputs = self(images)
            if not isinstance(outputs, Tensor):
                raise ValueError("Classification model output must be a tensor.")
            loss = self.loss_fn(outputs, targets)
            if self.val_metrics is None:
                raise ValueError("Classification validation metrics are not initialized.")
            self.val_metrics.update(outputs, targets)
            self.log(
                "val_loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=targets.shape[0],
            )
            return loss

        images, targets = batch  # type: ignore[assignment]
        images = [image.to(self.device) for image in images]
        targets = [_move_detection_target_to_device(target, self.device) for target in targets]

        was_training = self.model.training
        self.model.train()
        loss_dict = self.model(images, targets)
        if not isinstance(loss_dict, dict):
            raise ValueError("Detection model must return a dictionary of losses in train mode.")
        loss = _sum_loss_dict(loss_dict)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(images),
        )

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(images)
        if was_training:
            self.model.train()

        if not isinstance(predictions, list):
            raise ValueError("Detection model must return prediction list in eval mode.")
        if self.val_detection_map is None:
            raise ValueError("Detection mAP metric is not initialized.")
        self.val_detection_map.update(
            [_detach_detection_target(prediction) for prediction in predictions],
            [_detach_detection_target(target) for target in targets],
        )
        return loss

    def on_train_epoch_end(self) -> None:
        """Log train metrics aggregated over the epoch."""
        if self.task != "classification":
            return
        if self.train_metrics is None:
            raise ValueError("Classification train metrics are not initialized.")
        metrics = self.train_metrics.compute()
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.train_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        """Log validation metrics aggregated over the epoch."""
        if self.task == "classification":
            if self.val_metrics is None:
                raise ValueError("Classification validation metrics are not initialized.")
            metrics = self.val_metrics.compute()
            self.log_dict(
                metrics,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            self.val_metrics.reset()
            return

        if self.val_detection_map is None:
            raise ValueError("Detection mAP metric is not initialized.")
        metrics = self.val_detection_map.compute()
        map_50 = metrics.get("map_50")
        map_50_95 = metrics.get("map")
        mar_100 = metrics.get("mar_100")
        if map_50 is None or map_50_95 is None or mar_100 is None:
            raise ValueError("Detection mAP metric returned an unexpected payload.")
        self.log_dict(
            {
                "val_mAP_50": map_50,
                "val_mAP_50_95": map_50_95,
                "val_precision": map_50,
                "val_recall": mar_100,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.val_detection_map.reset()

    def configure_optimizers(self):
        """Build optimizer and scheduler from hyperparameters."""
        optimizer = build_optimizer(
            parameters=self.parameters(),
            config=self.hyperparameters_config,
        )
        scheduler = build_scheduler(optimizer=optimizer, config=self.hyperparameters_config)
        if scheduler is None:
            return optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


def _build_classification_metrics(*, num_classes: int) -> MetricCollection:
    return MetricCollection(
        {
            "accuracy": MulticlassAccuracy(num_classes=num_classes),
            "precision": MulticlassPrecision(num_classes=num_classes, average="macro"),
            "recall": MulticlassRecall(num_classes=num_classes, average="macro"),
            "f1": MulticlassF1Score(num_classes=num_classes, average="macro"),
        }
    )


def _sum_loss_dict(loss_dict: dict[str, Any]) -> Tensor:
    losses: list[Tensor] = []
    for value in loss_dict.values():
        if isinstance(value, Tensor):
            losses.append(value)
    if not losses:
        raise ValueError("Detection model did not return any tensor losses.")
    return torch.stack([loss for loss in losses]).sum()


def _move_detection_target_to_device(
    target: DetectionTarget,
    device: torch.device,
) -> DetectionTarget:
    return {key: value.to(device) for key, value in target.items()}


def _detach_detection_target(target: DetectionTarget) -> DetectionTarget:
    detached: DetectionTarget = {}
    for key, value in target.items():
        detached[key] = value.detach().cpu()
    return detached


__all__ = ["AIStudioModule"]
