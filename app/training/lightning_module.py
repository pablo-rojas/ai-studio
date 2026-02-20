from __future__ import annotations

from typing import Literal

import pytorch_lightning as pl
from torch import Tensor, nn
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

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
        if task != "classification":
            raise ValueError(f"Task '{task}' is not implemented yet in AIStudioModule.")
        if num_classes < 1:
            raise ValueError("num_classes must be at least 1.")

        self.task = task
        self.model = model
        self.loss_fn = loss_fn
        self.hyperparameters_config = hyperparameters
        self.num_classes = num_classes

        metrics = _build_classification_metrics(num_classes=num_classes)
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")

        self.save_hyperparameters(
            {
                "task": task,
                "num_classes": num_classes,
                **hyperparameters.model_dump(mode="json"),
            }
        )

    def forward(self, images: Tensor) -> Tensor:
        """Run a forward pass and return model outputs."""
        return self.model(images)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Compute and log training loss."""
        del batch_idx
        images, targets = batch
        targets = targets.long()
        outputs = self(images)
        loss = self.loss_fn(outputs, targets)
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

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Compute and log validation loss."""
        del batch_idx
        images, targets = batch
        targets = targets.long()
        outputs = self(images)
        loss = self.loss_fn(outputs, targets)
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

    def on_train_epoch_end(self) -> None:
        """Log train metrics aggregated over the epoch."""
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
        metrics = self.val_metrics.compute()
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.val_metrics.reset()

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


__all__ = ["AIStudioModule"]
