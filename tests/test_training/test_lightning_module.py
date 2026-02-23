from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.optim import Optimizer

from app.schemas.training import HyperparameterConfig
from app.training.lightning_module import AIStudioModule
from app.training.losses import build_loss


class _ToyDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, images, targets=None):
        if self.training and targets is not None:
            losses = [image.mean() for image in images]
            return {"loss_classifier": self.scale * torch.stack(losses).mean()}
        predictions = []
        for _ in images:
            predictions.append(
                {
                    "boxes": torch.tensor([[1.0, 1.0, 10.0, 10.0]], dtype=torch.float32),
                    "scores": torch.tensor([0.95], dtype=torch.float32),
                    "labels": torch.tensor([0], dtype=torch.int64),
                }
            )
        return predictions


def test_lightning_module_training_step_and_validation_step() -> None:
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 16 * 16, 3))
    module = AIStudioModule(
        task="classification",
        model=model,
        loss_fn=build_loss("classification", "cross_entropy"),
        hyperparameters=HyperparameterConfig(scheduler="none", warmup_epochs=0),
        num_classes=3,
    )
    module.log = lambda *args, **kwargs: None  # type: ignore[method-assign]
    module.log_dict = lambda *args, **kwargs: None  # type: ignore[method-assign]

    batch = (
        torch.randn(4, 3, 16, 16),
        torch.tensor([0, 1, 2, 1], dtype=torch.int64),
    )
    train_loss = module.training_step(batch, batch_idx=0)
    val_loss = module.validation_step(batch, batch_idx=0)
    module.on_train_epoch_end()
    module.on_validation_epoch_end()

    assert train_loss.ndim == 0
    assert val_loss.ndim == 0
    assert train_loss.item() > 0
    assert val_loss.item() > 0


def test_lightning_module_configure_optimizers_with_scheduler() -> None:
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 16 * 16, 2))
    module = AIStudioModule(
        task="classification",
        model=model,
        loss_fn=build_loss("classification", "cross_entropy"),
        hyperparameters=HyperparameterConfig(
            scheduler="cosine",
            warmup_epochs=1,
            max_epochs=5,
        ),
        num_classes=2,
    )

    configured = module.configure_optimizers()

    assert isinstance(configured, dict)
    assert "optimizer" in configured
    assert "lr_scheduler" in configured


def test_lightning_module_configure_optimizers_without_scheduler() -> None:
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 16 * 16, 2))
    module = AIStudioModule(
        task="classification",
        model=model,
        loss_fn=build_loss("classification", "cross_entropy"),
        hyperparameters=HyperparameterConfig(scheduler="none", warmup_epochs=0),
        num_classes=2,
    )

    configured = module.configure_optimizers()

    assert isinstance(configured, Optimizer)


def test_lightning_module_rejects_unsupported_task() -> None:
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 16 * 16, 2))

    with pytest.raises(ValueError, match="not implemented"):
        AIStudioModule(
            task="regression",
            model=model,
            loss_fn=build_loss("classification", "cross_entropy"),
            hyperparameters=HyperparameterConfig(),
            num_classes=2,
        )


def test_lightning_module_detection_training_and_validation_step() -> None:
    module = AIStudioModule(
        task="object_detection",
        model=_ToyDetector(),
        loss_fn=build_loss("object_detection", "default"),
        hyperparameters=HyperparameterConfig(scheduler="none", warmup_epochs=0),
        num_classes=2,
    )
    module.log = lambda *args, **kwargs: None  # type: ignore[method-assign]
    module.log_dict = lambda *args, **kwargs: None  # type: ignore[method-assign]

    batch = (
        [torch.rand(3, 16, 16), torch.rand(3, 16, 16)],
        [
            {
                "boxes": torch.tensor([[1.0, 1.0, 10.0, 10.0]], dtype=torch.float32),
                "labels": torch.tensor([0], dtype=torch.int64),
            },
            {
                "boxes": torch.tensor([[1.0, 1.0, 10.0, 10.0]], dtype=torch.float32),
                "labels": torch.tensor([0], dtype=torch.int64),
            },
        ],
    )

    train_loss = module.training_step(batch, batch_idx=0)
    val_loss = module.validation_step(batch, batch_idx=0)
    module.on_validation_epoch_end()

    assert train_loss.ndim == 0
    assert val_loss.ndim == 0
    assert train_loss.item() > 0
    assert val_loss.item() > 0
