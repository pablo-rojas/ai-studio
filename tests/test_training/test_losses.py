from __future__ import annotations

import pytest
import torch
from torch import nn

from app.training.losses import FocalLoss, build_loss, list_losses


def test_classification_losses_are_listed() -> None:
    assert list_losses("classification") == [
        "cross_entropy",
        "focal",
        "label_smoothing_cross_entropy",
    ]


def test_detection_losses_are_listed() -> None:
    assert list_losses("object_detection") == ["default"]


def test_build_cross_entropy_with_label_smoothing() -> None:
    loss = build_loss("classification", "cross_entropy", label_smoothing=0.1)

    assert isinstance(loss, nn.CrossEntropyLoss)
    assert loss.label_smoothing == pytest.approx(0.1)


def test_build_focal_loss_and_compute_value() -> None:
    loss = build_loss("classification", "focal", focal_gamma=2.0)
    logits = torch.tensor([[2.0, 0.5], [0.1, 1.7]], dtype=torch.float32)
    targets = torch.tensor([0, 1], dtype=torch.int64)

    value = loss(logits, targets)

    assert isinstance(loss, FocalLoss)
    assert value.item() > 0


def test_build_loss_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unsupported loss"):
        build_loss("classification", "invalid_loss")


def test_build_detection_default_loss_returns_identity() -> None:
    loss = build_loss("object_detection", "default")
    assert isinstance(loss, nn.Identity)
