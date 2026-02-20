from __future__ import annotations

import pytest
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, MultiStepLR, SequentialLR

from app.schemas.training import HyperparameterConfig
from app.training.schedulers import build_scheduler


def test_build_scheduler_returns_none_when_disabled() -> None:
    model = nn.Linear(8, 2)
    optimizer = Adam(model.parameters(), lr=1e-3)
    config = HyperparameterConfig(scheduler="none", warmup_epochs=0)

    scheduler = build_scheduler(optimizer, config)

    assert scheduler is None


def test_build_scheduler_supports_cosine_with_warmup() -> None:
    model = nn.Linear(8, 2)
    optimizer = Adam(model.parameters(), lr=1e-3)
    config = HyperparameterConfig(
        scheduler="cosine",
        warmup_epochs=2,
        max_epochs=10,
    )

    scheduler = build_scheduler(optimizer, config)

    assert isinstance(scheduler, SequentialLR)
    assert isinstance(scheduler._schedulers[0], LinearLR)
    assert isinstance(scheduler._schedulers[1], CosineAnnealingLR)


def test_build_scheduler_supports_multistep() -> None:
    model = nn.Linear(8, 2)
    optimizer = Adam(model.parameters(), lr=1e-3)
    config = HyperparameterConfig(
        scheduler="multistep",
        milestones=[2, 4, 6],
        warmup_epochs=0,
        max_epochs=12,
    )

    scheduler = build_scheduler(optimizer, config)

    assert isinstance(scheduler, MultiStepLR)


def test_build_scheduler_rejects_invalid_multistep_milestones() -> None:
    model = nn.Linear(8, 2)
    optimizer = Adam(model.parameters(), lr=1e-3)
    config = HyperparameterConfig(
        scheduler="multistep",
        milestones=[20],
        warmup_epochs=0,
        max_epochs=10,
    )

    with pytest.raises(ValueError, match="at least one milestone"):
        build_scheduler(optimizer, config)
