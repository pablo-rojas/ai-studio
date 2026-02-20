from __future__ import annotations

import pytest
from torch import nn
from torch.optim import SGD, Adam, AdamW

from app.schemas.training import HyperparameterConfig
from app.training.optimizers import build_optimizer


@pytest.mark.parametrize(
    ("optimizer_name", "optimizer_type"),
    [
        ("adam", Adam),
        ("adamw", AdamW),
        ("sgd", SGD),
    ],
)
def test_build_optimizer_supports_expected_types(
    optimizer_name: str,
    optimizer_type: type,
) -> None:
    model = nn.Linear(8, 3)
    config = HyperparameterConfig(optimizer=optimizer_name)

    optimizer = build_optimizer(model.parameters(), config)

    assert isinstance(optimizer, optimizer_type)


def test_build_optimizer_rejects_unknown_type() -> None:
    model = nn.Linear(8, 3)
    config = HyperparameterConfig().model_copy(update={"optimizer": "invalid"})

    with pytest.raises(ValueError, match="Unsupported optimizer"):
        build_optimizer(model.parameters(), config)
