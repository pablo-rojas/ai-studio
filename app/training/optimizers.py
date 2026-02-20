from __future__ import annotations

from collections.abc import Iterable

from torch import nn
from torch.optim import SGD, Adam, AdamW, Optimizer

from app.schemas.training import HyperparameterConfig


def build_optimizer(
    parameters: Iterable[nn.Parameter],
    config: HyperparameterConfig,
) -> Optimizer:
    """Build an optimizer from validated hyperparameters.

    Args:
        parameters: Trainable model parameters.
        config: Hyperparameter configuration.

    Returns:
        Configured torch optimizer.

    Raises:
        ValueError: If optimizer type is unsupported.
    """
    if config.optimizer == "adam":
        return Adam(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    if config.optimizer == "adamw":
        return AdamW(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    if config.optimizer == "sgd":
        return SGD(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            momentum=config.momentum,
        )
    raise ValueError(f"Unsupported optimizer '{config.optimizer}'.")


__all__ = ["build_optimizer"]
