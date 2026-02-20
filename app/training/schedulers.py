from __future__ import annotations

from collections.abc import Sequence

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    MultiStepLR,
    PolynomialLR,
    SequentialLR,
    StepLR,
)

from app.schemas.training import HyperparameterConfig


def build_scheduler(
    optimizer: Optimizer,
    config: HyperparameterConfig,
):
    """Build an LR scheduler from validated hyperparameters.

    Args:
        optimizer: Optimizer instance to schedule.
        config: Hyperparameter configuration.

    Returns:
        A configured LR scheduler, or `None` when no scheduler is requested.
    """
    scheduler = None
    if config.scheduler == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            T_max=config.max_epochs,
        )
    elif config.scheduler == "step":
        scheduler = StepLR(
            optimizer=optimizer,
            step_size=config.step_size,
            gamma=config.gamma,
        )
    elif config.scheduler == "multistep":
        milestones = _resolve_milestones(config)
        scheduler = MultiStepLR(
            optimizer=optimizer,
            milestones=milestones,
            gamma=config.gamma,
        )
    elif config.scheduler == "poly":
        scheduler = PolynomialLR(
            optimizer=optimizer,
            total_iters=config.max_epochs,
            power=config.poly_power,
        )
    elif config.scheduler == "none":
        scheduler = None
    else:
        raise ValueError(f"Unsupported scheduler '{config.scheduler}'.")

    return _apply_optional_warmup(
        optimizer=optimizer,
        scheduler=scheduler,
        warmup_epochs=config.warmup_epochs,
    )


def _apply_optional_warmup(
    *,
    optimizer: Optimizer,
    scheduler,
    warmup_epochs: int,
):
    if warmup_epochs <= 0:
        return scheduler

    warmup = LinearLR(
        optimizer=optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    if scheduler is None:
        return warmup
    return SequentialLR(
        optimizer=optimizer,
        schedulers=[warmup, scheduler],
        milestones=[warmup_epochs],
    )


def _resolve_milestones(config: HyperparameterConfig) -> Sequence[int]:
    if config.milestones:
        milestones = sorted(set(config.milestones))
    else:
        second = min(config.max_epochs - 1, config.step_size * 2)
        milestones = [config.step_size]
        if second > config.step_size:
            milestones.append(second)

    valid = [milestone for milestone in milestones if 0 < milestone < config.max_epochs]
    if not valid:
        raise ValueError("MultiStep scheduler requires at least one milestone < max_epochs.")
    return valid


__all__ = ["build_scheduler"]
