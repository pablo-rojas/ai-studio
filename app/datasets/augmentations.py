from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from torchvision.transforms import v2

from app.schemas.training import AugmentationStep

TransformStep = AugmentationStep | dict[str, Any]

_TRANSFORM_REGISTRY: dict[str, type[nn.Module]] = {
    "Resize": v2.Resize,
    "CenterCrop": v2.CenterCrop,
    "RandomResizedCrop": v2.RandomResizedCrop,
    "RandomHorizontalFlip": v2.RandomHorizontalFlip,
    "RandomVerticalFlip": v2.RandomVerticalFlip,
    "RandomRotation": v2.RandomRotation,
    "RandomAffine": v2.RandomAffine,
    "RandomPerspective": v2.RandomPerspective,
    "RandomCrop": v2.RandomCrop,
    "ColorJitter": v2.ColorJitter,
    "RandomGrayscale": v2.RandomGrayscale,
    "GaussianBlur": v2.GaussianBlur,
    "RandomAutocontrast": v2.RandomAutocontrast,
    "RandomEqualize": v2.RandomEqualize,
    "RandomPosterize": v2.RandomPosterize,
    "RandomSolarize": v2.RandomSolarize,
    "Normalize": v2.Normalize,
    "ToDtype": v2.ToDtype,
}


def _normalize_step(step: TransformStep) -> tuple[str, dict[str, Any]]:
    """Normalize transform steps from dicts or Pydantic models."""
    if isinstance(step, AugmentationStep):
        return step.name, dict(step.params)
    if isinstance(step, dict):
        raw_name = str(step.get("name", "")).strip()
        if not raw_name:
            raise ValueError("Augmentation step must include a non-empty 'name'.")
        raw_params = step.get("params", {})
        if raw_params is None:
            raw_params = {}
        if not isinstance(raw_params, dict):
            raise ValueError("Augmentation step 'params' must be an object.")
        return raw_name, dict(raw_params)
    raise ValueError("Unsupported augmentation step format.")


def _build_to_image_transform() -> nn.Module:
    """Build image conversion transform with float conversion for Normalize compatibility."""
    return v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])


def _build_transform(name: str, params: dict[str, Any]) -> nn.Module:
    """Build a single transform from config."""
    if name == "ToImage":
        return _build_to_image_transform()

    if name not in _TRANSFORM_REGISTRY:
        raise ValueError(f"Unknown augmentation transform '{name}'.")

    transform_cls = _TRANSFORM_REGISTRY[name]
    if name == "ToDtype":
        dtype_name = str(params.get("dtype", "float32")).lower()
        dtype_mapping: dict[str, torch.dtype] = {
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
        }
        if dtype_name not in dtype_mapping:
            raise ValueError("ToDtype 'dtype' must be one of: float16, float32, float64.")
        scale = bool(params.get("scale", False))
        return transform_cls(dtype_mapping[dtype_name], scale=scale)
    return transform_cls(**params)


def build_augmentation_pipeline(config: Iterable[TransformStep]) -> v2.Compose:
    """Build a torchvision transforms.v2 Compose pipeline."""
    transforms: list[nn.Module] = []
    for raw_step in config:
        name, params = _normalize_step(raw_step)
        transforms.append(_build_transform(name, params))
    return v2.Compose(transforms)
