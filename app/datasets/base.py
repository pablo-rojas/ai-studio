from __future__ import annotations

from pathlib import Path

import pytorch_lightning as pl
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from app.schemas.dataset import DatasetImage, DatasetMetadata
from app.schemas.training import AugmentationConfig


class ClassificationImageDataset(Dataset[tuple[Tensor, Tensor]]):
    """Dataset wrapper for image classification samples."""

    def __init__(
        self,
        *,
        samples: list[tuple[Path, int]],
        transform: v2.Transform | None = None,
    ) -> None:
        self.samples = samples
        self.transform = transform
        self._fallback_transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

    def __len__(self) -> int:
        """Return sample count."""
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """Load one image and return `(image_tensor, class_id)`."""
        image_path, class_id = self.samples[index]
        with Image.open(image_path) as image:
            image_data = image.convert("RGB")

        if self.transform is not None:
            transformed = self.transform(image_data)
        else:
            transformed = self._fallback_transform(image_data)
        if not isinstance(transformed, Tensor):
            transformed = self._fallback_transform(transformed)

        target = torch.tensor(class_id, dtype=torch.int64)
        return transformed, target


class AIStudioDataModule(pl.LightningDataModule):
    """Lightning data module that builds split-specific dataloaders."""

    def __init__(
        self,
        *,
        dataset: DatasetMetadata,
        images_dir: Path,
        split_name: str,
        augmentations: AugmentationConfig,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if num_workers < 0:
            raise ValueError("num_workers must be >= 0.")
        if dataset.task != "classification":
            raise ValueError(f"Task '{dataset.task}' is not implemented for AIStudioDataModule.")

        self.dataset = dataset
        self.images_dir = images_dir
        self.split_name = split_name
        self.augmentations = augmentations
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_dataset: ClassificationImageDataset | None = None
        self.val_dataset: ClassificationImageDataset | None = None
        self.test_dataset: ClassificationImageDataset | None = None

    @property
    def num_classes(self) -> int:
        """Return number of classes declared in dataset metadata."""
        return len(self.dataset.classes)

    def setup(self, stage: str | None = None) -> None:
        """Build train/val/test datasets for the requested split."""
        del stage
        split_index = self._resolve_split_index(self.split_name)
        train_samples: list[tuple[Path, int]] = []
        val_samples: list[tuple[Path, int]] = []
        test_samples: list[tuple[Path, int]] = []

        for image in self.dataset.images:
            class_id = self._extract_class_id(image)
            subset = image.split[split_index]
            sample = (self.images_dir / image.filename, class_id)
            if subset == "train":
                train_samples.append(sample)
            elif subset == "val":
                val_samples.append(sample)
            elif subset == "test":
                test_samples.append(sample)

        if not train_samples:
            raise ValueError(
                f"Split '{self.split_name}' has no training samples in dataset '{self.dataset.id}'."
            )

        train_transform = self._build_transform(self.augmentations.train)
        val_transform = self._build_transform(self.augmentations.val)

        self.train_dataset = ClassificationImageDataset(
            samples=train_samples,
            transform=train_transform,
        )
        self.val_dataset = ClassificationImageDataset(
            samples=val_samples,
            transform=val_transform,
        )
        self.test_dataset = ClassificationImageDataset(
            samples=test_samples,
            transform=val_transform,
        )

    def train_dataloader(self) -> DataLoader[tuple[Tensor, Tensor]]:
        """Return the training dataloader."""
        if self.train_dataset is None:
            raise RuntimeError(
                "AIStudioDataModule.setup() must be called before train_dataloader()."
            )
        return self._build_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader[tuple[Tensor, Tensor]]:
        """Return the validation dataloader."""
        if self.val_dataset is None:
            raise RuntimeError("AIStudioDataModule.setup() must be called before val_dataloader().")
        return self._build_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader[tuple[Tensor, Tensor]]:
        """Return the test dataloader."""
        if self.test_dataset is None:
            raise RuntimeError(
                "AIStudioDataModule.setup() must be called before test_dataloader()."
            )
        return self._build_dataloader(self.test_dataset, shuffle=False)

    def _build_dataloader(
        self,
        dataset: ClassificationImageDataset,
        *,
        shuffle: bool,
    ) -> DataLoader[tuple[Tensor, Tensor]]:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def _build_transform(self, config: list[object]) -> v2.Compose:
        from app.datasets.augmentations import build_augmentation_pipeline

        return build_augmentation_pipeline(config)

    def _resolve_split_index(self, split_name: str) -> int:
        try:
            return self.dataset.split_names.index(split_name)
        except ValueError as exc:
            raise ValueError(
                f"Unknown split '{split_name}' for dataset '{self.dataset.id}'."
            ) from exc

    def _extract_class_id(self, image: DatasetImage) -> int:
        for annotation in image.annotations:
            if annotation.type == "label":
                class_id = annotation.class_id
                if class_id < 0 or class_id >= len(self.dataset.classes):
                    raise ValueError(
                        f"Image '{image.filename}' has out-of-range class_id={class_id}."
                    )
                return class_id
        raise ValueError(f"Image '{image.filename}' has no classification label annotation.")


__all__ = ["AIStudioDataModule", "ClassificationImageDataset"]
