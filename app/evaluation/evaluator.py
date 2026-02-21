from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from app.datasets.augmentations import build_augmentation_pipeline
from app.evaluation.metrics import compute_classification_metrics
from app.models.catalog import create_model
from app.schemas.dataset import DatasetImage, DatasetMetadata
from app.schemas.evaluation import (
    ClassificationAggregateMetrics,
    ClassificationLabelRef,
    ClassificationPerClassAggregate,
    ClassificationPerImageResult,
    ClassificationPrediction,
    EvaluationConfig,
)
from app.schemas.training import ExperimentRecord

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True, slots=True)
class EvaluationRunOutput:
    """Output payload returned by one evaluator run."""

    results: list[ClassificationPerImageResult]
    aggregate: ClassificationAggregateMetrics


class _ClassificationEvaluationDataset(Dataset[tuple[Tensor, Tensor, str, str]]):
    """Classification dataset wrapper used during model evaluation."""

    def __init__(
        self,
        *,
        samples: list[tuple[Path, int, str, str]],
        transform: v2.Transform | None,
    ) -> None:
        self.samples = samples
        self.transform = transform
        self._fallback_transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

    def __len__(self) -> int:
        """Return number of available samples."""
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, str, str]:
        """Load one sample as `(image, class_id, filename, subset)`."""
        image_path, class_id, filename, subset = self.samples[index]
        with Image.open(image_path) as image:
            image_data = image.convert("RGB")

        if self.transform is not None:
            transformed = self.transform(image_data)
        else:
            transformed = self._fallback_transform(image_data)
        if not isinstance(transformed, Tensor):
            transformed = self._fallback_transform(transformed)

        target = torch.tensor(class_id, dtype=torch.int64)
        return transformed, target, filename, subset


class Evaluator:
    """Run one experiment-scoped evaluation for classification models."""

    def __init__(
        self,
        *,
        config: EvaluationConfig,
        project_id: str,
        experiment_id: str,
        dataset: DatasetMetadata,
        experiment: ExperimentRecord,
        images_dir: Path,
        checkpoint_path: Path,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        self.config = config
        self.project_id = project_id
        self.experiment_id = experiment_id
        self.dataset = dataset
        self.experiment = experiment
        self.images_dir = images_dir
        self.checkpoint_path = checkpoint_path
        self.progress_callback = progress_callback

    def run(self) -> EvaluationRunOutput:
        """Execute evaluation and return per-image and aggregate results."""
        if self.dataset.task != "classification":
            raise ValueError(f"Task '{self.dataset.task}' is not implemented yet in the evaluator.")

        split_index = self._resolve_split_index(self.experiment.split_name)
        samples = self._collect_samples(split_index=split_index)
        if not samples:
            selected = ", ".join(self.config.split_subsets)
            raise ValueError(
                f"No images found for split subsets [{selected}] in split "
                f"'{self.experiment.split_name}'."
            )

        # Checkpoint weights fully define model parameters; disable pretrained weight loading.
        model_config = self.experiment.model.model_copy(update={"pretrained": False})
        model = create_model(
            task=self.dataset.task,
            architecture=self.experiment.model.backbone,
            config=model_config,
            num_classes=len(self.dataset.classes),
        )
        self._load_model_checkpoint(model)

        device = self._resolve_device(self.config.device)
        model.to(device)
        model.eval()

        transform = build_augmentation_pipeline(self.experiment.augmentations.val)
        eval_dataset = _ClassificationEvaluationDataset(samples=samples, transform=transform)
        dataloader = DataLoader(
            eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )

        total = len(samples)
        processed = 0
        self._notify_progress(processed=processed, total=total)

        results: list[ClassificationPerImageResult] = []
        logits_batches: list[Tensor] = []
        target_batches: list[Tensor] = []

        with torch.no_grad():
            for images, targets, filenames, subsets in dataloader:
                images = images.to(device)
                outputs = model(images)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_ids = probabilities.argmax(dim=1)

                for index in range(len(filenames)):
                    target_id = int(targets[index].item())
                    predicted_id = int(predicted_ids[index].item())
                    confidence = float(probabilities[index][predicted_id].item())
                    probability_map = {
                        class_name: float(probabilities[index][class_index].item())
                        for class_index, class_name in enumerate(self.dataset.classes)
                    }

                    results.append(
                        ClassificationPerImageResult(
                            filename=filenames[index],
                            subset=subsets[index],
                            ground_truth=ClassificationLabelRef(
                                class_id=target_id,
                                class_name=self.dataset.classes[target_id],
                            ),
                            prediction=ClassificationPrediction(
                                class_id=predicted_id,
                                class_name=self.dataset.classes[predicted_id],
                                confidence=confidence,
                            ),
                            correct=predicted_id == target_id,
                            probabilities=probability_map,
                        )
                    )

                processed += len(filenames)
                self._notify_progress(processed=processed, total=total)
                logits_batches.append(outputs.detach().cpu())
                target_batches.append(targets.detach().cpu())

        all_logits = torch.cat(logits_batches, dim=0)
        all_targets = torch.cat(target_batches, dim=0)
        raw_metrics = compute_classification_metrics(
            all_logits,
            all_targets,
            num_classes=len(self.dataset.classes),
        )

        confusion_matrix_raw = raw_metrics["confusion_matrix"]
        if not isinstance(confusion_matrix_raw, list):
            raise ValueError("Evaluation metrics returned an invalid confusion matrix payload.")
        confusion_matrix = [[int(cell) for cell in row] for row in confusion_matrix_raw]
        per_class = self._build_per_class_metrics(confusion_matrix)

        aggregate = ClassificationAggregateMetrics(
            accuracy=float(raw_metrics["accuracy"]),
            precision_macro=float(raw_metrics["precision_macro"]),
            recall_macro=float(raw_metrics["recall_macro"]),
            f1_macro=float(raw_metrics["f1_macro"]),
            confusion_matrix=confusion_matrix,
            per_class=per_class,
        )
        return EvaluationRunOutput(results=results, aggregate=aggregate)

    def _resolve_split_index(self, split_name: str) -> int:
        try:
            return self.dataset.split_names.index(split_name)
        except ValueError as exc:
            raise ValueError(
                f"Split '{split_name}' was not found in dataset '{self.dataset.id}'."
            ) from exc

    def _collect_samples(self, *, split_index: int) -> list[tuple[Path, int, str, str]]:
        selected_subsets = set(self.config.split_subsets)
        samples: list[tuple[Path, int, str, str]] = []
        for image in self.dataset.images:
            subset = image.split[split_index]
            if subset not in selected_subsets:
                continue
            class_id = self._extract_class_id(image)
            samples.append((self.images_dir / image.filename, class_id, image.filename, subset))
        return samples

    def _extract_class_id(self, image: DatasetImage) -> int:
        for annotation in image.annotations:
            if annotation.type != "label":
                continue
            class_id = annotation.class_id
            if class_id < 0 or class_id >= len(self.dataset.classes):
                raise ValueError(f"Image '{image.filename}' has out-of-range class_id={class_id}.")
            return class_id
        raise ValueError(f"Image '{image.filename}' has no classification label annotation.")

    def _load_model_checkpoint(self, model: nn.Module) -> None:
        checkpoint_payload = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(checkpoint_payload, dict):
            raise ValueError(f"Checkpoint '{self.checkpoint_path.name}' is invalid.")

        raw_state_dict = checkpoint_payload.get("state_dict")
        if raw_state_dict is None:
            raw_state_dict = checkpoint_payload
        if not isinstance(raw_state_dict, dict):
            raise ValueError(
                f"Checkpoint '{self.checkpoint_path.name}' does not contain a state_dict."
            )

        state_dict = self._extract_model_state_dict(raw_state_dict)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys or unexpected_keys:
            raise ValueError(
                "Checkpoint weights do not match the selected model architecture. "
                f"missing={missing_keys}, unexpected={unexpected_keys}"
            )

    def _extract_model_state_dict(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        model_entries = {
            key[6:]: value for key, value in state_dict.items() if key.startswith("model.")
        }
        if model_entries:
            return model_entries
        return state_dict

    def _resolve_device(self, requested: str) -> torch.device:
        device_label = requested.strip().lower()
        if device_label == "cpu":
            return torch.device("cpu")

        if device_label in {"gpu", "cuda"}:
            if torch.cuda.is_available():
                return torch.device("cuda:0")
            logger.warning(
                "CUDA requested for evaluation project_id=%s experiment_id=%s but unavailable. "
                "Falling back to CPU.",
                self.project_id,
                self.experiment_id,
            )
            return torch.device("cpu")

        if device_label.startswith(("gpu:", "cuda:")):
            _, _, raw_index = device_label.partition(":")
            if raw_index.isdigit() and torch.cuda.is_available():
                index = int(raw_index)
                if index < torch.cuda.device_count():
                    return torch.device(f"cuda:{index}")
            logger.warning(
                "Device '%s' requested for evaluation project_id=%s experiment_id=%s "
                "but unavailable. Falling back to CPU.",
                requested,
                self.project_id,
                self.experiment_id,
            )
            return torch.device("cpu")

        raise ValueError(f"Unsupported device label '{requested}'.")

    def _build_per_class_metrics(
        self,
        confusion_matrix: list[list[int]],
    ) -> dict[str, ClassificationPerClassAggregate]:
        per_class: dict[str, ClassificationPerClassAggregate] = {}
        for class_index, class_name in enumerate(self.dataset.classes):
            row = confusion_matrix[class_index]
            support = sum(row)
            true_positive = row[class_index]
            false_positive = sum(
                confusion_matrix[other_index][class_index]
                for other_index in range(len(confusion_matrix))
                if other_index != class_index
            )
            false_negative = sum(
                row[other_index] for other_index in range(len(row)) if other_index != class_index
            )

            precision = (
                float(true_positive / (true_positive + false_positive))
                if (true_positive + false_positive) > 0
                else 0.0
            )
            recall = (
                float(true_positive / (true_positive + false_negative))
                if (true_positive + false_negative) > 0
                else 0.0
            )
            f1 = (
                float((2 * precision * recall) / (precision + recall))
                if (precision + recall) > 0
                else 0.0
            )

            per_class[class_name] = ClassificationPerClassAggregate(
                precision=precision,
                recall=recall,
                f1=f1,
                support=support,
            )

        return per_class

    def _notify_progress(self, *, processed: int, total: int) -> None:
        if self.progress_callback is None:
            return
        self.progress_callback(processed, total)


__all__ = ["EvaluationRunOutput", "Evaluator"]
