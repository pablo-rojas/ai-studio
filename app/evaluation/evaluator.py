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
from torchvision import tv_tensors
from torchvision.ops import box_iou
from torchvision.transforms import v2

from app.datasets.augmentations import build_augmentation_pipeline
from app.evaluation.metrics import compute_classification_metrics, compute_object_detection_metrics
from app.models.catalog import create_model
from app.schemas.dataset import DatasetImage, DatasetMetadata
from app.schemas.evaluation import (
    ClassificationAggregateMetrics,
    ClassificationLabelRef,
    ClassificationPerClassAggregate,
    ClassificationPerImageResult,
    ClassificationPrediction,
    EvaluationConfig,
    EvaluationPerImageResult,
    ObjectDetectionAggregateMetrics,
    ObjectDetectionLabelRef,
    ObjectDetectionPerImageResult,
    ObjectDetectionPrediction,
)
from app.schemas.training import ExperimentRecord

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], None]
DetectionTarget = dict[str, Tensor]


@dataclass(frozen=True, slots=True)
class EvaluationRunOutput:
    """Output payload returned by one evaluator run."""

    results: list[EvaluationPerImageResult]
    aggregate: ClassificationAggregateMetrics | ObjectDetectionAggregateMetrics


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


class _DetectionEvaluationDataset(Dataset[tuple[Tensor, DetectionTarget, str, str]]):
    """Object detection dataset wrapper used during model evaluation."""

    def __init__(
        self,
        *,
        samples: list[tuple[Path, int, int, list[tuple[int, list[float]]], str, str]],
        transform: v2.Transform | None,
    ) -> None:
        self.samples = samples
        self.transform = transform
        self._fallback_transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

    def __len__(self) -> int:
        """Return number of available samples."""
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, DetectionTarget, str, str]:
        """Load one sample as `(image, target, filename, subset)`."""
        image_path, width, height, annotations, filename, subset = self.samples[index]
        with Image.open(image_path) as image:
            image_data = image.convert("RGB")

        boxes_xyxy = torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.zeros((0,), dtype=torch.int64)
        if annotations:
            boxes_xyxy = torch.tensor(
                [
                    [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
                    for _, bbox in annotations
                ],
                dtype=torch.float32,
            )
            labels = torch.tensor([class_id for class_id, _ in annotations], dtype=torch.int64)

        target: dict[str, Any] = {
            "boxes": tv_tensors.BoundingBoxes(
                boxes_xyxy,
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=(height, width),
            ),
            "labels": labels,
        }
        if self.transform is not None:
            transformed_image, transformed_target = self.transform(image_data, target)
        else:
            transformed_image = self._fallback_transform(image_data)
            transformed_target = target

        if not isinstance(transformed_image, Tensor):
            transformed_image = self._fallback_transform(transformed_image)

        boxes_data = transformed_target.get("boxes", torch.zeros((0, 4), dtype=torch.float32))
        if isinstance(boxes_data, tv_tensors.BoundingBoxes):
            boxes_tensor = boxes_data.as_subclass(torch.Tensor)
        else:
            boxes_tensor = torch.as_tensor(boxes_data, dtype=torch.float32)
        if boxes_tensor.numel() == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
        else:
            boxes_tensor = boxes_tensor.reshape(-1, 4).to(dtype=torch.float32)

        labels_tensor = torch.as_tensor(
            transformed_target.get("labels", torch.zeros((0,), dtype=torch.int64)),
            dtype=torch.int64,
        ).reshape(-1)
        if boxes_tensor.shape[0] != labels_tensor.shape[0]:
            raise ValueError("Detection target boxes and labels must have matching lengths.")

        return transformed_image, {"boxes": boxes_tensor, "labels": labels_tensor}, filename, subset


def _detection_eval_collate(
    batch: list[tuple[Tensor, DetectionTarget, str, str]],
) -> tuple[list[Tensor], list[DetectionTarget], list[str], list[str]]:
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    filenames = [item[2] for item in batch]
    subsets = [item[3] for item in batch]
    return images, targets, filenames, subsets


class Evaluator:
    """Run one experiment-scoped evaluation for implemented tasks."""

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
        if self.dataset.task == "classification":
            return self._run_classification()
        if self.dataset.task == "object_detection":
            return self._run_object_detection()
        raise ValueError(f"Task '{self.dataset.task}' is not implemented yet in the evaluator.")

    def _run_classification(self) -> EvaluationRunOutput:
        split_index = self._resolve_split_index(self.experiment.split_name)
        samples = self._collect_classification_samples(split_index=split_index)
        if not samples:
            selected = ", ".join(self.config.split_subsets)
            raise ValueError(
                f"No images found for split subsets [{selected}] in split "
                f"'{self.experiment.split_name}'."
            )

        model = self._build_model()
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
                if not isinstance(outputs, Tensor):
                    raise ValueError("Classification evaluator expected tensor model outputs.")
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

    def _run_object_detection(self) -> EvaluationRunOutput:
        split_index = self._resolve_split_index(self.experiment.split_name)
        samples = self._collect_detection_samples(split_index=split_index)
        if not samples:
            selected = ", ".join(self.config.split_subsets)
            raise ValueError(
                f"No images found for split subsets [{selected}] in split "
                f"'{self.experiment.split_name}'."
            )

        model = self._build_model()
        self._load_model_checkpoint(model)
        device = self._resolve_device(self.config.device)
        model.to(device)
        model.eval()

        transform = build_augmentation_pipeline(self.experiment.augmentations.val)
        eval_dataset = _DetectionEvaluationDataset(samples=samples, transform=transform)
        dataloader = DataLoader(
            eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
            collate_fn=_detection_eval_collate,
        )

        total = len(samples)
        processed = 0
        self._notify_progress(processed=processed, total=total)

        results: list[ObjectDetectionPerImageResult] = []
        all_predictions: list[DetectionTarget] = []
        all_targets: list[DetectionTarget] = []
        total_gt = 0
        total_predictions = 0
        total_tp = 0
        total_fp = 0
        total_fn = 0

        with torch.no_grad():
            for images, targets, filenames, subsets in dataloader:
                images_on_device = [image.to(device) for image in images]
                predictions = model(images_on_device)
                if not isinstance(predictions, list):
                    raise ValueError("Detection evaluator expected list model outputs.")

                detached_predictions = [_detach_detection_target(item) for item in predictions]
                detached_targets = [_detach_detection_target(item) for item in targets]
                all_predictions.extend(detached_predictions)
                all_targets.extend(detached_targets)

                for index in range(len(filenames)):
                    result = self._build_detection_result(
                        filename=filenames[index],
                        subset=subsets[index],
                        target=detached_targets[index],
                        prediction=detached_predictions[index],
                    )
                    results.append(result)
                    total_gt += result.num_gt
                    total_predictions += result.num_predictions
                    total_tp += result.true_positives
                    total_fp += result.false_positives
                    total_fn += result.false_negatives

                processed += len(filenames)
                self._notify_progress(processed=processed, total=total)

        map_metrics = compute_object_detection_metrics(
            all_predictions,
            all_targets,
            class_names=self.dataset.classes,
        )
        if not isinstance(map_metrics["per_class_AP"], dict):
            raise ValueError("Detection metrics returned invalid per_class_AP values.")

        precision = float(total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
        recall = float(total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else 0.0
        aggregate = ObjectDetectionAggregateMetrics(
            mAP_50=float(map_metrics["mAP_50"]),
            mAP_50_95=float(map_metrics["mAP_50_95"]),
            precision=precision,
            recall=recall,
            per_class_AP={key: float(value) for key, value in map_metrics["per_class_AP"].items()},
            total_gt=total_gt,
            total_predictions=total_predictions,
            total_tp=total_tp,
            total_fp=total_fp,
            total_fn=total_fn,
        )
        return EvaluationRunOutput(results=results, aggregate=aggregate)

    def _build_model(self) -> nn.Module:
        model_config = self.experiment.model.model_copy(update={"pretrained": False})
        return create_model(
            task=self.dataset.task,
            architecture=self.experiment.model.backbone,
            config=model_config,
            num_classes=len(self.dataset.classes),
        )

    def _resolve_split_index(self, split_name: str) -> int:
        try:
            return self.dataset.split_names.index(split_name)
        except ValueError as exc:
            raise ValueError(
                f"Split '{split_name}' was not found in dataset '{self.dataset.id}'."
            ) from exc

    def _collect_classification_samples(
        self,
        *,
        split_index: int,
    ) -> list[tuple[Path, int, str, str]]:
        selected_subsets = set(self.config.split_subsets)
        samples: list[tuple[Path, int, str, str]] = []
        for image in self.dataset.images:
            subset = image.split[split_index]
            if subset not in selected_subsets:
                continue
            class_id = self._extract_class_id(image)
            samples.append((self.images_dir / image.filename, class_id, image.filename, subset))
        return samples

    def _collect_detection_samples(
        self,
        *,
        split_index: int,
    ) -> list[tuple[Path, int, int, list[tuple[int, list[float]]], str, str]]:
        selected_subsets = set(self.config.split_subsets)
        samples: list[tuple[Path, int, int, list[tuple[int, list[float]]], str, str]] = []
        for image in self.dataset.images:
            subset = image.split[split_index]
            if subset not in selected_subsets:
                continue
            samples.append(
                (
                    self.images_dir / image.filename,
                    image.width,
                    image.height,
                    self._extract_detection_annotations(image),
                    image.filename,
                    subset,
                )
            )
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

    def _extract_detection_annotations(
        self,
        image: DatasetImage,
    ) -> list[tuple[int, list[float]]]:
        parsed: list[tuple[int, list[float]]] = []
        for annotation in image.annotations:
            if annotation.type != "bbox":
                continue
            class_id = annotation.class_id
            if class_id < 0 or class_id >= len(self.dataset.classes):
                raise ValueError(f"Image '{image.filename}' has out-of-range class_id={class_id}.")
            parsed.append((class_id, [float(value) for value in annotation.bbox]))
        return parsed

    def _build_detection_result(
        self,
        *,
        filename: str,
        subset: str,
        target: DetectionTarget,
        prediction: DetectionTarget,
    ) -> ObjectDetectionPerImageResult:
        gt_boxes = target.get("boxes", torch.zeros((0, 4), dtype=torch.float32)).reshape(-1, 4)
        gt_labels = target.get("labels", torch.zeros((0,), dtype=torch.int64)).reshape(-1)
        pred_boxes = prediction.get("boxes", torch.zeros((0, 4), dtype=torch.float32)).reshape(
            -1, 4
        )
        pred_labels = prediction.get("labels", torch.zeros((0,), dtype=torch.int64)).reshape(-1)
        pred_scores = prediction.get("scores", torch.zeros((0,), dtype=torch.float32)).reshape(-1)

        matched_gt_indices, true_positives = _match_predictions(
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            pred_boxes=pred_boxes,
            pred_labels=pred_labels,
            pred_scores=pred_scores,
        )
        false_positives = int(pred_boxes.shape[0] - true_positives)
        false_negatives = int(gt_boxes.shape[0] - true_positives)

        gt_payload = [
            ObjectDetectionLabelRef(
                class_name=self.dataset.classes[int(label.item())],
                bbox=_xyxy_to_xywh(box),
            )
            for box, label in zip(gt_boxes, gt_labels, strict=False)
        ]
        prediction_payload: list[ObjectDetectionPrediction] = []
        for index in range(pred_boxes.shape[0]):
            label_id = int(pred_labels[index].item())
            class_name = (
                self.dataset.classes[label_id]
                if 0 <= label_id < len(self.dataset.classes)
                else str(label_id)
            )
            prediction_payload.append(
                ObjectDetectionPrediction(
                    class_name=class_name,
                    bbox=_xyxy_to_xywh(pred_boxes[index]),
                    confidence=float(pred_scores[index].item()),
                    matched_gt_idx=matched_gt_indices[index],
                )
            )

        return ObjectDetectionPerImageResult(
            filename=filename,
            subset=subset,
            ground_truth=gt_payload,
            predictions=prediction_payload,
            num_gt=len(gt_payload),
            num_predictions=len(prediction_payload),
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
        )

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


def _detach_detection_target(target: dict[str, Any]) -> DetectionTarget:
    boxes = target.get("boxes", torch.zeros((0, 4), dtype=torch.float32))
    labels = target.get("labels", torch.zeros((0,), dtype=torch.int64))
    scores = target.get("scores")

    boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32).detach().cpu().reshape(-1, 4)
    labels_tensor = torch.as_tensor(labels, dtype=torch.int64).detach().cpu().reshape(-1)
    result: DetectionTarget = {"boxes": boxes_tensor, "labels": labels_tensor}
    if scores is not None:
        result["scores"] = torch.as_tensor(scores, dtype=torch.float32).detach().cpu().reshape(-1)
    return result


def _xyxy_to_xywh(box: Tensor) -> list[float]:
    x1, y1, x2, y2 = [float(value) for value in box.tolist()]
    width = max(x2 - x1, 1e-6)
    height = max(y2 - y1, 1e-6)
    return [x1, y1, width, height]


def _match_predictions(
    *,
    gt_boxes: Tensor,
    gt_labels: Tensor,
    pred_boxes: Tensor,
    pred_labels: Tensor,
    pred_scores: Tensor,
    iou_threshold: float = 0.5,
) -> tuple[list[int | None], int]:
    if pred_boxes.numel() == 0:
        return [], 0

    matched_gt_indices: list[int | None] = [None] * int(pred_boxes.shape[0])
    used_gt: set[int] = set()
    true_positives = 0

    order = torch.argsort(pred_scores, descending=True)
    for ordered_index in order.tolist():
        if gt_boxes.numel() == 0:
            continue
        candidate_box = pred_boxes[ordered_index].unsqueeze(0)
        ious = box_iou(candidate_box, gt_boxes).squeeze(0)
        best_iou = 0.0
        best_gt_index: int | None = None

        for gt_index in range(int(gt_boxes.shape[0])):
            if gt_index in used_gt:
                continue
            if int(pred_labels[ordered_index].item()) != int(gt_labels[gt_index].item()):
                continue
            iou_value = float(ious[gt_index].item())
            if iou_value > best_iou:
                best_iou = iou_value
                best_gt_index = gt_index

        if best_gt_index is not None and best_iou >= iou_threshold:
            used_gt.add(best_gt_index)
            matched_gt_indices[ordered_index] = best_gt_index
            true_positives += 1

    return matched_gt_indices, true_positives


__all__ = ["EvaluationRunOutput", "Evaluator"]
