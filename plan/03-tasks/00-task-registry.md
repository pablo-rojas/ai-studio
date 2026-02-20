# Tasks — Task Registry

This document defines the abstract interface that every task must implement and how tasks are registered and discovered.

---

## 1. Task Enum

```python
from enum import Enum

class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    ANOMALY_DETECTION = "anomaly_detection"
    OBJECT_DETECTION = "object_detection"
    ORIENTED_OBJECT_DETECTION = "oriented_object_detection"
    SEGMENTATION = "segmentation"
    INSTANCE_SEGMENTATION = "instance_segmentation"
    REGRESSION = "regression"
```

---

## 2. Phased Availability

Each task has a phase number. Tasks with phase > current release phase are hidden from the UI (or shown as "Coming Soon").

| Task | Phase | Status |
|------|-------|--------|
| Classification | 8 | **Active** |
| Anomaly Detection | 23 | Planned |
| Object Detection | 19 | Planned |
| Oriented Object Detection | 25 | Planned |
| Segmentation | 20 | Planned |
| Instance Segmentation | 21 | Planned |
| Regression | 24 | Planned |

Configuration: `ACTIVE_PHASES = [1]` initially, incremented as phases are implemented.

---

## 3. Task Interface

Each task module must provide the following components. These are not enforced via abstract base classes initially, but follow a convention that can be formalized later.

### 3.1 Dataset Adapter

- **Annotation type(s)**: which annotation `type` values in `dataset.json` this task expects.
- **Dataset class**: a PyTorch `Dataset` (or `LightningDataModule` mixin) that loads images + annotations for this task and returns `(image_tensor, target)` tuples.
- **Collate function**: if the default collate doesn't work (e.g., variable-length bbox lists for detection), provide a custom collate.

### 3.2 Model Head

- **Head module**: a `nn.Module` that takes backbone feature maps and produces task-specific output.
- **Compatible backbones**: which backbones work with this head (most heads work with any backbone, but some may require FPN features).
- **Output format**: what the head returns (logits, bbox deltas, mask maps, scalar, etc.).

### 3.3 Loss Function(s)

- **Default loss**: the default loss function for the task (e.g., `CrossEntropyLoss` for classification).
- **Alternate losses**: optional alternative losses the user can select.

### 3.4 Metrics

- **Primary metric**: the metric used for early stopping and "best model" selection.
- **Secondary metrics**: additional metrics displayed in the UI and logged.

| Task | Primary Metric | Secondary Metrics |
|------|---------------|-------------------|
| Classification | Accuracy | F1, Precision, Recall, Confusion Matrix |
| Anomaly Detection | AUROC | F1, Precision, Recall |
| Object Detection | mAP@0.5 | mAP@0.5:0.95, Precision, Recall |
| Oriented OD | mAP@0.5 | mAP@0.5:0.95, Precision, Recall |
| Segmentation | mIoU | Pixel Accuracy, Dice, per-class IoU |
| Instance Segmentation | mAP@0.5 (mask) | mAP@0.5:0.95, per-class AP |
| Regression | MAE | MSE, RMSE, R² |

### 3.5 Visualization

- **Annotation overlay**: how to draw annotations on an image in the Dataset page.
- **Prediction overlay**: how to draw predictions on an image in the Evaluation page.

| Task | Overlay Type |
|------|-------------|
| Classification | Class label badge (top-left corner) |
| Anomaly Detection | Heatmap overlay + normal/anomalous badge |
| Object Detection | Colored bounding boxes + class labels |
| Oriented OD | Rotated bounding boxes + class labels |
| Segmentation | Semi-transparent colored mask overlay |
| Instance Segmentation | Per-instance colored mask + class labels |
| Regression | Values badge (top-left corner) |

### 3.6 Default Augmentations

Each task provides a sensible default augmentation pipeline for training:

- **All tasks**: `Resize`, `Normalize` (ImageNet stats).
- **Classification / Regression**: `RandomHorizontalFlip`, `RandomRotation`, `ColorJitter`.
- **Detection**: `RandomHorizontalFlip`, `RandomResize`, `RandomCrop` (with bbox adjustment).
- **Segmentation**: `RandomHorizontalFlip`, `RandomCrop`, `RandomRotation` (applied to image + mask jointly).

---

## 4. Task Registry (Code)

```python
# app/models/catalog.py (simplified)

TASK_REGISTRY = {
    TaskType.CLASSIFICATION: {
        "annotation_types": ["label"],
        "dataset_class": ClassificationDataset,
        "heads": ["fc_classifier"],
        "default_loss": "cross_entropy",
        "primary_metric": "accuracy",
        "default_augmentations": CLASSIFICATION_AUGMENTATIONS,
    },
    # ... one entry per task
}

def get_task_config(task: TaskType) -> dict:
    """Return the registry entry for a task."""
    return TASK_REGISTRY[task]
```

---

## 5. Adding a New Task (Checklist)

When implementing a new task phase:

1. [ ] Define annotation type(s) in `dataset.json` spec.
2. [ ] Implement format parsers for applicable formats.
3. [ ] Implement a Dataset class in `app/datasets/`.
4. [ ] Implement head module(s) in `app/models/heads/`.
5. [ ] Define loss function(s) in `app/training/losses.py`.
6. [ ] Define metrics in `app/evaluation/metrics.py`.
7. [ ] Define default augmentations.
8. [ ] Implement visualization overlays for Dataset + Evaluation pages.
9. [ ] Register in `TASK_REGISTRY`.
10. [ ] Update `ACTIVE_PHASES` to include the new phase.
11. [ ] Add tests for the new task end-to-end.

---

## 6. Related Documents

- Per-task details → [01-classification.md](01-classification.md) through [07-regression.md](07-regression.md)
- Model catalog → [../04-models/00-architecture-catalog.md](../04-models/00-architecture-catalog.md)
- Training pipeline → [../05-training/00-training-pipeline.md](../05-training/00-training-pipeline.md)
- Evaluation metrics → [../07-evaluation/02-aggregate-metrics.md](../07-evaluation/02-aggregate-metrics.md)
