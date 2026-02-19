# Tasks — Object Detection

**Phase**: 3

---

## 1. Task Description

Locate objects in an image with **axis-aligned bounding boxes** and assign a class label to each detected object. The model outputs a list of `(bbox, class, confidence)` tuples.

---

## 2. Label Schema

```json
{
  "type": "bbox",
  "class_id": 2,
  "class_name": "car",
  "bbox": [120, 50, 200, 150]
}
```

- `bbox` format: `[x, y, width, height]` in pixel coordinates (top-left origin).
- An image can have zero or more `"bbox"` annotations.

---

## 3. Compatible Dataset Formats

| Format | Notes |
|--------|-------|
| **COCO JSON** | Native bbox format `[x, y, w, h]` |
| **YOLO** | Normalized `[cx, cy, w, h]` → denormalize to pixel `[x, y, w, h]` |

---

## 4. Compatible Architectures

| Architecture | Backbone | Type | Notes |
|-------------|----------|------|-------|
| Faster R-CNN | ResNet-50 + FPN | Two-stage | High accuracy, slower |
| FCOS | ResNet-50 + FPN | Anchor-free | Good balance |
| RetinaNet | ResNet-50 + FPN | One-stage | Focal loss, handles imbalance |
| SSD | MobileNetV3 / VGG | One-stage | Fast, lower accuracy |
| SSDLite | MobileNetV3 | One-stage | Mobile-optimized |

### Head: Detection Head

The head depends on the architecture:
- **Faster R-CNN**: Region Proposal Network (RPN) + RoI head with bbox regression + classification.
- **RetinaNet**: Feature Pyramid Network (FPN) + classification subnet + regression subnet.
- **FCOS**: FPN + per-pixel classification + centerness + bbox regression.

These are available as complete models in `torchvision.models.detection`.

---

## 5. Loss Functions

| Architecture | Losses |
|-------------|--------|
| **Faster R-CNN** | RPN classification + RPN regression + RoI classification + RoI regression (built-in) |
| **RetinaNet** | Focal loss (classification) + Smooth L1 (regression) |
| **FCOS** | Focal loss + GIoU loss + centerness BCE |

Detection models in torchvision compute their own losses internally. The `LightningModule` receives the loss dict and sums them.

---

## 6. Metrics

| Metric | Role | Notes |
|--------|------|-------|
| **mAP@0.5** | Primary | Mean Average Precision at IoU threshold 0.5 |
| **mAP@0.5:0.95** | Secondary | COCO-style mAP averaged over IoU thresholds 0.5 to 0.95 |
| **Precision** | Secondary | At confidence threshold 0.5 |
| **Recall** | Secondary | At confidence threshold 0.5 |
| **Per-class AP** | Detail | AP for each class individually |

Use `torchmetrics.detection.MeanAveragePrecision`.

---

## 7. Default Augmentations

### Training

```json
[
  { "name": "RandomHorizontalFlip", "params": { "p": 0.5 } },
  { "name": "RandomPhotometricDistort", "params": {} },
  { "name": "RandomZoomOut", "params": { "fill": [123, 117, 104], "p": 0.5 } },
  { "name": "RandomIoUCrop", "params": {} },
  { "name": "Resize", "params": { "size": [640, 640] } },
  { "name": "ToTensor", "params": {} },
  { "name": "Normalize", "params": { "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225] } }
]
```

**Important**: Spatial augmentations (flip, crop, zoom) must transform both the image and the bounding boxes.

### Validation / Test

```json
[
  { "name": "Resize", "params": { "size": [640, 640] } },
  { "name": "ToTensor", "params": {} },
  { "name": "Normalize", "params": { "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225] } }
]
```

---

## 8. Default Hyperparameters

```json
{
  "optimizer": "sgd",
  "learning_rate": 0.005,
  "momentum": 0.9,
  "weight_decay": 0.0005,
  "scheduler": "step",
  "step_size": 10,
  "gamma": 0.1,
  "batch_size": 8,
  "max_epochs": 50,
  "early_stopping_patience": 15,
  "pretrained": true,
  "freeze_backbone": false,
  "nms_threshold": 0.5,
  "score_threshold": 0.05
}
```

---

## 9. Visualization

### Dataset Page
- **Colored bounding boxes** overlaid on thumbnails, one color per class.
- **Class label** text above each box.
- In detail view: all boxes with class + coordinates in a side panel.

### Evaluation Page
- **Predicted boxes** (solid lines) overlaid on image.
- **Ground-truth boxes** (dashed lines) for comparison.
- Color coding: green for correct detections, red for false positives, yellow for missed (false negatives).
- Confidence score displayed next to each predicted box.

---

## 10. Evaluation Specifics

Per-image result:

```json
{
  "filename": "img_001.png",
  "ground_truth": [
    { "class_name": "car", "bbox": [120, 50, 200, 150] }
  ],
  "predictions": [
    { "class_name": "car", "bbox": [118, 48, 205, 152], "confidence": 0.92, "matched_gt_idx": 0 }
  ],
  "num_gt": 1,
  "num_predictions": 1,
  "true_positives": 1,
  "false_positives": 0,
  "false_negatives": 0
}
```

---

## 11. Collate Function

Detection datasets return variable-length annotation lists. A custom collate function is required:

```python
def detection_collate(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets  # List of tensors, not stacked
```

---

## 12. Related Documents

- Task registry → [00-task-registry.md](00-task-registry.md)
- Oriented detection (similar) → [04-oriented-object-detection.md](04-oriented-object-detection.md)
- Architecture catalog → [../04-models/00-architecture-catalog.md](../04-models/00-architecture-catalog.md)
