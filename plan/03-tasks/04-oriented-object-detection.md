# Tasks — Oriented Object Detection

**Phase**: 3

---

## 1. Task Description

Locate objects in an image with **rotated (oriented) bounding boxes** and assign a class label. Unlike standard object detection, boxes can be rotated at any angle, which is important for aerial imagery, document analysis, and objects at arbitrary orientations.

---

## 2. Label Schema

```json
{
  "type": "oriented_bbox",
  "class_id": 1,
  "class_name": "ship",
  "bbox": [320.0, 240.0, 100.0, 40.0, 35.0]
}
```

- `bbox` format: `[cx, cy, width, height, angle]` where:
  - `cx, cy`: center coordinates in pixels.
  - `width, height`: dimensions of the rotated box.
  - `angle`: rotation angle in degrees (typically -90 to 90 or 0 to 180, depending on convention).

---

## 3. Compatible Dataset Formats

| Format | Notes |
|--------|-------|
| **YOLO OBB** | `class cx cy w h angle` (normalized) — denormalize to pixel coords |
| **DOTA** | 4-corner format `x1 y1 x2 y2 x3 y3 x4 y4 class` → convert to `cx, cy, w, h, angle` |
| **Custom CSV** | `filename, class, cx, cy, w, h, angle` |

A dedicated parser for DOTA format will be needed.

---

## 4. Compatible Architectures

| Architecture | Notes |
|-------------|-------|
| Rotated Faster R-CNN | torchvision does not include this natively — requires custom implementation or third-party |
| Oriented RetinaNet | Custom head with angle regression |
| FCOS with angle head | FCOS + angle prediction branch |

**Note**: torchvision does not ship oriented detection models. Options:
1. Extend standard detection models with an angle regression head.
2. Use a third-party library (e.g., `mmrotate` concepts adapted to our codebase).
3. Implement a lightweight rotated prediction head on top of FPN features.

Initial approach: extend the FCOS or RetinaNet head with an additional angle regression output.

---

## 5. Loss Functions

| Loss Component | Function |
|---------------|----------|
| Classification | Focal Loss |
| Bbox regression (x, y, w, h) | Smooth L1 / GIoU |
| Angle regression | Smooth L1 on angle, or periodic loss (to handle angle wrapping) |

Angle loss consideration: angles wrap around (0° = 180° for symmetric objects). May use a CSL (Circular Smooth Label) approach or direct regression with modular arithmetic loss.

---

## 6. Metrics

| Metric | Role |
|--------|------|
| **mAP@0.5 (rotated IoU)** | Primary |
| **mAP@0.5:0.95 (rotated IoU)** | Secondary |
| Per-class AP | Detail |

IoU computation for oriented boxes uses the Shapely polygon intersection or a custom rotated IoU kernel.

---

## 7. Default Augmentations

### Training

```json
[
  { "name": "RandomHorizontalFlip", "params": { "p": 0.5 } },
  { "name": "RandomVerticalFlip", "params": { "p": 0.5 } },
  { "name": "RandomRotation", "params": { "degrees": 90 } },
  { "name": "Resize", "params": { "size": [800, 800] } },
  { "name": "ToTensor", "params": {} },
  { "name": "Normalize", "params": { "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225] } }
]
```

**Important**: When rotating the image, all oriented bounding box angles must be adjusted accordingly.

---

## 8. Default Hyperparameters

Same as object detection with:
```json
{
  "optimizer": "sgd",
  "learning_rate": 0.005,
  "momentum": 0.9,
  "weight_decay": 0.0005,
  "scheduler": "step",
  "batch_size": 4,
  "max_epochs": 60,
  "early_stopping_patience": 15,
  "pretrained": true,
  "angle_loss_weight": 1.0,
  "nms_threshold": 0.3,
  "score_threshold": 0.05
}
```

---

## 9. Visualization

### Dataset Page
- **Rotated bounding boxes** drawn as quadrilaterals, one color per class.
- **Class label** at the top of the rotated box.

### Evaluation Page
- Predicted rotated boxes (solid) vs. ground-truth rotated boxes (dashed).
- Angle displayed alongside class and confidence.

---

## 10. Related Documents

- Standard object detection → [03-object-detection.md](03-object-detection.md)
- Task registry → [00-task-registry.md](00-task-registry.md)
