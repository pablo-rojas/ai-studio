# Tasks — Instance Segmentation

**Phase**: 21

---

## 1. Task Description

Detect individual object instances in an image, providing both a **bounding box** and a **pixel-level mask** for each instance, along with a class label and confidence score.

Combines object detection (localization + classification) with semantic segmentation (pixel-level delineation) at the instance level.

---

## 2. Label Schema

```json
{
  "type": "instance",
  "class_id": 1,
  "class_name": "person",
  "bbox": [50, 30, 120, 250],
  "mask_path": "masks/instances/img_001_inst_0.png",
  "polygon": [[50, 30], [170, 30], [170, 280], [50, 280]]
}
```

- Either `mask_path` (binary per-instance mask) or `polygon` (list of `[x, y]` vertices) should be provided.
- `bbox` is the axis-aligned bounding box enclosing the instance (can be derived from the mask/polygon if not provided).
- Multiple instances per image, potentially of different classes.

---

## 3. Compatible Dataset Formats

| Format | Notes |
|--------|-------|
| **COCO JSON** | `annotations[].segmentation` (polygon or RLE) + `bbox` + `category_id` |

---

## 4. Compatible Architectures

| Architecture | Backbone | Notes |
|-------------|----------|-------|
| **Mask R-CNN** | ResNet-50 + FPN | Extends Faster R-CNN with a mask head per instance |

Available in `torchvision.models.detection.maskrcnn_resnet50_fpn`.

### Head: Mask Head

Extends the Faster R-CNN RoI head:
- **Box head**: classifies + refines bounding box (same as Faster R-CNN).
- **Mask head**: for each detected RoI, predicts a binary mask of shape `(num_classes, 14, 14)` or `(num_classes, 28, 28)`.

---

## 5. Loss Functions

Mask R-CNN uses a multi-task loss (all built-in):

| Component | Loss |
|-----------|------|
| RPN classification | Binary Cross-Entropy |
| RPN box regression | Smooth L1 |
| RoI classification | Cross-Entropy |
| RoI box regression | Smooth L1 |
| Mask prediction | Per-pixel Binary Cross-Entropy (per class) |

---

## 6. Metrics

| Metric | Role |
|--------|------|
| **mAP@0.5 (mask IoU)** | Primary — uses mask-level IoU, not box IoU |
| **mAP@0.5:0.95 (mask)** | Secondary |
| **mAP@0.5 (box)** | Secondary |
| Per-class AP (mask) | Detail |

Use COCO evaluation protocol via `torchmetrics.detection.MeanAveragePrecision` with `iou_type="segm"`.

---

## 7. Default Augmentations

### Training

```json
[
  { "name": "RandomHorizontalFlip", "params": { "p": 0.5 } },
  { "name": "RandomPhotometricDistort", "params": {} },
  { "name": "Resize", "params": { "size": [800, 800] } },
  { "name": "ToImage", "params": {} },
  { "name": "Normalize", "params": { "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225] } }
]
```

Spatial transforms must be applied to image, bboxes, **and** masks/polygons jointly.

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
  "batch_size": 4,
  "max_epochs": 50,
  "early_stopping_patience": 15,
  "pretrained": true,
  "nms_threshold": 0.5,
  "score_threshold": 0.05,
  "mask_threshold": 0.5
}
```

---

## 9. Visualization

### Dataset Page
- **Per-instance colored masks** overlaid on the image (each instance gets a unique color, with class-based hue).
- **Class labels** near each instance.
- Detail view: list of instances with class, bbox, and mask statistics.

### Evaluation Page
- **Predicted instance masks** (solid colors) vs. ground-truth (dashed outlines).
- Confidence score per instance.
- Color coding for matched/unmatched instances.

---

## 10. Collate Function

Same as object detection — variable-length targets require a custom collate function.

---

## 11. Related Documents

- Semantic segmentation → [05-segmentation.md](05-segmentation.md)
- Object detection → [03-object-detection.md](03-object-detection.md)
- Task registry → [00-task-registry.md](00-task-registry.md)
