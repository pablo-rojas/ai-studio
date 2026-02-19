# Tasks — Segmentation (Semantic)

**Phase**: 4

---

## 1. Task Description

Assign a class label to **every pixel** in the image. The output is a dense prediction map of the same spatial dimensions as the input.

---

## 2. Label Schema

```json
{
  "type": "segmentation_mask",
  "mask_path": "masks/img_001.png"
}
```

- `mask_path`: relative path to a single-channel PNG where each pixel's value is the `class_id`.
- Pixel value `0` typically represents background.
- `classes` in `dataset.json` defines the mapping from ID to class name.

---

## 3. Compatible Dataset Formats

| Format | Notes |
|--------|-------|
| **COCO JSON** | Polygon annotations → rasterize to pixel masks |

---

## 4. Compatible Architectures

| Architecture | Backbone | Notes |
|-------------|----------|-------|
| **FCN** | ResNet-50/101 | Fully Convolutional Network, simple decoder |
| **DeepLabV3** | ResNet-50/101 | Atrous convolutions, multi-scale context |
| **DeepLabV3+** | ResNet-50/101 | Encoder-decoder with atrous separable convolution |
| **LRASPP** | MobileNetV3 | Lightweight, mobile-friendly |

All available in `torchvision.models.segmentation`.

### Head: Segmentation Decoder

The head takes backbone features (with FPN or dilated convolutions) and upsamples to full resolution:
- Output shape: `(N, num_classes, H, W)` — per-pixel logits.
- `argmax` over the class dimension gives the predicted mask.

---

## 5. Loss Functions

| Loss | When to Use |
|------|------------|
| **CrossEntropyLoss** (per-pixel) (default) | Standard multi-class segmentation |
| **DiceLoss** | Handles class imbalance better, optimizes IoU-like metric |
| **CE + Dice** (combined) | Often gives best results — combines stability of CE with Dice's imbalance handling |
| **FocalLoss** (per-pixel) | Extreme class imbalance |

---

## 6. Metrics

| Metric | Role |
|--------|------|
| **mIoU** | Primary — mean Intersection over Union across all classes |
| **Pixel Accuracy** | Secondary — fraction of correctly classified pixels |
| **Dice Score** (macro) | Secondary |
| **Per-class IoU** | Detail |

---

## 7. Default Augmentations

### Training

```json
[
  { "name": "RandomResizedCrop", "params": { "size": [512, 512], "scale": [0.5, 2.0] } },
  { "name": "RandomHorizontalFlip", "params": { "p": 0.5 } },
  { "name": "RandomRotation", "params": { "degrees": 10 } },
  { "name": "ColorJitter", "params": { "brightness": 0.3, "contrast": 0.3 } },
  { "name": "ToTensor", "params": {} },
  { "name": "Normalize", "params": { "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225] } }
]
```

**Important**: All spatial transforms (crop, flip, rotation) must be applied identically to both the image and the mask. The `torchvision.transforms.v2` API supports this via joint transforms.

### Validation / Test

```json
[
  { "name": "Resize", "params": { "size": [512, 512] } },
  { "name": "ToTensor", "params": {} },
  { "name": "Normalize", "params": { "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225] } }
]
```

---

## 8. Default Hyperparameters

```json
{
  "optimizer": "sgd",
  "learning_rate": 0.01,
  "momentum": 0.9,
  "weight_decay": 0.0001,
  "scheduler": "poly",
  "poly_power": 0.9,
  "batch_size": 8,
  "max_epochs": 60,
  "early_stopping_patience": 15,
  "loss": "ce_dice",
  "pretrained": true,
  "freeze_backbone": false,
  "ignore_index": 255
}
```

`ignore_index`: pixel value in masks to ignore during loss computation (commonly 255 for unlabeled regions).

---

## 9. Visualization

### Dataset Page
- **Semi-transparent colored mask overlay** on the thumbnail, one color per class.
- In detail view: full mask overlay with a class color legend.

### Evaluation Page
- **Predicted mask overlay** alongside ground-truth mask overlay.
- **Error map**: highlight pixels where prediction ≠ ground truth.
- Per-class IoU shown in a side panel.

---

## 10. Evaluation Specifics

Per-image result:

```json
{
  "filename": "img_001.png",
  "predicted_mask_path": "per_image/img_001_pred.png",
  "pixel_accuracy": 0.94,
  "per_class_iou": { "background": 0.97, "road": 0.91, "building": 0.88 },
  "miou": 0.92
}
```

---

## 11. Related Documents

- Instance segmentation → [06-instance-segmentation.md](06-instance-segmentation.md)
- Task registry → [00-task-registry.md](00-task-registry.md)
- Architecture catalog → [../04-models/00-architecture-catalog.md](../04-models/00-architecture-catalog.md)
