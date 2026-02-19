# Training — Augmentations

This document describes the data augmentation system: pipeline configuration, per-task defaults, and GUI controls.

---

## 1. Overview

Augmentations apply random transformations to training images (and their annotations) to increase data diversity and reduce overfitting. The pipeline is defined as a JSON array stored in `experiment.json`.

- **Training set**: augmentations applied (random transforms + resize + normalize).
- **Validation / Test sets**: deterministic transforms only (resize + normalize).

---

## 2. Augmentation Framework

Use `torchvision.transforms.v2` (transforms v2 API), which supports:
- **Joint transforms** for image + bboxes + masks + keypoints simultaneously.
- **Tensor and PIL** input.
- All standard augmentations.

For augmentations not available in torchvision v2 (e.g., `GridDistortion`, `ElasticTransform`), consider Albumentations as a fallback.

---

## 3. JSON Configuration

```json
{
  "augmentations": {
    "train": [
      { "name": "RandomResizedCrop", "params": { "size": [224, 224], "scale": [0.8, 1.0] } },
      { "name": "RandomHorizontalFlip", "params": { "p": 0.5 } },
      { "name": "RandomRotation", "params": { "degrees": 15 } },
      { "name": "ColorJitter", "params": { "brightness": 0.2, "contrast": 0.2 } },
      { "name": "ToImage", "params": {} },
      { "name": "Normalize", "params": { "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225] } }
    ],
    "val": [
      { "name": "Resize", "params": { "size": [256, 256] } },
      { "name": "CenterCrop", "params": { "size": [224, 224] } },
      { "name": "ToImage", "params": {} },
      { "name": "Normalize", "params": { "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225] } }
    ]
  }
}
```

---

## 4. Available Augmentations

### Geometric Transforms

| Name | Key Params | Applicable Tasks | Notes |
|------|-----------|-----------------|-------|
| `Resize` | `size: [H, W]` | All | Deterministic resize |
| `CenterCrop` | `size: [H, W]` | All | Deterministic center crop |
| `RandomResizedCrop` | `size, scale, ratio` | Classification, Regression, Anomaly | Crop + resize |
| `RandomHorizontalFlip` | `p` | All | Flip left-right |
| `RandomVerticalFlip` | `p` | All | Flip top-bottom |
| `RandomRotation` | `degrees` | All | Random rotation |
| `RandomAffine` | `degrees, translate, scale, shear` | All | Combined affine |
| `RandomPerspective` | `distortion_scale, p` | Classification, Regression | Perspective warp |
| `RandomCrop` | `size, padding` | All | Random crop (with optional padding) |

### Photometric Transforms

| Name | Key Params | Notes |
|------|-----------|-------|
| `ColorJitter` | `brightness, contrast, saturation, hue` | Random color changes |
| `RandomGrayscale` | `p` | Convert to grayscale |
| `GaussianBlur` | `kernel_size, sigma` | Gaussian blur |
| `RandomAutocontrast` | `p` | Auto-adjust contrast |
| `RandomEqualize` | `p` | Histogram equalization |
| `RandomPosterize` | `bits, p` | Reduce color bits |
| `RandomSolarize` | `threshold, p` | Invert pixels above threshold |

### Normalization / Conversion

| Name | Key Params | Notes |
|------|-----------|-------|
| `ToImage` | — | PIL / ndarray → Tensor (torchvision transforms v2) |
| `Normalize` | `mean, std` | Channel-wise normalization |

### Advanced (future)

| Name | Notes |
|------|-------|
| `Mosaic` | 4 images tiled (detection) |
| `MixUp` | Blend two images + labels |
| `CutMix` | Cut-paste between images |
| `GridDistortion` | Elastic grid deformation |

---

## 5. Pipeline Builder

```python
# app/datasets/augmentations.py

import torchvision.transforms.v2 as T

TRANSFORM_REGISTRY = {
    "Resize": T.Resize,
    "CenterCrop": T.CenterCrop,
    "RandomResizedCrop": T.RandomResizedCrop,
    "RandomHorizontalFlip": T.RandomHorizontalFlip,
    "RandomVerticalFlip": T.RandomVerticalFlip,
    "RandomRotation": T.RandomRotation,
    "ColorJitter": T.ColorJitter,
    "GaussianBlur": T.GaussianBlur,
    "ToImage": T.ToImage,
    "Normalize": T.Normalize,
    # ... etc
}

def build_augmentation_pipeline(config: list[dict]) -> T.Compose:
    transforms = []
    for item in config:
        cls = TRANSFORM_REGISTRY[item["name"]]
        transforms.append(cls(**item["params"]))
    return T.Compose(transforms)
```

---

## 6. Task-Specific Considerations

### Detection / Oriented Detection / Instance Segmentation

Spatial transforms must also transform bounding boxes, oriented boxes, and masks. `torchvision.transforms.v2` handles this automatically when using the new `TVTensor` types:

```python
from torchvision import tv_tensors

image = tv_tensors.Image(image_tensor)
boxes = tv_tensors.BoundingBoxes(box_tensor, format="XYXY", canvas_size=(H, W))
masks = tv_tensors.Mask(mask_tensor)

# Transforms are applied jointly
transformed = transform(image, boxes, masks)
```

### Segmentation

Masks must receive the **same** spatial transform (flip, crop, rotation) as the image, but **not** photometric transforms (color jitter, blur). This is handled automatically by `torchvision.transforms.v2`.

---

## 7. GUI: Augmentation Configuration

In the Training page center column, below the hyperparameter form:

### Layout

1. **"Augmentations" section header** with a toggle to expand/collapse.
2. **Augmentation list**: ordered list of current augmentation steps, each showing:
   - Transform name.
   - Key parameters (inline, editable).
   - Drag handle for reordering.
   - Delete button.
3. **"Add Augmentation" button**: opens a dropdown of available augmentations (filtered by task compatibility).
4. **"Reset to Defaults" button**: restores the task-specific default pipeline.
5. **Preview** (optional/future): show a sample image with the current pipeline applied (4–6 random previews).

### Validation / Test Pipeline

A separate collapsible section for the val/test pipeline. Typically just Resize + CenterCrop + Normalize. Users can edit but are warned if they deviate from standard practice.

---

## 8. Related Documents

- Training pipeline → [00-training-pipeline.md](00-training-pipeline.md)
- Hyperparameters → [01-hyperparameters.md](01-hyperparameters.md)
- Task-specific defaults → [../03-tasks/](../03-tasks/)
