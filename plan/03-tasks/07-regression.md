# Tasks — Regression

**Phase**: 5

---

## 1. Task Description

Predict one or more continuous numeric values from an image. Use cases include age estimation, quality scoring, count estimation, measurement prediction, multi-dimensional property estimation, etc.

---

## 2. Label Schema

```json
{
  "type": "value",
  "values": [3.7]
}
```

Multi-output example:
```json
{
  "type": "value",
  "values": [3.7, 1.2, 0.8]
}
```

- Each image has exactly one annotation of type `"value"`.
- `values` is always a list of one or more floats (can be negative, zero, or positive).
- Single-output regression: `[3.7]`. Multi-output regression: `[3.7, 1.2, 0.8]`.
- The number of outputs (`len(values)`) must be consistent across all images in the dataset.

---

## 3. Compatible Dataset Formats

| Format | Notes |
|--------|-------|
| **CSV** | `filename, value` columns |

---

## 4. Compatible Architectures

Any backbone + regression head:

| Backbone | Notes |
|----------|-------|
| ResNet-18/34/50 | General purpose |
| EfficientNet-B0/B3 | Parameter-efficient |
| MobileNetV3 | Lightweight |

### Head: Regression Head

```
backbone features (N, C) → Global Average Pool → Dropout(p) → Linear(C, num_outputs) → (N, num_outputs)
```

- `num_outputs` is determined by `len(values)` in the dataset (1 for single-output, >1 for multi-output).
- No activation (raw scalar predictions).
- Dropout rate is configurable (default `0.2`).

---

## 5. Loss Functions

| Loss | When to Use |
|------|------------|
| **MSELoss** (default) | Standard regression, penalizes large errors |
| **L1Loss (MAE)** | More robust to outliers |
| **SmoothL1Loss (Huber)** | Mix of MSE and L1 — robust, smooth gradients |

---

## 6. Metrics

| Metric | Role |
|--------|------|
| **MAE** | Primary — Mean Absolute Error |
| **MSE** | Secondary |
| **RMSE** | Secondary |
| **R²** | Secondary — coefficient of determination |

---

## 7. Default Augmentations

### Training

```json
[
  { "name": "RandomResizedCrop", "params": { "size": [224, 224], "scale": [0.8, 1.0] } },
  { "name": "RandomHorizontalFlip", "params": { "p": 0.5 } },
  { "name": "ColorJitter", "params": { "brightness": 0.2, "contrast": 0.2 } },
  { "name": "ToTensor", "params": {} },
  { "name": "Normalize", "params": { "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225] } }
]
```

### Validation / Test

```json
[
  { "name": "Resize", "params": { "size": [256, 256] } },
  { "name": "CenterCrop", "params": { "size": [224, 224] } },
  { "name": "ToTensor", "params": {} },
  { "name": "Normalize", "params": { "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225] } }
]
```

---

## 8. Default Hyperparameters

```json
{
  "optimizer": "adam",
  "learning_rate": 0.001,
  "weight_decay": 0.0001,
  "scheduler": "cosine",
  "batch_size": 32,
  "max_epochs": 50,
  "early_stopping_patience": 10,
  "loss": "mse",
  "dropout": 0.2,
  "pretrained": true,
  "freeze_backbone": false
}
```

---

## 9. Visualization

### Dataset Page
- **Value badge**: displayed in the top-left corner of the thumbnail (e.g., "3.7").
- Optionally color-coded by value range (e.g., gradient from blue = low to red = high).

### Evaluation Page
- **Predicted value vs. ground truth** displayed on each image.
- **Error value** shown (prediction - ground truth).
- **Scatter plot** of predicted vs. actual values (aggregate view).
- Color coding: green for small errors, red for large errors.

---

## 10. Evaluation Specifics

Per-image result:

```json
{
  "filename": "img_001.png",
  "ground_truth": [3.7],
  "prediction": [3.5],
  "error": [-0.2],
  "absolute_error": [0.2]
}
```

Multi-output example:
```json
{
  "filename": "img_001.png",
  "ground_truth": [3.7, 1.2, 0.8],
  "prediction": [3.5, 1.3, 0.7],
  "error": [-0.2, 0.1, -0.1],
  "absolute_error": [0.2, 0.1, 0.1]
}
```

Aggregate metrics:

```json
{
  "mae": 0.35,
  "mse": 0.18,
  "rmse": 0.42,
  "r_squared": 0.92,
  "min_error": -1.2,
  "max_error": 0.8,
  "median_absolute_error": 0.28
}
```

---

## 11. Normalization

Consider normalizing target values (z-score or min-max) before training to improve convergence:

- **z-score**: `(value - mean) / std` — computed on training set only.
- Store `mean` and `std` in the experiment config for inverse transform at inference time.
- The `LightningModule` handles denormalization when computing metrics and outputting predictions.

---

## 12. Related Documents

- Task registry → [00-task-registry.md](00-task-registry.md)
- Classification (similar architecture, different head) → [01-classification.md](01-classification.md)
- Architecture catalog → [../04-models/00-architecture-catalog.md](../04-models/00-architecture-catalog.md)
