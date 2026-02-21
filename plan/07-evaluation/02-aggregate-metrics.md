# Evaluation — Aggregate Metrics

This document describes the aggregate (dataset-level) metrics computed after evaluation.

---

## 1. Aggregate Metrics by Task

### Classification

```json
{
  "accuracy": 0.956,
  "f1_macro": 0.951,
  "precision_macro": 0.948,
  "recall_macro": 0.953,
  "confusion_matrix": [[38, 1, 1], [0, 40, 1], [2, 0, 37]],
  "per_class": {
    "cat": { "precision": 0.95, "recall": 0.95, "f1": 0.95, "support": 40 },
    "dog": { "precision": 0.98, "recall": 0.98, "f1": 0.98, "support": 41 },
    "bird": { "precision": 0.95, "recall": 0.95, "f1": 0.95, "support": 39 }
  }
}
```

### Anomaly Detection

```json
{
  "auroc": 0.94,
  "f1": 0.91,
  "precision": 0.89,
  "recall": 0.93,
  "optimal_threshold": 0.52,
  "num_normal": 80,
  "num_anomalous": 40
}
```

### Object Detection / Oriented OD

```json
{
  "mAP_50": 0.78,
  "mAP_50_95": 0.62,
  "precision": 0.82,
  "recall": 0.75,
  "per_class_AP": {
    "car": 0.85,
    "person": 0.72,
    "bicycle": 0.77
  },
  "total_gt": 350,
  "total_predictions": 380,
  "total_tp": 290,
  "total_fp": 90,
  "total_fn": 60
}
```

### Segmentation

```json
{
  "mIoU": 0.72,
  "pixel_accuracy": 0.89,
  "mean_dice": 0.78,
  "per_class_IoU": {
    "background": 0.95,
    "road": 0.82,
    "building": 0.68,
    "vegetation": 0.71,
    "car": 0.45
  }
}
```

### Instance Segmentation

```json
{
  "mAP_mask_50": 0.65,
  "mAP_mask_50_95": 0.48,
  "mAP_box_50": 0.72,
  "mAP_box_50_95": 0.55,
  "per_class_AP_mask": {
    "person": 0.70,
    "car": 0.60
  }
}
```

### Regression

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

## 2. Metric Computation Libraries

| Metric | Library |
|--------|---------|
| Accuracy, F1, Precision, Recall | `torchmetrics` |
| Confusion Matrix | `torchmetrics.ConfusionMatrix` |
| mAP (detection) | `torchmetrics.detection.MeanAveragePrecision` |
| IoU / mIoU (segmentation) | `torchmetrics.JaccardIndex` |
| Dice | `torchmetrics.Dice` |
| AUROC | `torchmetrics.AUROC` |
| MAE, MSE, R² | `torchmetrics.MeanAbsoluteError`, `MeanSquaredError`, `R2Score` |

---

## 3. Visualization (Evaluation Page)

### Metrics Summary Panel

A card showing all aggregate metrics in a table:

```
┌──────────────────────────────┐
│  Evaluation Results          │
├──────────────────────────────┤
│  Accuracy:    95.6%          │
│  F1 (macro):  95.1%          │
│  Precision:   94.8%          │
│  Recall:      95.3%          │
│  Loss:         0.118         │
└──────────────────────────────┘
```

### Confusion Matrix (Classification)

- Rendered as a heatmap table.
- Rows = ground truth, columns = predicted.
- Cell color intensity proportional to count.
- Diagonal (correct) in green tones, off-diagonal (errors) in red tones.

### Per-Class Bar Chart

- Horizontal bar chart showing per-class metric (accuracy, AP, IoU) for each class.
- Sorted from best to worst.
- Helps identify which classes the model struggles with.

### Scatter Plot (Regression)

- X-axis: ground truth value.
- Y-axis: predicted value.
- Diagonal line = perfect prediction.
- Points colored by error magnitude.

---

## 4. Storing Aggregate Metrics

Aggregate metrics are computed over the **combined pool** of all selected split subsets (e.g., if the user selected `["test", "val"]`, metrics are computed over all images from both subsets together — no per-subset breakdown).

Aggregate metrics are stored in a separate `aggregate.json` file inside the experiment's evaluation subfolder:

```
experiments/<experiment-id>/evaluation/aggregate.json
```

```json
{
  "accuracy": 0.956,
  "f1_macro": 0.951,
  "precision_macro": 0.948,
  "recall_macro": 0.953,
  "confusion_matrix": [[38, 1, 1], [0, 40, 1], [2, 0, 37]],
  "per_class": {
    "cat": { "precision": 0.95, "recall": 0.95, "f1": 0.95, "support": 40 },
    "dog": { "precision": 0.98, "recall": 0.98, "f1": 0.98, "support": 41 },
    "bird": { "precision": 0.95, "recall": 0.95, "f1": 0.95, "support": 39 }
  }
}
```

---

## 5. Related Documents

- Per-image results → [01-per-image-results.md](01-per-image-results.md)
- Evaluation pipeline → [00-evaluation-pipeline.md](00-evaluation-pipeline.md)
- Task-specific metrics → [../03-tasks/](../03-tasks/)
- Evaluation GUI page → [../10-gui/01-pages/05-evaluation-page.md](../10-gui/01-pages/05-evaluation-page.md)
