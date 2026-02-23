# Evaluation — Per-Image Results

This document describes how per-image predictions are stored and displayed.

---

## 1. Overview

After running evaluation, all per-image predictions are stored in a single `results.json` file inside the experiment's `evaluation/` subfolder (`experiments/<exp-id>/evaluation/results.json`). This avoids creating thousands of tiny files for large datasets (10k+ images), which would be slow to list, sort, and paginate server-side.

When multiple split subsets are evaluated together (e.g., `["test", "val"]`), all images are pooled into the same `results.json`. Each per-image entry includes a `"subset"` field to identify which subset the image belongs to, enabling filtering by subset in the GUI.

On the Evaluation page, the per-image list stays compact (thumbnail + filename + status + subset). Detailed fields are shown in a modal detail view loaded on demand for a selected filename.

---

## 2. Per-Image JSON Schema (by Task)

Every per-image entry includes a `"subset"` field (e.g., `"test"`, `"val"`, `"train"`) identifying which split subset the image comes from.

### Classification

```json
{
  "filename": "img_001.png",
  "subset": "test",
  "ground_truth": { "class_id": 0, "class_name": "cat" },
  "prediction": { "class_id": 0, "class_name": "cat", "confidence": 0.94 },
  "correct": true,
  "probabilities": { "cat": 0.94, "dog": 0.04, "bird": 0.02 }
}
```

### Anomaly Detection

```json
{
  "filename": "img_001.png",
  "subset": "test",
  "ground_truth": { "is_anomalous": true },
  "prediction": { "is_anomalous": true, "anomaly_score": 0.87 },
  "correct": true
}
```

### Object Detection

```json
{
  "filename": "img_001.png",
  "subset": "test",
  "ground_truth": [
    { "class_name": "car", "bbox": [120, 50, 200, 150] }
  ],
  "predictions": [
    { "class_name": "car", "bbox": [118, 48, 205, 152], "confidence": 0.92, "matched_gt_idx": 0 }
  ],
  "true_positives": 1,
  "false_positives": 0,
  "false_negatives": 0
}
```

### Oriented Object Detection

```json
{
  "filename": "img_001.png",
  "subset": "test",
  "ground_truth": [
    { "class_name": "ship", "bbox": [320, 240, 100, 40, 35.0] }
  ],
  "predictions": [
    { "class_name": "ship", "bbox": [318, 242, 102, 38, 34.5], "confidence": 0.88, "matched_gt_idx": 0 }
  ],
  "true_positives": 1,
  "false_positives": 0,
  "false_negatives": 0
}
```

### Segmentation

```json
{
  "filename": "img_001.png",
  "subset": "test",
  "predicted_mask_path": "pred_masks/img_001_pred.png",
  "pixel_accuracy": 0.94,
  "per_class_iou": { "background": 0.97, "road": 0.91, "building": 0.88 },
  "miou": 0.92
}
```

### Instance Segmentation

```json
{
  "filename": "img_001.png",
  "subset": "test",
  "ground_truth": [
    { "class_name": "person", "bbox": [50, 30, 120, 250] }
  ],
  "predictions": [
    { "class_name": "person", "bbox": [48, 28, 122, 252], "confidence": 0.91, "mask_iou": 0.85, "matched_gt_idx": 0 }
  ],
  "true_positives": 1,
  "false_positives": 0,
  "false_negatives": 0
}
```

### Regression

```json
{
  "filename": "img_001.png",
  "subset": "test",
  "ground_truth": [3.7],
  "prediction": [3.5],
  "error": [-0.2],
  "absolute_error": [0.2]
}
```

- Regression values are always stored as lists (single-output = list length 1, multi-output = list length > 1).

---

## 3. Prediction Overlays

For the Evaluation page's per-image detail view, prediction overlays are rendered client-side:

| Task | Overlay Rendering |
|------|------------------|
| Classification | Predicted class badge (green if correct, red if wrong) + confidence bar |
| Anomaly Detection | Anomaly score badge (color gradient) |
| Object Detection | Predicted boxes (solid) + GT boxes (dashed) + TP/FP/FN coloring |
| Oriented OD | Rotated predicted boxes + GT boxes |
| Segmentation | Predicted mask overlay + error map |
| Instance Segmentation | Per-instance masks + boxes |
| Regression | Predicted vs. actual values badge + error display |

### Rendering Approach

- The original image is loaded in an `<img>` tag.
- A `<canvas>` overlay draws annotations.
- JavaScript reads the per-image JSON and draws using Canvas 2D API.
- For segmentation masks: the predicted mask PNG is loaded as a semi-transparent overlay.

---

## 4. Sorting & Filtering

The Evaluation page result grid supports:

| Filter/Sort | Description |
|-------------|-------------|
| **Sort by confidence** (asc/desc) | See low-confidence predictions first |
| **Sort by error** (asc/desc) | See largest errors first (regression) |
| **Filter: correct only** | Show only correctly predicted images |
| **Filter: incorrect only** | Show only errors for debugging |
| **Filter by class** | Show only images of a specific class |
| **Filter by GT class** | Filter by ground truth class |
| **Filter by predicted class** | Filter by what the model predicted |
| **Filter by subset** | Show only images from a specific split subset (test, val, train) |

---

## 5. Pagination

- Per-image results are loaded in pages (50 per page, configurable).
- The API endpoint `/api/evaluation/{project_id}/{experiment_id}/results` accepts `page`, `page_size`, `sort_by`, `filter_*` parameters.
- The endpoint reads `results.json`, applies filters/sorting in memory, and returns the requested page slice.
- The detail endpoint `/api/evaluation/{project_id}/{experiment_id}/results/{filename}/info` uses the same query parameters to preserve context and compute previous/next navigation within the current page.
- `results.json` contains a top-level `"results"` array with one entry per image.

### `results.json` Structure

```json
{
  "results": [
    { "filename": "img_001.png", "...per-image fields..." },
    { "filename": "img_002.png", "...per-image fields..." }
  ]
}
```

---

## 6. Related Documents

- Evaluation pipeline → [00-evaluation-pipeline.md](00-evaluation-pipeline.md)
- Aggregate metrics → [02-aggregate-metrics.md](02-aggregate-metrics.md)
- Evaluation GUI page → [../10-gui/01-pages/05-evaluation-page.md](../10-gui/01-pages/05-evaluation-page.md)
