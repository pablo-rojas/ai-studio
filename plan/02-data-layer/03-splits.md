# Data Layer — Splits

This document defines how dataset splits (train / val / test) are created, stored, and managed. Splits are stored **inline in `dataset.json`** — there are no separate split files or folders.

---

## 1. Overview

A split divides the dataset's images into non-overlapping subsets: **train**, **val**, and **test**. Each project can have multiple named splits, all stored inside `dataset.json` via the `split_names` and per-image `split` fields.

---

## 2. Storage Mechanism

See [01-dataset-management.md](01-dataset-management.md) for the full `dataset.json` schema. The split-relevant fields are:

### Top-Level

```json
{
  "split_names": ["80-10-10", "70-15-15"]
}
```

`split_names` is an ordered list. The index of each entry is the split's identity.

### Per-Image

```json
{
  "filename": "img_0001.png",
  "split": ["train", "test"],
  "annotations": [{ "..." }]
}
```

`split[i]` is the subset assignment for the split at `split_names[i]`. Valid values: `"train"`, `"val"`, `"test"`, `"none"`.

When no splits exist, `split_names` is `[]` and every image's `split` is `[]`.

---

## 3. Split Algorithm

All splits use **stratified splitting** to ensure each subset (train / val / test) has approximately the same class distribution as the full dataset.

- User specifies percentages (e.g., 70% train, 20% val, 10% test).
- Uses `sklearn.model_selection.StratifiedShuffleSplit` (or equivalent logic).
- A user-provided or auto-generated random seed ensures reproducibility.
- Requires class labels — applicable to classification, detection (dominant class), anomaly detection (normal/anomalous).
- For tasks without discrete classes (regression), falls back to binned stratification (divide value range into bins, stratify on bins).

### 3.1 Anomaly Detection Split (Semi-Supervised Constraint)

Anomaly detection is **semi-supervised** — the model trains exclusively on normal (good) images. The split engine must enforce this constraint automatically.

#### Rules

1. **Normal images** (`is_anomalous == false`) follow the user-specified train/val/test percentages as usual.
2. **Anomalous images** (`is_anomalous == true`) are **never placed in the train set**. They are distributed between **val and test only**, proportionally to the val/test ratio from the user's percentages.

#### Ratio Derivation

Given user ratios `T%-V%-E%` (train-val-test):
- Normal images: `T%` train, `V%` val, `E%` test (as specified).
- Anomalous images: `V/(V+E)` val, `E/(V+E)` test.

| User Percentages | Normal Images | Anomalous Images |
|------------------|---------------|------------------|
| 70-15-15 | 70% train, 15% val, 15% test | 50% val, 50% test |
| 70-20-10 | 70% train, 20% val, 10% test | 67% val, 33% test |
| 80-10-10 | 80% train, 10% val, 10% test | 50% val, 50% test |
| 60-30-10 | 60% train, 30% val, 10% test | 75% val, 25% test |

#### Implementation Notes

- The split function checks the project's task type. If `task == "anomaly"`, it activates the semi-supervised split logic.
- The split preview (shown before saving) must clearly display the separation: "Normal: X train / Y val / Z test" and "Anomalous: 0 train / A val / B test".
- Stratification is applied within the normal-image pool for train/val/test and within the anomalous-image pool for val/test.

See also: [../03-tasks/02-anomaly-detection.md](../03-tasks/02-anomaly-detection.md) — Section 13.

---

## 4. Split Creation Flow

```
User navigates to Split page
           │
           ▼
   Clicks "New Split"
           │
           ▼
   Enters split name
           │
           ▼
   Set ratios + optional seed
           │
           ▼
   Preview stats → Save
```

When saved:
1. The split name is appended to `split_names`.
2. Each image's `split` list is extended by one element with its assigned subset (`"train"` / `"val"` / `"test"` / `"none"`).
3. `dataset.json` is written atomically.

---

## 5. Referencing Splits in Experiments

Experiments store a `split_index` (integer) that points into `split_names`:

```json
{
  "split_index": 0,
  "...": "..."
}
```

This replaces the previous `split_id` approach. The experiment reads `split_names[split_index]` to resolve the split name, and filters images by `image.split[split_index]` to get each subset.

---

## 6. Managing Existing Splits

### Immutability

Splits are **immutable once created** to ensure experiment reproducibility. To modify:
- Create a new split with different settings.
- The old split can be deleted (see below).

### Deleting a Split

1. Remove the entry from `split_names` at the target index.
2. Remove the element at that index from every image's `split` list.
3. Re-index any experiments that reference a higher `split_index` (decrement by 1).
4. Warn if experiments reference the split being deleted.

### Listing Splits

The Split page reads `split_names` from `dataset.json` and computes per-split statistics dynamically by iterating over the image list.

---

## 7. Edge Cases

| Edge Case | Handling |
|-----------|----------|
| Dataset has < 3 images | Warn: at least one image per subset. Allow single-image subsets but show warning. |
| Class has only 1 image | Split assigns it to train. Warn the user. |
| All images same class | All classes identical — split proceeds as a simple random shuffle. Info message shown. |
| Anomaly task: anomalous image in train | Blocked by automatic split. |
| Anomaly task: no anomalous images | All images are normal — split proceeds as standard stratified. Warning: evaluation metrics will lack anomaly samples. |
| Dataset re-imported | `split_names` is reset to `[]` and all `split` fields are reset to `[]`. Experiments become invalid. |
| No splits exist | `split_names` is `[]`. Training page prompts user to create one. |

---

## 8. Implementation Notes

- Splits are computed server-side in `app/datasets/splits.py`.
- All split data is persisted in `dataset.json` — no separate split files or folders.
- Computing split statistics (counts, class distribution per subset) is done on-the-fly by iterating the image list filtered by `split[index]`.
- For anomaly detection projects, the split function partitions images by `is_anomalous` before applying the ratio logic (see Section 3.1).

---

## 9. Related Documents

- Dataset management (`dataset.json` schema) → [01-dataset-management.md](01-dataset-management.md)
- Storage layout → [00-storage-layout.md](00-storage-layout.md)
- Training config (references split) → [../05-training/01-hyperparameters.md](../05-training/01-hyperparameters.md)
- Split GUI page → [../10-gui/01-pages/03-split-page.md](../10-gui/01-pages/03-split-page.md)
