# Data Layer — Dataset Management

This document describes how datasets are imported, stored, and accessed in AI Studio.

---

## 1. Import Flow

```
User clicks "Import" on Dataset page
           │
           ▼
   Select source folder
   (local filesystem path)
           │
           ▼
   Select annotation format
   (COCO / YOLO / CSV / Image Folders)
           │
           ▼
   Format parser reads annotations
   & validates against task type
           │
           ▼
   Images copied to project/dataset/images/
   Annotations merged into dataset.json
   (masks saved in dataset/masks/ if applicable)
           │
           ▼
   dataset.json created with metadata,
   image list, annotations, stats, and split fields
```

---

## 2. Unified `dataset.json`

All external formats (COCO, YOLO, CSV, Image Folders) are converted into a single **`dataset.json`** file that contains both metadata and per-image annotations. There is no separate `annotations.json`.

The only external files referenced by `dataset.json` are **mask images** for segmentation / instance segmentation tasks, which are stored in `dataset/masks/`.

### Full Schema

```json
{
  "version": "1.0",
  "id": "dataset-xyz789",
  "task": "classification",
  "source_format": "coco",
  "source_path": "C:/data/pcb_defects",
  "imported_at": "2026-02-19T10:05:00Z",
  "classes": ["cat", "dog", "bird"],
  "split_names": [],
  "image_stats": {
    "num_images": 1200,
    "min_width": 224,
    "max_width": 1024,
    "min_height": 224,
    "max_height": 1024,
    "formats": ["png", "jpg"]
  },
  "images": [
    {
      "filename": "img_0001.png",
      "width": 640,
      "height": 480,
      "split": [],
      "annotations": [
        {
          "type": "label",
          "class_id": 0,
          "class_name": "cat"
        }
      ]
    }
  ]
}
```

### Key Fields

| Field | Description |
|-------|-------------|
| `classes` | Ordered list of class names (index = `class_id`). Empty for regression. |
| `split_names` | List of split name strings (e.g., `["80-10-10", "70-15-15"]`). Each entry is a named split configuration. |
| `images[i].split` | List of strings, one per split in `split_names`. Values: `"train"`, `"val"`, `"test"`, or `"none"`. Empty list `[]` when no splits exist. |

See section 3 below for the split mechanism details.

---

## 3. Split Storage (Inline in `dataset.json`)

Splits are stored **inline** in `dataset.json` rather than in separate files. This keeps all dataset state in one place.

### How It Works

1. `split_names` is a top-level list that stores the name of each split. Index position is the split's identity.
2. Each image's `split` list has the same length as `split_names`. The value at index `i` is the subset assignment (`"train"` / `"val"` / `"test"` / `"none"`) for the split named by `split_names[i]`.

### Example: Two Splits

```json
{
  "split_names": ["80-10-10", "70-15-15"],
  "images": [
    {
      "filename": "img_0001.png",
      "split": ["train", "test"],
      "annotations": [{ "type": "label", "class_id": 0, "class_name": "cat" }]
    },
    {
      "filename": "img_0002.png",
      "split": ["val", "train"],
      "annotations": [{ "type": "label", "class_id": 1, "class_name": "dog" }]
    },
    {
      "filename": "img_0003.png",
      "split": ["test", "none"],
      "annotations": [{ "type": "label", "class_id": 2, "class_name": "bird" }]
    }
  ]
}
```

Reading: `img_0001.png` is in `train` for split "80-10-10" (index 0) and in `test` for split "70-15-15" (index 1).

### Referencing Splits in Experiments

Experiments reference a split by its **index** into `split_names` (stored as `split_index` in the experiment config). This is simpler than managing separate split IDs and files.

See [03-splits.md](03-splits.md) for full split creation / management details.

---

## 4. Annotation Types by Task

The `annotations` array per image uses a `type` field that varies by task:

| Task | `type` | Fields |
|------|--------|--------|
| Classification | `"label"` | `class_id`, `class_name` |
| Anomaly Detection | `"anomaly"` | `is_anomalous` (bool) |
| Object Detection | `"bbox"` | `class_id`, `class_name`, `bbox` `[x, y, w, h]` |
| Oriented OD | `"oriented_bbox"` | `class_id`, `class_name`, `bbox` `[cx, cy, w, h, angle]` |
| Segmentation | `"segmentation_mask"` | `mask_path` (relative path to PNG in `dataset/masks/`) |
| Instance Segmentation | `"instance"` | `class_id`, `class_name`, `mask_path` or `polygon` |
| Regression | `"value"` | `values` (list of one or more floats) |

Notes:
- **Anomaly Detection**: `mask_path` is not included in the initial implementation (may be added later for pixel-level localization).
- **Regression**: `values` is always a list. Single-output regression uses `[3.7]`, multi-output regression uses `[3.7, 1.2, 0.8]`.
- **Segmentation / Instance Segmentation**: `mask_path` references a file in `dataset/masks/` — this is the only case where annotation data lives outside `dataset.json`.

---

## 5. Re-import

If the user re-imports a dataset:

1. Warn that existing dataset will be replaced.
2. Warn if experiments reference the current dataset.
3. Delete existing `dataset/` folder contents (images, masks, thumbnails).
4. Run the import flow again.
5. Existing split assignments are lost (since `split_names` and `split` fields are reset).

---

## 6. Validation

On import, the parser validates:

- All referenced image files exist.
- Annotations are well-formed for the project's task type.
- Class names are consistent (no unnamed classes).
- Bounding boxes are within image bounds (warning, not error).
- For segmentation: mask dimensions match image dimensions.
- For regression: `values` contains at least one numeric value.

Validation errors are returned to the user as a summary (count of errors + first N examples).

---

## 7. Thumbnails

Thumbnails (150×150px) are generated on first access and cached in `dataset/.thumbs/`. This is an implementation detail — the API serves them transparently via `/api/datasets/{project_id}/thumbnails/{filename}`.

---

## 8. Related Documents

- Storage layout → [00-storage-layout.md](00-storage-layout.md)
- Supported formats → [02-dataset-formats.md](02-dataset-formats.md)
- Splits → [03-splits.md](03-splits.md)
- Dataset GUI page → [../10-gui/01-pages/02-dataset-page.md](../10-gui/01-pages/02-dataset-page.md)
- API endpoints → [../09-api/01-endpoints.md](../09-api/01-endpoints.md)
