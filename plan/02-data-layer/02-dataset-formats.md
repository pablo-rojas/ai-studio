# Data Layer — Dataset Formats

This document lists all supported external annotation formats and how each is mapped to the internal annotation representation.

---

## 1. Supported Formats

| Format | File Type | Applicable Tasks |
|--------|-----------|-----------------|
| **COCO JSON** | Single `.json` file | Classification, Object Detection, Instance Segmentation, Segmentation |
| **YOLO** | Per-image `.txt` files + `classes.txt` | Object Detection, Oriented Object Detection |
| **CSV** | Single `.csv` file | Classification, Regression |
| **Image Folders** | Subfolders of images (one per class) | Classification, Anomaly Detection |

---

## 2. COCO JSON

### Source Structure

```
dataset/
├── images/
│   ├── img_001.jpg
│   └── ...
└── annotations.json        # or instances_train.json, etc.  (source file — converted to dataset.json on import)
```

### Mapping

| COCO Field | Internal Field |
|-----------|---------------|
| `images[].file_name` | `images[].filename` |
| `images[].width/height` | `images[].width/height` |
| `annotations[].category_id` | Look up `categories[]` → `class_id`, `class_name` |
| `annotations[].bbox` `[x, y, w, h]` | `annotations[].bbox` (same format) |
| `annotations[].segmentation` (polygon) | `annotations[].polygon` (for instance segmentation) |
| `categories[].name` | `classes[]` |

### Notes

- For **classification**: if each image has exactly one annotation, extract the category as the label.
- For **segmentation**: if RLE masks are provided, decode them to PNG masks and save in `dataset/masks/`.
- `iscrowd` annotations are skipped.

---

## 3. YOLO

### Source Structure

```
dataset/
├── images/
│   ├── img_001.jpg
│   └── ...
├── labels/
│   ├── img_001.txt
│   └── ...
└── classes.txt             # One class name per line
```

### Label File Format

```
# class_id center_x center_y width height  (normalized 0-1)
0 0.5 0.4 0.3 0.2
1 0.7 0.8 0.1 0.15
```

For **oriented detection** (OBB extension):

```
# class_id center_x center_y width height angle_degrees
0 0.5 0.4 0.3 0.2 45.0
```

### Mapping

| YOLO Field | Internal Field |
|-----------|---------------|
| `class_id` (line index) | Look up `classes.txt` → `class_id`, `class_name` |
| `cx, cy, w, h` (normalized) | Denormalize to pixel coords → `bbox [x, y, w, h]` |
| `angle` (if present) | `bbox [cx, cy, w, h, angle]` for oriented detection |

### Notes

- Image dimensions must be read from the actual image files to denormalize coordinates.
- Images without a corresponding `.txt` file are treated as having no annotations (negative samples).

---

## 4. CSV

### Source Structure

```
dataset/
├── images/
│   ├── img_001.jpg
│   └── ...
└── labels.csv
```

### CSV Format — Classification

```csv
filename,label
img_001.jpg,cat
img_002.jpg,dog
```

### CSV Format — Regression

```csv
filename,value
img_001.jpg,3.7
img_002.jpg,1.2
```

### Mapping

| CSV Column | Internal Field |
|-----------|---------------|
| `filename` | `images[].filename` |
| `label` | `class_name` → resolve `class_id` |
| `value` | `annotations[].value` (float) |

### Notes

- Column names are case-insensitive.
- Extra columns are ignored with a warning.
- Missing images or labels produce validation errors.

---

## 5. Image Folders

### Overview

A simple folder-based format where the directory structure itself encodes the labels. Each subfolder name becomes a class name and all images inside it are assigned that class.

### Source Structure — Classification

One root folder containing one subfolder per class:

```
dataset/
├── cats/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
├── dogs/
│   ├── img_003.jpg
│   └── ...
└── birds/
    ├── img_004.jpg
    └── ...
```

Each subfolder name (`cats`, `dogs`, `birds`) is used as the class name. The number of subfolders is unrestricted.

### Source Structure — Anomaly Detection

Two subfolders: one for **normal** images and one for **anomalous** images:

```
dataset/
├── good/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
└── anomaly/
    ├── img_003.jpg
    └── ...
```

The importer recognizes common naming variants (case-insensitive):

| Semantic | Accepted Folder Names |
|----------|----------------------|
| Normal   | `good`, `ok`, `normal`, `pass` |
| Anomalous | `anomaly`, `anomalous`, `bad`, `nok`, `defect`, `defective`, `ng`, `fail` |

If exactly two subfolders are found and each matches one of the two groups above, the format is detected as anomaly-detection folders. Any unrecognized folder name triggers a validation error asking the user to rename or manually assign the mapping.

### Mapping

| Folder Data | Internal Field |
|------------|---------------|
| Subfolder name | `class_name` → resolve `class_id` |
| Image file path | `images[].filename` |
| Image dimensions | Read from file → `images[].width`, `images[].height` |

### Notes

- Nested subfolders beyond one level are **not** supported; only direct children of the root folder are treated as classes.
- Hidden folders/files (names starting with `.`) are ignored.
- For anomaly detection the normal class is always assigned `class_id = 0`.
- Empty subfolders produce a validation warning.
- Supported image extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`.

---

## 6. Format Detection (Auto-detect)

When the user selects a source folder, the importer can attempt auto-detection:

1. Look for `*.json` → try COCO.
2. Look for `labels/*.txt` + `classes.txt` → try YOLO.
3. Look for `labels.csv` or `*.csv` → try CSV.
4. If the root contains only subfolders (no loose files other than images), and each subfolder contains images → try Image Folders.

If ambiguous, ask the user to choose. Format is always overridable.

---

## 7. Related Documents

- Internal annotation format → [01-dataset-management.md](01-dataset-management.md)
- Storage layout → [00-storage-layout.md](00-storage-layout.md)
- Task-specific annotation details → [../03-tasks/](../03-tasks/)
