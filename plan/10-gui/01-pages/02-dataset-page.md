# GUI — Dataset Page

**Route**: `/projects/{id}/dataset`

---

## 1. Purpose

Import datasets, browse images, and view annotations. This page is for data **visualization and import** — annotation editing is planned for a future phase.

---

## 2. Layout

```
┌────────────────────────────────────────────────────────┐
│  AI Studio  │ Project: "Cats vs Dogs" │ Classification │
├────────────────────────────────────────────────────────┤
│  [Proj] [Data] [Split] [Train] [Eval] [Export]         │
├────────────────────────────────────────────────────────┤
│  Dataset  [Import ▼]  [Stats]                          │
│                                                        │
│  Filter: [Class ▼] [Search ___________] Sort: [Name ▼] │
├────────────────────────────────────────────────────────┤
│ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐  │
│ │ img1 │ │ img2 │ │ img3 │ │ img4 │ │ img5 │ │ img6 │  │
│ │ cat  │ │ dog  │ │ cat  │ │ bird │ │ dog  │ │ cat  │  │
│ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘  │
│ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐  │
│ │ img7 │ │ img8 │ │ img9 │ │img10 │ │img11 │ │img12 │  │
│ │ dog  │ │ cat  │ │ dog  │ │ cat  │ │ bird │ │ dog  │  │
│ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘  │
│                                                        │
│  ◀ 1 2 3 4 5 ... 20 ▶              50 images/page     │
└────────────────────────────────────────────────────────┘
```

---

## 3. Import Flow

### Import Button (Dropdown)

- **From Local Path**: Enter path to image folder on the server.
- **Upload ZIP**: Upload a ZIP archive from the browser.

### Import Modal

```
┌──────────────────────────────────────┐
│  Import Dataset                      │
├──────────────────────────────────────┤
│                                      │
│  Source: [Local Path ▼]              │
│                                      │
│  Path: [/data/cats_dogs_dataset]     │
│                                      │
│  Format: [Auto-detect ▼]             │
│    ○ Auto-detect                     │
│    ○ COCO JSON                       │
│    ○ YOLO                            │
│    ○ CSV                             │
│    ○ Image Folder                    │
│                                      │
│         [Cancel]  [Import]           │
└──────────────────────────────────────┘
```

After import:
- Progress bar shows copy/conversion progress.
- On completion, image grid refreshes to show all images.
- Stats panel updates with class distribution.

---

## 4. Image Grid

- **Thumbnail** (128×128px max) with lazy loading.
- **Annotation overlay** (task-dependent):
  - Classification: class label badge at bottom.
  - Detection: bounding boxes drawn on thumbnail.
  - Segmentation: mask overlay on thumbnail.
  - Regression: values badge at bottom.
- **Selection**: click to open image detail view.

### Loading

- Uses HTMX pagination: clicking page number loads a new grid fragment.
- Grid container: `<div id="image-grid" hx-get="..." hx-trigger="...">`.

---

## 5. Image Detail View

Clicking an image opens a detail panel (right-side overlay or modal):

```
┌──────────────────────────────────────┐
│  ◀ img_0042.jpg                  ✕  │
├──────────────────────────────────────┤
│                                      │
│          [Full-size image]           │
│     (with annotation overlay)        │
│                                      │
├──────────────────────────────────────┤
│  Filename: img_0042.jpg              │
│  Size: 640 × 480                     │
│  File size: 128 KB                   │
│  Class: cat                          │
│  Split: train (80-10-10)             │
│                                      │
│       [◀ Previous] [Next ▶]         │
└──────────────────────────────────────┘
```

- Shows full annotation details for the image.
- Left/right navigation between images.
- For detection: shows box coordinates + class for each object.
- For segmentation: shows mask overlay with class legend.

---

## 6. Stats Panel

Toggle button shows/hides a stats card:

| Stat | Content |
|------|---------|
| Total images | 1,200 |
| Classes | 3 (cat: 420, dog: 410, bird: 370) |
| Image sizes | Min: 320×240, Max: 1920×1080, Avg: 800×600 |
| File formats | JPEG: 1100, PNG: 100 |

For classification, includes a **class distribution bar chart** (horizontal bars, one per class).

---

## 7. Toolbar Controls

| Control | Type | Description |
|---------|------|-------------|
| Class filter | Dropdown | Show only images of a specific class |
| Search | Text input | Filter by filename |
| Sort | Dropdown | `name`, `class`, `size` |
| Grid size | Slider/toggle | Small / medium / large thumbnails |
| View mode | Toggle | Grid / List |

---

## 8. Empty State

If no dataset is imported:

```
┌────────────────────────────────────────┐
│                                        │
│       📁 No dataset imported           │
│                                        │
│    Import images to get started        │
│                                        │
│         [Import Dataset]               │
│                                        │
└────────────────────────────────────────┘
```

---

## 9. Related Documents

- Dataset management → [../../02-data-layer/01-dataset-management.md](../../02-data-layer/01-dataset-management.md)
- Dataset formats → [../../02-data-layer/02-dataset-formats.md](../../02-data-layer/02-dataset-formats.md)
- Dataset API → [../../09-api/01-endpoints.md](../../09-api/01-endpoints.md#2-datasets)
