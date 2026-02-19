# Data Layer — Storage Layout

This document defines the folder structure for **user data** (projects, datasets, runs, exports). All data lives under a single workspace root with no external database.

---

## 1. Workspace Root

The workspace root defaults to `./workspace` relative to the application but is configurable via `WORKSPACE_ROOT` in [app/config.py](../../app/config.py).

```
workspace/
├── workspace.json                  # Global workspace metadata
└── projects/
    ├── <project-id>/               # One folder per project
    │   ├── project.json            # Project metadata (name, task, created, etc.)
    │   │
    │   ├── dataset/                # Imported dataset
    │   │   ├── dataset.json        # All metadata, images, annotations, and splits
    │   │   ├── images/             # Image files (copied or symlinked)
    │   │   │   ├── img_0001.png
    │   │   │   ├── img_0002.jpg
    │   │   │   └── ...
    │   │   ├── masks/              # Segmentation masks (if applicable)
    │   │   │   └── ...
    │   │   └── .thumbs/            # Cached thumbnails (auto-generated)
    │   │       └── ...
    │   │
    │   ├── experiments/            # Training experiments
    │   │   ├── experiments_index.json
    │   │   ├── <experiment-id>/
    │   │   │   ├── experiment.json # Experiment config (model, hparams, augmentations)
    │   │   │   └── runs/           # Training runs for this experiment
    │   │   │       ├── <run-id>/
    │   │   │       │   ├── run.json        # Run metadata (status, start/end time)
    │   │   │       │   ├── config.json     # Frozen snapshot of experiment config
    │   │   │       │   ├── metrics.json    # Per-epoch metrics
    │   │   │       │   ├── checkpoints/
    │   │   │       │   │   ├── best.ckpt
    │   │   │       │   │   └── last.ckpt
    │   │   │       │   └── logs/
    │   │   │       │       └── training.log
    │   │   │       └── ...
    │   │   └── ...
    │   │
    │   ├── evaluations/            # Evaluation results
    │   │   ├── evaluations_index.json
    │   │   ├── <evaluation-id>/
    │   │   │   ├── evaluation.json   # Eval config (checkpoint, split, params)
    │   │   │   ├── results.json      # Aggregate metrics
    │   │   │   └── per_image/        # Per-image prediction details
    │   │   │       ├── img_0001.json
    │   │   │       └── ...
    │   │   └── ...
    │   │
    │   └── exports/                # Exported models
    │       ├── exports_index.json
    │       ├── <export-id>/
    │       │   ├── export.json     # Export config (format, checkpoint, settings)
    │       │   └── model.onnx      # The exported model file
    │       └── ...
    │
    └── ...
```

---

## 2. JSON File Schemas

### `workspace.json`

```json
{
  "version": "1.0",
  "created_at": "2026-02-19T10:00:00Z",
  "projects": ["project-abc123", "project-def456"]
}
```

### `project.json`

```json
{
  "id": "project-abc123",
  "name": "Defect Classifier",
  "task": "classification",
  "description": "Classify PCB defects into 5 categories",
  "created_at": "2026-02-19T10:00:00Z",
  "updated_at": "2026-02-19T12:30:00Z"
}
```

Valid values for `task`:
- `"classification"`
- `"anomaly_detection"`
- `"object_detection"`
- `"oriented_object_detection"`
- `"segmentation"`
- `"instance_segmentation"`
- `"regression"`

### `dataset.json`

This is the single source of truth for dataset metadata, images, annotations, and split assignments. See [01-dataset-management.md](01-dataset-management.md) for the full schema.

```json
{
  "version": "1.0",
  "id": "dataset-xyz789",
  "task": "classification",
  "source_format": "coco",
  "source_path": "C:/data/pcb_defects",
  "imported_at": "2026-02-19T10:05:00Z",
  "classes": ["bridge", "scratch", "missing", "short", "spur"],
  "split_names": ["Default 70/20/10"],
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
      "split": ["train"],
      "annotations": [
        { "type": "label", "class_id": 0, "class_name": "bridge" }
      ]
    }
  ]
}
```

Splits are stored inline — see [03-splits.md](03-splits.md) for details.

### `experiment.json`

```json
{
  "id": "exp-001",
  "name": "ResNet50 baseline",
  "created_at": "2026-02-19T11:00:00Z",
  "split_index": 0,
  "model": {
    "backbone": "resnet50",
    "head": "classification",
    "pretrained": true,
    "freeze_backbone": false
  },
  "hyperparameters": {
    "optimizer": "adam",
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "scheduler": "cosine",
    "batch_size": 32,
    "max_epochs": 50,
    "early_stopping_patience": 10
  },
  "augmentations": {
    "train": [
      { "name": "RandomHorizontalFlip", "params": { "p": 0.5 } },
      { "name": "RandomRotation", "params": { "degrees": 15 } },
      { "name": "ColorJitter", "params": { "brightness": 0.2, "contrast": 0.2 } },
      { "name": "Resize", "params": { "size": [224, 224] } },
      { "name": "Normalize", "params": { "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225] } }
    ],
    "val": [
      { "name": "Resize", "params": { "size": [224, 224] } },
      { "name": "Normalize", "params": { "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225] } }
    ]
  },
  "hardware": {
    "selected_devices": ["gpu:0"],
    "precision": "32"
  }
}
```

### `run.json`

```json
{
  "id": "run-001",
  "experiment_id": "exp-001",
  "status": "completed",
  "started_at": "2026-02-19T11:05:00Z",
  "completed_at": "2026-02-19T11:45:00Z",
  "best_epoch": 38,
  "best_metric": { "val_accuracy": 0.956 },
  "final_metrics": {
    "train_loss": 0.042,
    "val_loss": 0.118,
    "val_accuracy": 0.956,
    "val_f1": 0.951
  }
}
```

Valid `status` values: `"pending"`, `"running"`, `"completed"`, `"failed"`, `"cancelled"`.

### `metrics.json`

```json
{
  "epochs": [
    {
      "epoch": 1,
      "train_loss": 2.31,
      "val_loss": 1.89,
      "val_accuracy": 0.32,
      "val_f1": 0.28,
      "lr": 0.001,
      "duration_s": 45.2
    },
    {
      "epoch": 2,
      "train_loss": 1.45,
      "val_loss": 1.12,
      "val_accuracy": 0.58,
      "val_f1": 0.54,
      "lr": 0.001,
      "duration_s": 44.8
    }
  ]
}
```

---

## 3. ID Generation

All IDs are generated as `<entity>-<short-uuid>`, e.g., `project-a1b2c3d4`. Use Python's `uuid.uuid4().hex[:8]` for the short UUID portion.

---

## 4. Concurrency & File Locking

Since only one user is expected at a time (local tool), basic concurrency handling is sufficient:

- Use `filelock` (or `fcntl` on Linux / `msvcrt` on Windows) when writing JSON files.
- Reads do not require locks (JSON files are written atomically via write-to-temp + rename).
- Training runs are managed via FastAPI `BackgroundTasks` — only one training run should be active at a time per project.

---

## 5. Image Storage

- On import, images are **copied** into `dataset/images/` (not symlinked) for portability.
- Original filenames are preserved. If collisions occur, a numeric suffix is appended (`img.png` → `img_1.png`).
- Images are not modified during import — resizing happens at training time via the augmentation pipeline.

---

## 6. Related Documents

- Dataset import flow → [01-dataset-management.md](01-dataset-management.md)
- Supported annotation formats → [02-dataset-formats.md](02-dataset-formats.md)
- Split logic → [03-splits.md](03-splits.md)
- Experiment tracking → [../06-experiment-tracking/00-run-management.md](../06-experiment-tracking/00-run-management.md)
