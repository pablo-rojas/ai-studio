# AI Studio — Overview

## 1. Vision

AI Studio is a self-contained, browser-based application for training, evaluating, and exporting deep-learning models for image-based tasks. It targets engineers and researchers who need a streamlined, GUI-driven workflow without leaving their local machine.

The studio covers the full lifecycle:

1. **Create a project** and pick a task type.
2. **Import a dataset** (pre-annotated) and browse images + annotations.
3. **Split** the dataset into train / val / test sets.
4. **Configure & train** a model — select architecture, tune hyperparameters, define augmentations.
5. **Evaluate** the trained model on a test set with per-image detail.
6. **Export** the model to a deployable format (ONNX initially).

---

## 2. Supported Tasks (Phased)

| Phase | Task | Description |
|-------|------|-------------|
| 8 | **Classification** | Assign a single label to an entire image. |
| 19 | **Object Detection** | Locate objects with axis-aligned bounding boxes + class labels. |
| 20 | **Segmentation** | Assign a class label to every pixel (semantic segmentation). |
| 21 | **Instance Segmentation** | Like segmentation, but distinguish individual object instances. |
| 22 | **Anomaly Detection** | Detect whether an image is normal or anomalous, optionally localize the anomaly. |
| 23 | **Regression** | Predict one or more continuous numeric values from an image. |
| 24 | **Oriented Object Detection** | Like object detection, but with rotated bounding boxes. |

---

## 3. Tech Stack

| Layer | Technology | Role |
|-------|-----------|------|
| **Deep Learning** | PyTorch | Tensor ops, autograd, model definition |
| **Training Framework** | PyTorch Lightning | Training loop, callbacks, multi-GPU, logging |
| **Model Zoo** | torchvision | Pretrained backbones (ResNet, EfficientNet, MobileNet, etc.) |
| **Web Framework** | FastAPI | REST API, request validation (Pydantic), background tasks |
| **Server** | uvicorn | ASGI server to run the FastAPI application |
| **Templating** | Jinja2 | Server-rendered HTML pages |
| **Interactivity** | HTMX + Alpine.js | Dynamic UI without a full SPA framework |
| **Charts** | Chart.js | Loss curves, metric plots |
| **Persistence** | Filesystem + JSON | Images in folders, metadata/configs/experiments in JSON files |

---

## 4. Architecture Style

**Monolithic** — a single FastAPI application serves both the REST API and the Jinja2-rendered web interface. There is no separate frontend build step or process.

```
┌─────────────────────────────────────────────────┐
│                   uvicorn                        │
│  ┌─────────────────────────────────────────────┐ │
│  │              FastAPI App                     │ │
│  │  ┌──────────┐  ┌───────────┐  ┌──────────┐ │ │
│  │  │ Jinja2   │  │ API       │  │ Static   │ │ │
│  │  │ Templates│  │ Routers   │  │ Files    │ │ │
│  │  └──────────┘  └───────────┘  └──────────┘ │ │
│  │         │            │                      │ │
│  │  ┌──────────────────────────────────────┐   │ │
│  │  │         Core Services                │   │ │
│  │  │  datasets · training · evaluation    │   │ │
│  │  │  export · models · experiments       │   │ │
│  │  └──────────────────────────────────────┘   │ │
│  │         │            │                      │ │
│  │  ┌──────────────────────────────────────┐   │ │
│  │  │       Filesystem + JSON Store        │   │ │
│  │  └──────────────────────────────────────┘   │ │
│  └─────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

---

## 5. Persistence Strategy

No database. All state lives on the filesystem:

- **Images** are stored in a folder hierarchy under each project.
- **Annotations, dataset metadata, splits** are stored as JSON files.
- **Training configs (hyperparameters, augmentation pipelines)** are stored as JSON files per experiment.
- **Training experiments** produce metrics JSON, checkpoint `.ckpt` files, and logs within the experiment folder.
- **Evaluation results** are stored as JSON (aggregate + per-image) alongside the evaluation config.
- **Exported models** are saved as `.onnx` (or other format) files in the project's export folder.

See [02-data-layer/00-storage-layout.md](02-data-layer/00-storage-layout.md) for the full folder schema.

---

## 6. Hardware Support

- **Single GPU** — the default path (GPU 0 pre-selected when available).
- **Multi-GPU** — via PyTorch Lightning's `DDPStrategy`. Automatically activated when the user selects multiple GPUs in the device selector.
- **CPU fallback** — training works on CPU (slow, but functional). Auto-selected when no GPU is detected.

---

## 7. Key Design Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | Monolithic FastAPI + Jinja2 | Single deployment, no frontend build tooling, fast iteration. |
| 2 | Filesystem + JSON (no database) | Zero external dependencies, portable, simple to inspect and version-control. |
| 3 | Phased task rollout, classification first | Limits initial scope; architecture stabilizes before adding complexity. |
| 4 | Predefined model catalog (not plugin system) | Curated torchvision architectures; simpler, fewer failure modes. |
| 5 | Import-only annotation (initially) | Avoids building a full annotation tool in early phases. |
| 6 | ONNX-first export with extensible format registry | Covers the most common deployment target; TensorRT/OpenVINO/PyTorch added later. |
| 7 | HTMX + Alpine.js for interactivity | Minimal JS footprint, progressive enhancement, no SPA complexity. |
| 8 | Stratified splits | Balanced dataset splitting with class-distribution preservation from day one. |

---

## 8. GUI Pages

The web interface consists of 6 pages, reflecting the natural workflow:

| # | Page | Purpose |
|---|------|---------|
| 1 | **Project** | Select or create a project. Task type is chosen at project creation. |
| 2 | **Dataset** | Browse images + annotations in a grid. Click an image for detail view. |
| 3 | **Split** | Create new train/val/test splits or manage existing ones. |
| 4 | **Training** | 3-column layout: experiment list (left) · hparams + augmentations (center) · results (right). |
| 5 | **Evaluation** | Configure and run evaluations on a test split. View per-image + aggregate results. |
| 6 | **Export** | Select a trained model, choose export format (ONNX), configure, and download. |

See [10-gui/00-gui-overview.md](10-gui/00-gui-overview.md) for detailed GUI architecture.

---

## 9. Related Documents

| Area | Document |
|------|----------|
| Code structure | [01-project-structure.md](01-project-structure.md) |
| Storage & datasets | [02-data-layer/](02-data-layer/) |
| Task definitions | [03-tasks/](03-tasks/) |
| Model catalog | [04-models/](04-models/) |
| Training pipeline | [05-training/](05-training/) |
| Experiment tracking | [06-experiment-tracking/](06-experiment-tracking/) |
| Evaluation | [07-evaluation/](07-evaluation/) |
| Export | [08-export/](08-export/) |
| API layer | [09-api/](09-api/) |
| GUI | [10-gui/](10-gui/) |
| Roadmap | [11-roadmap.md](11-roadmap.md) |
