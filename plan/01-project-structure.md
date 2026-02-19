# AI Studio — Project Structure (Code Layout)

This document defines the folder layout for the **source code** of the AI Studio application. This is separate from the **workspace data** layout (see [02-data-layer/00-storage-layout.md](02-data-layer/00-storage-layout.md)).

---

## 1. Root Layout

```
ai-studio/
├── plan/                          # Planning documents (this folder)
├── app/                           # Application source code
│   ├── __init__.py
│   ├── main.py                    # FastAPI app factory, uvicorn entry point
│   ├── config.py                  # App-wide configuration (paths, defaults)
│   │
│   ├── api/                       # FastAPI routers (REST endpoints)
│   │   ├── __init__.py
│   │   ├── projects.py            # /api/projects — CRUD, list, select
│   │   ├── datasets.py            # /api/datasets — import, list, detail
│   │   ├── splits.py              # /api/splits — create, list, manage
│   │   ├── training.py            # /api/training — experiments, runs
│   │   ├── evaluation.py          # /api/evaluation — configure, run, results
│   │   └── export.py              # /api/export — trigger export, download
│   │
│   ├── core/                      # Business logic (framework-agnostic)
│   │   ├── __init__.py
│   │   ├── project_service.py     # Project CRUD, task type management
│   │   ├── dataset_service.py     # Import, registry, image listing
│   │   ├── split_service.py       # Stratified splitting
│   │   ├── training_service.py    # Experiment config, run orchestration
│   │   ├── evaluation_service.py  # Inference pipeline, metric computation
│   │   └── export_service.py      # Model export orchestration
│   │
│   ├── models/                    # Model definitions
│   │   ├── __init__.py
│   │   ├── catalog.py             # Architecture registry (name → factory)
│   │   ├── backbones.py           # Feature extractor wrappers around torchvision
│   │   ├── heads/                 # Task-specific heads
│   │   │   ├── __init__.py
│   │   │   ├── classification.py
│   │   │   ├── anomaly.py
│   │   │   ├── detection.py
│   │   │   ├── oriented_detection.py
│   │   │   ├── segmentation.py
│   │   │   ├── instance_segmentation.py
│   │   │   └── regression.py
│   │   └── pretrained.py          # Weight download / cache manager
│   │
│   ├── datasets/                  # Dataset & data loading
│   │   ├── __init__.py
│   │   ├── base.py                # Base dataset class / LightningDataModule
│   │   ├── formats/               # Import parsers per annotation format
│   │   │   ├── __init__.py
│   │   │   ├── coco.py            # COCO JSON parser
│   │   │   ├── yolo.py            # YOLO txt parser
│   │   │   └── csv_labels.py      # CSV label parser
│   │   ├── augmentations.py       # Augmentation pipeline builder
│   │   └── splits.py              # Split logic (stratified)
│   │
│   ├── training/                  # Training pipeline
│   │   ├── __init__.py
│   │   ├── lightning_module.py    # Generic LightningModule wrapping model + loss
│   │   ├── callbacks.py           # Custom callbacks (JSON logger, etc.)
│   │   ├── trainer_factory.py     # Build Lightning Trainer from config
│   │   └── losses.py              # Loss function registry per task
│   │
│   ├── evaluation/                # Evaluation pipeline
│   │   ├── __init__.py
│   │   ├── evaluator.py           # Load checkpoint, run inference on split
│   │   ├── metrics.py             # Task-specific metric computation
│   │   └── results.py             # Per-image result storage & aggregation
│   │
│   ├── export/                    # Model export
│   │   ├── __init__.py
│   │   ├── registry.py            # Format registry (extensible)
│   │   ├── onnx_export.py         # ONNX-specific export logic
│   │   └── validation.py          # Post-export validation (sample inference)
│   │
│   ├── schemas/                   # Pydantic models (request/response + JSON schemas)
│   │   ├── __init__.py
│   │   ├── project.py
│   │   ├── dataset.py
│   │   ├── split.py
│   │   ├── training.py
│   │   ├── evaluation.py
│   │   └── export.py
│   │
│   ├── storage/                   # Filesystem abstraction
│   │   ├── __init__.py
│   │   ├── paths.py               # Path resolution (project root, dataset dir, etc.)
│   │   └── json_store.py          # Read/write JSON metadata files
│   │
│   ├── templates/                 # Jinja2 HTML templates
│   │   ├── base.html              # Base layout (top nav bar, content area)
│   │   ├── pages/
│   │   │   ├── project.html       # Project selector / creator
│   │   │   ├── dataset.html       # Image grid + detail view
│   │   │   ├── split.html         # Split management
│   │   │   ├── training.html      # 3-column training page
│   │   │   ├── evaluation.html    # Evaluation config + results
│   │   │   └── export.html        # Export page
│   │   └── components/            # Reusable Jinja2 macros
│   │       ├── image_grid.html
│   │       ├── metric_chart.html
│   │       ├── form_controls.html
│   │       ├── progress_bar.html
│   │       └── toast.html
│   │
│   └── static/                    # Static assets
│       ├── css/
│       │   └── main.css
│       ├── js/
│       │   ├── htmx.min.js
│       │   ├── alpine.min.js
│       │   ├── chart.min.js
│       │   └── app.js             # Shared JS helpers
│       └── img/
│           └── logo.svg
│
├── workspace/                     # Default workspace root (user data)
│   └── projects/                  # Created at first run
│
├── tests/                         # Test suite
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_api/
│   ├── test_core/
│   ├── test_models/
│   ├── test_datasets/
│   ├── test_training/
│   ├── test_evaluation/
│   └── test_export/
│
├── pyproject.toml                 # Project metadata + dependencies
├── requirements.txt               # Pinned dependencies (generated)
├── README.md
└── .gitignore
```

---

## 2. Module Responsibilities

### `app/main.py` — Entry Point

- Creates the FastAPI application instance.
- Mounts static file serving (`/static`).
- Registers all API routers under `/api/`.
- Registers Jinja2 template rendering for page routes.
- Includes a startup hook to ensure the workspace directory exists.
- Run with: `uvicorn app.main:app --reload`

### `app/config.py` — Configuration

- `WORKSPACE_ROOT`: path to the workspace folder (default: `./workspace`).
- `DEFAULT_DEVICE`: `"cuda"` if available, else `"cpu"`.
- `MAX_UPLOAD_SIZE`: limit for image uploads.
- Configurable via environment variables or a `config.json` at the workspace root.

### `app/api/` — API Routers

Each file defines a FastAPI `APIRouter` with endpoints for one domain. Routers delegate to `core/` service functions and return Pydantic `schemas/` models.

Page routes (returning HTML) live alongside API routes but are separated by prefix:
- `/api/*` — JSON API endpoints (used by HTMX calls).
- `/*` — HTML page routes.

### `app/core/` — Business Logic

Pure Python service functions. No FastAPI or HTTP dependencies. Each service:
- Receives validated data (Pydantic models or primitives).
- Reads/writes data via `storage/`.
- Calls into `models/`, `datasets/`, `training/`, `evaluation/`, or `export/` as needed.
- Returns result data (Pydantic models or primitives).

### `app/models/` — Model Definitions

- `catalog.py`: a dictionary mapping `(task, architecture_name)` → factory function that returns a `nn.Module`.
- `backbones.py`: wrappers around `torchvision.models` that extract feature maps at specified layers.
- `heads/`: one file per task, each defining the head module(s) compatible with that task.
- `pretrained.py`: manages downloading and caching pretrained weights from torchvision.

### `app/datasets/` — Data Loading

- `base.py`: a `LightningDataModule` subclass that unifies all tasks — loads any task's dataset given a project config.
- `formats/`: parsers that read external annotation formats and convert them to a common internal representation.
- `augmentations.py`: builds a `torchvision.transforms.v2` pipeline from a JSON config.
- `splits.py`: implements the stratified splitting algorithm.

### `app/training/` — Training

- `lightning_module.py`: a generic `LightningModule` that wraps any model from the catalog, configures optimizer/scheduler, and logs metrics.
- `trainer_factory.py`: builds a `pl.Trainer` from an experiment config JSON (resolves device selection into accelerator/devices/strategy, sets precision, callbacks, max_epochs, etc.).
- `callbacks.py`: custom Lightning callbacks — `JSONMetricLogger` writes per-epoch metrics to the run folder.
- `losses.py`: registry mapping `(task, loss_name)` → loss function.

### `app/evaluation/` — Evaluation

- `evaluator.py`: loads a checkpoint, builds the model, runs inference on a DataLoader for a given split.
- `metrics.py`: computes task-specific metrics (accuracy, F1, mAP, IoU, Dice, MAE, R², etc.).
- `results.py`: stores per-image predictions + aggregate summaries as JSON.

### `app/export/` — Export

- `registry.py`: maps format names (`"onnx"`, `"torchscript"`, ...) to export functions. Extensible.
- `onnx_export.py`: calls `torch.onnx.export` with appropriate settings (opset, dynamic axes).
- `validation.py`: runs a sample input through the exported model and compares output with PyTorch output.

### `app/schemas/` — Pydantic Models

Request/response schemas for the API and JSON file schemas for persistence. Each file mirrors a domain module.

### `app/storage/` — Filesystem Abstraction

- `paths.py`: resolves absolute paths for any entity (project dir, dataset dir, run dir, etc.) given the workspace root + IDs.
- `json_store.py`: generic read/write/update helpers for JSON metadata files with file locking.

### `app/templates/` — Jinja2 Templates

- `base.html`: common layout — horizontal navigation bar (Project / Dataset / Split / Training / Evaluation / Export), content area, footer.
- `pages/`: one template per page.
- `components/`: reusable macros that pages `{% include %}` or `{% call %}`.

### `app/static/` — Static Assets

- Third-party JS: HTMX, Alpine.js, Chart.js (vendored, no CDN dependency).
- `app.js`: Shared utilities (toast notifications, SSE event handling for training progress).
- `main.css`: Application styles.

---

## 3. Dependency Map

```
api/ ──────► core/ ──────► storage/
  │            │
  │            ├──► models/
  │            ├──► datasets/
  │            ├──► training/   ──► models/ + datasets/
  │            ├──► evaluation/ ──► models/ + datasets/
  │            └──► export/     ──► models/
  │
  └──► schemas/
```

Rules:
- `api/` depends on `core/` and `schemas/` only — never on `models/`, `training/`, etc. directly.
- `core/` orchestrates but does not contain PyTorch code — it delegates to `models/`, `training/`, `evaluation/`, `export/`.
- `storage/` is the only module that touches the filesystem.
- `models/`, `datasets/`, `training/`, `evaluation/`, `export/` never import from `api/`.

---

## 4. Entry Point & Startup

```python
# app/main.py (simplified)
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api import projects, datasets, splits, training, evaluation, export
from app.config import settings

app = FastAPI(title="AI Studio")

# Static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Templates
templates = Jinja2Templates(directory="app/templates")

# API routers
app.include_router(projects.router, prefix="/api/projects", tags=["projects"])
app.include_router(datasets.router, prefix="/api/datasets", tags=["datasets"])
app.include_router(splits.router,   prefix="/api/splits",   tags=["splits"])
app.include_router(training.router, prefix="/api/training",  tags=["training"])
app.include_router(evaluation.router, prefix="/api/evaluation", tags=["evaluation"])
app.include_router(export.router,   prefix="/api/export",   tags=["export"])

# Page routes (HTML)
@app.get("/")
async def project_page(request: Request):
    return templates.TemplateResponse("pages/project.html", {"request": request})

# ... additional page routes
```

Run: `uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload`

---

## 5. Dependencies (`pyproject.toml`)

```toml
[project]
name = "ai-studio"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "fastapi>=0.110",
    "uvicorn[standard]>=0.29",
    "jinja2>=3.1",
    "python-multipart>=0.0.9",   # File uploads
    "torch>=2.2",
    "torchvision>=0.17",
    "pytorch-lightning>=2.2",
    "onnx>=1.15",
    "onnxruntime>=1.17",
    "pillow>=10.0",
    "numpy>=1.26",
    "scikit-learn>=1.4",          # Stratified splits, metrics
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "httpx>=0.27",               # Async test client for FastAPI
    "ruff>=0.3",                 # Linter + formatter
]
```

---

## 6. Related Documents

- Workspace data layout → [02-data-layer/00-storage-layout.md](02-data-layer/00-storage-layout.md)
- API endpoints detail → [09-api/01-endpoints.md](09-api/01-endpoints.md)
- GUI architecture → [10-gui/00-gui-overview.md](10-gui/00-gui-overview.md)
