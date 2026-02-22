# AGENTS.md — AI Studio

---

## Project Overview

AI Studio is a self-contained, browser-based application for training, evaluating, and exporting deep-learning models for image-based tasks. It is a **monolithic Python application** — a single FastAPI process serves both the REST API and the Jinja2-rendered web interface.

The full lifecycle: create project → import dataset → split → train → evaluate → export (ONNX).

**Detailed plans live in `plan/`** — always read the relevant plan document before implementing a feature. The plan is the source of truth for architecture, schemas, folder layout, and API contracts.

---

## Tech Stack

| Layer | Technology | Version constraint |
|-------|-----------|-------------------|
| Language | **Python 3.10+** | `>=3.10` |
| Web framework | **FastAPI** | `>=0.110` |
| Server | **uvicorn** | `>=0.29` |
| Templating | **Jinja2** | `>=3.1` |
| Deep learning | **PyTorch** | `>=2.2` |
| Training | **PyTorch Lightning** | `>=2.2` |
| Model zoo | **torchvision** | `>=0.17` |
| Interactivity | **HTMX + Alpine.js** | Vendored in `app/static/js/` |
| Charts | **Chart.js** | Vendored in `app/static/js/` |
| Persistence | **Filesystem + JSON** | No database |
| Linter/formatter | **Ruff** | `>=0.3` |
| Tests | **pytest** + **httpx** | `>=8.0`, `>=0.27` |
| Export | **ONNX** + **ONNX Runtime** | `>=1.15`, `>=1.17` |

**No SPA framework.** Do not introduce React, Vue, Svelte, or any frontend build toolchain. All interactivity is HTMX + Alpine.js. JS files are vendored — no CDN, no npm.

---

## Development Environment Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 2. Install in editable mode with dev dependencies
pip install -e ".[dev]"

# 3. Start the development server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 4. Open in browser
# http://localhost:8000
```

The workspace data folder (`./workspace`) is created automatically on first launch. Do not commit the `workspace/` folder.

### Virtual Environment — Important for Agents

**The `.venv` is NOT activated automatically in agent/terminal sessions.** All Python tools (`python`, `pytest`, `ruff`, `uvicorn`, etc.) must be invoked using their full path inside `.venv` to ensure the correct interpreter and installed packages are used.

Use these paths on Linux/macOS:

| Tool | Command to use |
|------|---------------|
| Python interpreter | `.venv/bin/python` |
| pip | `.venv/bin/pip` |
| pytest | `.venv/bin/pytest` |
| ruff | `.venv/bin/ruff` |
| uvicorn | `.venv/bin/uvicorn` |

**Never assume `python`, `pytest`, `ruff`, or `uvicorn` on `PATH` are the venv versions.** Always prefix with `.venv/bin/` (or activate the venv first with `source .venv/bin/activate` in the same shell session).


---

## Project Structure

```
ai-studio/
├── plan/                     # Architecture & design docs (READ BEFORE CODING)
├── app/                      # Application source code
│   ├── main.py               # FastAPI app factory + entry point
│   ├── config.py             # App-wide config (paths, defaults, env vars)
│   ├── api/                  # FastAPI routers (REST + page routes)
│   ├── core/                 # Business logic (no HTTP, no PyTorch)
│   ├── models/               # Model definitions (catalog, backbones, heads)
│   ├── datasets/             # Data loading, format parsers, augmentations, splits
│   ├── training/             # LightningModule, Trainer factory, callbacks, losses
│   ├── evaluation/           # Evaluator, metrics, results
│   ├── export/               # ONNX export, validation, format registry
│   ├── schemas/              # Pydantic models (API + JSON file schemas)
│   ├── storage/              # Filesystem abstraction (paths, JSON read/write)
│   ├── templates/            # Jinja2 HTML templates
│   └── static/               # CSS, JS (vendored), images
├── tests/                    # Test suite (mirrors app/ structure)
├── workspace/                # Runtime user data (git-ignored)
├── pyproject.toml            # Project metadata + dependencies
└── AGENTS.md                 # This file
```

Full structure details: `plan/01-project-structure.md`.
Workspace data layout: `plan/02-data-layer/00-storage-layout.md`.

---

## Phased Roadmap — The Master Plan

This project is built in **25 phases**, always in **backend → API → GUI** order. The full roadmap with acceptance criteria is in `plan/11-roadmap.md`.

### Current Phase

> **Update this line when starting a new phase:**
>
> `CURRENT_PHASE = 18`

Only implement features belonging to the current phase. Do not jump ahead. Each phase must be fully complete (all acceptance criteria met, tests passing) before moving on.

### Phase Summary

| Phase | Focus | Key files |
|-------|-------|-----------|
| 1 | Core Platform & Storage | `app/storage/`, `app/core/project_service.py`, `app/schemas/project.py` |
| 2 | Dataset Management | `app/core/dataset_service.py`, `app/datasets/formats/`, `app/schemas/dataset.py` |
| 3 | Splits | `app/core/split_service.py`, `app/datasets/splits.py`, `app/schemas/split.py` |
| 4 | API: Projects, Datasets, Splits | `app/api/projects.py`, `app/api/datasets.py`, `app/api/splits.py` |
| 5 | GUI: Project Page | `app/templates/pages/projects.html`, `app/api/pages.py` |
| 6 | GUI: Dataset Page | `app/templates/pages/dataset.html`, image grid, pagination |
| 7 | GUI: Split Page | `app/templates/pages/split.html`, split form, preview |
| 8 | Classification Task | `app/models/catalog.py`, `app/models/heads/classification.py`, task registry |
| 9 | Training Pipeline | `app/training/lightning_module.py`, `app/training/trainer_factory.py` |
| 10 | Experiment Tracking | `app/core/training_service.py`, run management, metrics JSON |
| 11 | API: Training & Experiments | `app/api/training.py`, SSE streaming |
| 12 | GUI: Training Page | `app/templates/pages/training.html`, live loss curves |
| 13 | Evaluation Pipeline | `app/evaluation/evaluator.py`, `app/core/evaluation_service.py`, `app/schemas/evaluation.py` |
| 14 | API: Evaluation | `app/api/evaluation.py` (experiment-scoped, no separate eval ID) |
| 15 | GUI: Evaluation Page | `app/templates/pages/evaluation.html`, 2-col layout, 3 collapsible sections |
| 16 | ONNX Export | `app/export/onnx_export.py`, `app/export/validation.py` |
| 17 | API: Export | `app/api/export.py` |
| 18 | GUI: Export Page | `app/templates/pages/export.html` |
| 19–24 | Additional tasks | Object detection, segmentation, instance seg, anomaly detection, regression, OBB |
| 25 | Polish & Extras | Multi-GPU, mixed precision, extra exports, dark mode |

### Phase Dependencies

```
Phase 1 → 2 → 3 → 4 → {5, 6, 7}
Phase 3 → 8 → 9 → 10 → 11 → 12
Phase 10 → 13 → 14 → 15
Phase 10 → 16 → 17 → 18
Phases 19–24 depend on Groups A–D being complete.
```

---

## Architecture Rules — MUST Follow

These rules are non-negotiable. Violating them creates coupling that is hard to undo.

### Dependency Hierarchy

```
api/ ──► core/ ──► storage/
 │         │
 │         ├──► models/
 │         ├──► datasets/
 │         ├──► training/
 │         ├──► evaluation/
 │         └──► export/
 │
 └──► schemas/
```

1. **`app/api/`** depends only on `app/core/` and `app/schemas/`. Never import from `models/`, `training/`, etc.
2. **`app/core/`** orchestrates business logic. It calls into `models/`, `datasets/`, `training/`, `evaluation/`, `export/`, and `storage/`. It does NOT contain PyTorch code directly.
3. **`app/storage/`** is the only module that touches the filesystem. All file reads/writes go through `storage/paths.py` and `storage/json_store.py`.
4. **`models/`, `datasets/`, `training/`, `evaluation/`, `export/`** never import from `api/`.
5. **`app/schemas/`** has no dependencies on other app modules — it is pure Pydantic.

### API Patterns

- All JSON API endpoints live under `/api/*` and return the standard envelope:
  ```json
  { "status": "ok", "data": { ... } }
  ```
- Error responses use:
  ```json
  { "status": "error", "error": { "code": "NOT_FOUND", "message": "..." } }
  ```
- If `HX-Request` header is present, return an HTML fragment instead of JSON (for HTMX).
- Page routes (serving full HTML) live at `/*` (not under `/api/`).
- Use Pydantic models from `app/schemas/` for request validation and response serialization.
- Custom exceptions inherit from `AIStudioError` defined in `app/core/exceptions.py`. See `plan/09-api/02-error-handling.md`.

### Persistence Patterns

- **No database.** All state lives as JSON files on the filesystem under `workspace/`.
- Use `app/storage/json_store.py` for reading/writing JSON files (with file locking).
- Use `app/storage/paths.py` to resolve paths — never hardcode paths.
- JSON schemas are defined in `plan/02-data-layer/00-storage-layout.md`. Follow them exactly.
- IDs are generated as short random strings (e.g., `proj-a1b2c3d4`). Use a consistent format: `{prefix}-{8 hex chars}`.

### GUI Patterns

- **Jinja2 templates** for server-rendered pages.
- **HTMX** for partial page updates (fragments in `app/templates/fragments/`).
- **Alpine.js** for client-side reactivity (modals, toggles, form binding).
- **Chart.js** for charts and visualizations.
- **Tailwind CSS** for styling (via Play CDN `<script>` tag — no build step, no local CSS file).
- All JS is vendored in `app/static/js/` — no npm, no bundler, no build step.
- Templates extend `base.html` which provides the top bar + sidebar layout.
- Reusable UI pieces go in `app/templates/components/` as Jinja2 macros.

---

## Code Style & Conventions

### Python

- **Ruff** for linting and formatting. Run before every commit:
  ```bash
  .venv/bin/ruff check app/ tests/ --fix
  .venv/bin/ruff format app/ tests/
  ```
- Type hints on all function signatures. Use `from __future__ import annotations` in every file.
- Pydantic v2 models for all data validation and serialization.
- Use `pathlib.Path` for all path operations — never `os.path`.
- Prefer `async def` for API route handlers; use regular `def` for CPU-bound core/service functions.
- Docstrings on all public functions and classes (Google-style).
- Imports: standard library → third-party → local app, separated by blank lines. Ruff handles ordering.

### Naming Conventions

| Entity | Convention | Example |
|--------|-----------|---------|
| Python files | `snake_case` | `dataset_service.py` |
| Python classes | `PascalCase` | `ProjectSchema`, `AIStudioModule` |
| Python functions | `snake_case` | `create_project()`, `import_dataset()` |
| API endpoints | `snake_case` paths | `/api/projects/{project_id}` |
| JSON keys | `snake_case` | `"created_at"`, `"split_names"` |
| HTML templates | `snake_case` | `project_list.html`, `image_grid.html` |
| CSS classes | Follow Tailwind or `kebab-case` | `image-grid`, `nav-active` |
| JS functions | `camelCase` | `startTraining()`, `updateChart()` |
| IDs in JSON | `{prefix}-{8 hex}` | `proj-a1b2c3d4`, `exp-12345678` |

### Error Handling

- Define domain exceptions in `app/core/exceptions.py`.
- All custom exceptions extend `AIStudioError`.
- The global exception handler in `app/main.py` converts them to standard error JSON.
- Never catch generic `Exception` in API handlers — let the global handler deal with it.
- Log errors with `logging` (not `print`).

---

## Testing

### Running Tests

```bash
# Run the full test suite
pytest tests/ -v

# Run tests for a specific module
pytest tests/test_core/ -v
pytest tests/test_api/ -v

# Run a single test file
pytest tests/test_core/test_project_service.py -v

# Run with coverage
pytest tests/ --cov=app --cov-report=term-missing
```

### Test Structure

```
tests/
├── conftest.py              # Shared fixtures (tmp workspace, test client, sample data)
├── test_api/                # API endpoint tests (use httpx AsyncClient)
├── test_core/               # Business logic unit tests
├── test_models/             # Model creation and forward pass tests
├── test_datasets/           # Format parser and dataloader tests
├── test_training/           # Training pipeline tests
├── test_evaluation/         # Evaluation tests
└── test_export/             # Export tests
```

### Testing Conventions

- Use `pytest` with `httpx.AsyncClient` for API tests (FastAPI's recommended approach).
- Use `tmp_path` fixture for workspace isolation — never use the real `workspace/` folder in tests.
- Each test function should be independent and create its own data.
- Name test files `test_<module>.py` and test functions `test_<what_it_tests>`.
- Write tests for **every phase** as you implement it. Tests are part of the acceptance criteria.
- Mock PyTorch model training in API/core tests (training tests can be slow).
- For GUI-related tests: test the API responses, not the HTML rendering.

### conftest.py Fixtures to Provide

```python
@pytest.fixture
def workspace(tmp_path):
    """Provide a clean temporary workspace root."""
    ...

@pytest.fixture
def test_client(workspace):
    """Provide an httpx AsyncClient pointing at the test app."""
    ...

@pytest.fixture
def sample_project(workspace):
    """Create and return a sample project."""
    ...

@pytest.fixture
def sample_dataset(sample_project):
    """Import a small test dataset into the sample project."""

    
    ...
```

---

## Working on a Phase — Step-by-Step

When starting a phase, follow this sequence:

1. **Run the full test suite** before writing any code. This establishes a clean baseline so that any test failures introduced later can be attributed to the new phase's changes.
   ```bash
   .venv/bin/pytest tests/ -v
   ```
   If existing tests fail before you start, stop and notify the user. **You must not skip this part.**
2. **Read the plan.** Open `plan/11-roadmap.md` and the relevant section-specific docs in `plan/`. Understand the deliverables and acceptance criteria.
3. **Update `CURRENT_PHASE`** in this file.
4. **Implement backend logic first** (`app/core/`, `app/storage/`, domain modules).
5. **Define Pydantic schemas** (`app/schemas/`) for any new data structures.
6. **Write tests** for the backend logic.
7. **Implement API endpoints** (`app/api/`) if the phase includes them.
8. **Write API tests** using the test client.
9. **Implement GUI** (`app/templates/`) if the phase includes it.
10. **Manually test end-to-end** via `uvicorn app.main:app --reload`.
11. **Verify all acceptance criteria** from `plan/11-roadmap.md`.
12. **Run the full test suite** and ensure everything passes.
13. **Lint and format**: `.venv/bin/ruff check app/ tests/ --fix && .venv/bin/ruff format app/ tests/`

---

## Phase-Specific Guidance

### Phase 1 — Core Platform & Storage

**Read:** `plan/02-data-layer/00-storage-layout.md`, `plan/01-project-structure.md`

Key deliverables:
- `app/storage/paths.py` — path resolution for all entity types.
- `app/storage/json_store.py` — generic JSON read/write with file locking.
- `app/core/project_service.py` — create, list, rename, delete projects.
- `app/schemas/project.py` — `ProjectCreate`, `ProjectResponse` Pydantic models.
- `app/config.py` — `WORKSPACE_ROOT`, `DEFAULT_DEVICE`, etc.
- Startup hook in `app/main.py` to create workspace dir if it doesn't exist.

ID format: `proj-{8 hex chars}` using `uuid.uuid4().hex[:8]`.

Test: create a project, list it, rename it, delete it — all via service functions.

### Phase 2 — Dataset Management

**Read:** `plan/02-data-layer/01-dataset-management.md`, `plan/02-data-layer/02-dataset-formats.md`

Key deliverables:
- `app/datasets/formats/` — parsers for folder structure, COCO JSON, CSV.
- `app/core/dataset_service.py` — import orchestration, thumbnail generation.
- Image metadata stored in `dataset.json` (follow the schema exactly).
- Thumbnail generation via Pillow (128×128 max, stored in `.thumbs/`).

Test: import a small folder-structured dataset, verify metadata JSON is correct.

### Phase 3 — Splits

**Read:** `plan/02-data-layer/03-splits.md`

Splits are stored **inline** in `dataset.json` — each image has a `split` list where index N corresponds to `split_names[N]`. Values are `"train"`, `"val"`, `"test"`, or `"none"`.

Use `sklearn.model_selection.train_test_split` with `stratify` for splits.

### Phases 4, 11, 14, 17 — API Phases

**Read:** `plan/09-api/00-api-overview.md`, `plan/09-api/01-endpoints.md`, `plan/09-api/02-error-handling.md`

- Follow the endpoint reference exactly.
- Implement the `HX-Request` header check for dual JSON/HTML responses.
- Use background tasks for long-running operations (import, training, evaluation, export).
- Only one training run at a time — use `asyncio.Lock`.
- SSE for live training updates (see `plan/09-api/00-api-overview.md` §6).

### Phases 5–7, 12, 15, 18 — GUI Phases

**Read:** `plan/10-gui/00-gui-overview.md`, `plan/10-gui/01-pages/`

- Extend `base.html` for every page.
- Use HTMX attributes (`hx-get`, `hx-post`, `hx-target`, `hx-swap`) for dynamic updates.
- Use Alpine.js `x-data`, `x-show`, `x-bind` for client-side state.
- Fragments go in `app/templates/fragments/`.
- Keep JavaScript minimal. If you need JS, put it in `app/static/js/app.js`.

### Phase 8 — Classification Task

**Read:** `plan/03-tasks/00-task-registry.md`, `plan/03-tasks/01-classification.md`, `plan/04-models/00-architecture-catalog.md`

- Register `ClassificationTask` in `TASK_REGISTRY`.
- Implement `app/models/heads/classification.py` — simple FC head on top of backbone features.
- 6 backbones: ResNet-18/34/50, EfficientNet-B0/B3, MobileNetV3-Small/Large.
- Default loss: `CrossEntropyLoss` with optional label smoothing.
- Metrics: accuracy, precision, recall, F1, confusion matrix.

### Phase 9 — Training Pipeline

**Read:** `plan/05-training/00-training-pipeline.md`, `plan/05-training/01-hyperparameters.md`

- `AIStudioModule(pl.LightningModule)` — generic, works for all tasks.
- `AIStudioDataModule(pl.LightningDataModule)` — loads dataset + split, applies augmentations.
- `trainer_factory.py` — builds `pl.Trainer` from `experiment.json`.
- Training runs in a **subprocess** (not a thread) to avoid GIL issues.
- JSON metrics logger callback writes to `metrics.json` every epoch.
- Checkpoints: `best.ckpt` (by primary metric) + `last.ckpt`.

### Phase 10 — Experiment Tracking

**Read:** `plan/06-experiment-tracking/00-run-management.md`

- Experiments → runs hierarchy (one experiment can have many runs).
- Each run gets a folder with `run.json`, `config.json`, `metrics.json`, `checkpoints/`.
- Run statuses: `pending`, `running`, `completed`, `failed`, `cancelled`.

### Phases 13–15 — Evaluation

**Read:** `plan/07-evaluation/00-evaluation-pipeline.md`, `plan/07-evaluation/01-per-image-results.md`, `plan/07-evaluation/02-aggregate-metrics.md`

**Key design:** Evaluation is **1:1 with an experiment** — no separate evaluation IDs, no `evaluations/` folder, no `evaluations_index.json`. Evaluation data lives inside the experiment folder at `experiments/<exp-id>/evaluation/`.

- `app/evaluation/evaluator.py` — loads checkpoint, builds combined dataloader from selected subsets, runs inference.
- `app/core/evaluation_service.py` — start, get, reset evaluation; list checkpoints.
- `app/schemas/evaluation.py` — `EvaluationConfig`, `EvaluationRecord`, per-image result models.
- Multiple split subsets (e.g., `["test", "val"]`) are pooled into a single combined evaluation. Per-image results are tagged with `"subset"` field.
- Aggregate metrics are computed over the combined pool — no per-subset breakdown.
- **Reset** immediately deletes the `evaluation/` subfolder (no confirmation dialog).
- **Checkpoint selector** only shows `.ckpt` files that actually exist in the experiment's `checkpoints/` directory.
- Storage path helpers: `experiment_evaluation_dir()`, `experiment_evaluation_metadata_file()`, `experiment_evaluation_aggregate_file()`, `experiment_evaluation_results_file()` in `app/storage/paths.py`.
- GUI: 2-column layout identical to Training page. Left panel = completed experiments only. Right panel = 3 collapsible Alpine.js sections (config + actions, metrics + visualizations, per-image results grid).

---

## Security Considerations

- This is a **local-only** application. Do not expose it to the public internet without authentication.
- File paths from user input must be validated — prevent path traversal attacks.
- Sanitize project/experiment names (restrict to alphanumeric, spaces, hyphens, underscores).
- Limit upload file sizes (`MAX_UPLOAD_SIZE` in config).
- Do not execute arbitrary user code — the training pipeline only uses predefined model architectures and loss functions.

---

## Things NOT To Do

- **Do NOT introduce a database** (SQLite, Postgres, etc.). All persistence is JSON on filesystem.
- **Do NOT add npm, webpack, vite, or any JS build toolchain.** JS is vendored.
- **Do NOT add React, Vue, Svelte, or Angular.** The GUI is Jinja2 + HTMX + Alpine.js.
- **Do NOT implement features from future phases.** Stick to the current phase.
- **Do NOT use `os.path`.** Use `pathlib.Path`.
- **Do NOT commit the `workspace/` data folder.**
- **Do NOT hardcode file paths.** Use `app/storage/paths.py`.
- **Do NOT catch generic `Exception` in API handlers.**
- **Do NOT put PyTorch code in `app/core/` or `app/api/`.** PyTorch code belongs in `models/`, `datasets/`, `training/`, `evaluation/`, `export/`.
- **Do NOT create circular imports.** Follow the dependency hierarchy strictly.
- **Do NOT use print() for logging.** Use `import logging; logger = logging.getLogger(__name__)`.

---

## Useful Commands

| Command | Purpose |
|---------|---------|
| `.venv/bin/uvicorn app.main:app --reload` | Start dev server with hot reload |
| `.venv/bin/pytest tests/ -v` | Run full test suite |
| `.venv/bin/pytest tests/ --cov=app` | Run tests with coverage |
| `.venv/bin/ruff check app/ tests/ --fix` | Lint and auto-fix |
| `.venv/bin/ruff format app/ tests/` | Format code |
| `.venv/bin/python -c "import torch; print(torch.cuda.is_available())"` | Check GPU availability |

---

## Key Plan Documents Reference

| Topic | Document |
|-------|----------|
| **Overview** | `plan/00-overview.md` |
| **Code layout** | `plan/01-project-structure.md` |
| **Data storage** | `plan/02-data-layer/00-storage-layout.md` |
| **Dataset schemas** | `plan/02-data-layer/01-dataset-management.md` |
| **Dataset formats** | `plan/02-data-layer/02-dataset-formats.md` |
| **Splits** | `plan/02-data-layer/03-splits.md` |
| **Task registry** | `plan/03-tasks/00-task-registry.md` |
| **Classification** | `plan/03-tasks/01-classification.md` |
| **Model catalog** | `plan/04-models/00-architecture-catalog.md` |
| **Backbones** | `plan/04-models/01-backbones.md` |
| **Training pipeline** | `plan/05-training/00-training-pipeline.md` |
| **Hyperparameters** | `plan/05-training/01-hyperparameters.md` |
| **Augmentations** | `plan/05-training/02-augmentation.md` |
| **Experiment tracking** | `plan/06-experiment-tracking/00-run-management.md` |
| **Evaluation** | `plan/07-evaluation/00-evaluation-pipeline.md` |
| **Export** | `plan/08-export/00-export-overview.md` |
| **API overview** | `plan/09-api/00-api-overview.md` |
| **API endpoints** | `plan/09-api/01-endpoints.md` |
| **Error handling** | `plan/09-api/02-error-handling.md` |
| **GUI overview** | `plan/10-gui/00-gui-overview.md` |
| **Roadmap** | `plan/11-roadmap.md` |

---

## Adding a New Task (Phases 19–24 Checklist)

When adding a new task type, follow this checklist from `plan/03-tasks/00-task-registry.md`:

1. Define annotation type(s) in `dataset.json` spec.
2. Implement format parsers in `app/datasets/formats/`.
3. Implement a Dataset class in `app/datasets/`.
4. Implement head module(s) in `app/models/heads/`.
5. Define loss function(s) in `app/training/losses.py`.
6. Define metrics in `app/evaluation/metrics.py`.
7. Define default augmentations.
8. Implement visualization overlays for Dataset + Evaluation pages.
9. Register in `TASK_REGISTRY` in `app/models/catalog.py`.
10. Update `ACTIVE_PHASES` to include the new phase.
11. Add end-to-end tests.
