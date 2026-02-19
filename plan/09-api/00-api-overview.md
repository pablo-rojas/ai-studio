# API — Overview

This document describes the FastAPI application architecture, router layout, and shared patterns.

---

## 1. Application Entry Point

```python
# app/main.py

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api import projects, datasets, splits, training, evaluation, export, system

app = FastAPI(title="AI Studio", version="0.1.0")

# Static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Templates
templates = Jinja2Templates(directory="app/templates")

# Routers
app.include_router(projects.router, prefix="/api/projects", tags=["projects"])
app.include_router(datasets.router, prefix="/api/datasets", tags=["datasets"])
app.include_router(splits.router, prefix="/api/splits", tags=["splits"])
app.include_router(training.router, prefix="/api/training", tags=["training"])
app.include_router(evaluation.router, prefix="/api/evaluation", tags=["evaluation"])
app.include_router(export.router, prefix="/api/export", tags=["export"])
app.include_router(system.router, prefix="/api/system", tags=["system"])

# HTML page routes
from app.api import pages
app.include_router(pages.router)
```

---

## 2. Router Layout

```
app/api/
├── pages.py            # HTML page routes (Jinja2 templates)
├── projects.py         # Project CRUD
├── datasets.py         # Dataset import / browse / manage
├── splits.py           # Split CRUD
├── training.py         # Experiment management, start/stop training
├── evaluation.py       # Evaluation CRUD, run evaluation
├── export.py           # Export CRUD, trigger export, download
└── system.py           # GPU info, workspace stats, health
```

### Page Routes (`pages.py`)

These routes serve the Jinja2 HTML templates:

| Route | Template | Page |
|-------|----------|------|
| `GET /` | Redirect to `/projects` | — |
| `GET /projects` | `pages/project.html` | Project page |
| `GET /projects/{id}/dataset` | `pages/dataset.html` | Dataset page |
| `GET /projects/{id}/split` | `pages/split.html` | Split page |
| `GET /projects/{id}/training` | `pages/training.html` | Training page |
| `GET /projects/{id}/evaluation` | `pages/evaluation.html` | Evaluation page |
| `GET /projects/{id}/export` | `pages/export.html` | Export page |

---

## 3. Request / Response Patterns

### JSON API Responses

All API endpoints return JSON. Standard envelope:

```json
{
  "status": "ok",
  "data": { "..." }
}
```

Error response:
```json
{
  "status": "error",
  "error": {
    "code": "NOT_FOUND",
    "message": "Project proj-abc123 not found"
  }
}
```

### HTMX Fragment Responses

Some endpoints return HTML fragments for HTMX partial updates:

| Condition | Response |
|-----------|----------|
| `HX-Request` header present | Return HTML fragment |
| No `HX-Request` header | Return JSON |

```python
from fastapi import Request

@router.get("/api/projects")
async def list_projects(request: Request):
    projects = load_projects()
    if request.headers.get("HX-Request"):
        return templates.TemplateResponse("fragments/project_list.html", {
            "request": request, "projects": projects
        })
    return {"status": "ok", "data": projects}
```

---

## 4. Background Tasks

Long-running operations use FastAPI's `BackgroundTasks`:

| Operation | Endpoint | Background Task |
|-----------|----------|-----------------|
| Training | `POST /api/training/run` | `run_training()` |
| Evaluation | `POST /api/evaluation/run` | `run_evaluation()` |
| Export | `POST /api/export/run` | `run_export()` |
| Dataset import | `POST /api/datasets/import` | `import_dataset()` |

### Concurrency

- Only **one experiment training** at a time (GPU exclusivity).
- Evaluations can run concurrently with each other (if GPU memory allows).
- Exports are quick (seconds to minutes) and run sequentially.

A simple lock mechanism:

```python
# app/core/locks.py
import asyncio

training_lock = asyncio.Lock()

async def acquire_training():
    if training_lock.locked():
        raise HTTPException(409, "An experiment is already training")
    await training_lock.acquire()
```

---

## 5. File Upload Handling

Dataset import supports two modes:

### Local Path Import
User provides a path to images on the server filesystem. Backend copies/moves files.

```python
@router.post("/api/datasets/import/local")
async def import_from_local(config: LocalImportConfig):
    # config.source_path: Path on server
    # config.format: "coco" | "yolo" | "csv" | "folder"
    ...
```

### File Upload
User uploads a ZIP archive via the browser.

```python
from fastapi import UploadFile

@router.post("/api/datasets/import/upload")
async def import_from_upload(file: UploadFile, format: str):
    # Extract ZIP, then process as local import
    ...
```

---

## 6. SSE (Server-Sent Events)

For live training updates:

```python
from starlette.responses import StreamingResponse

@router.get("/api/training/{project_id}/experiments/{experiment_id}/stream")
async def training_stream(project_id: str, experiment_id: str):
    async def event_generator():
        queue = subscribe_to_experiment(experiment_id)
        try:
            while True:
                event = await queue.get()
                yield f"event: {event['type']}\ndata: {json.dumps(event['data'])}\n\n"
                if event["type"] == "complete":
                    break
        except asyncio.CancelledError:
            pass
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

Client-side (JavaScript):
```javascript
const source = new EventSource(`/api/training/${runId}/stream`);
source.addEventListener('epoch_end', (e) => {
    const data = JSON.parse(e.data);
    updateChart(data);
});
source.addEventListener('complete', () => source.close());
```

---

## 7. Image Serving

Images are served from the workspace directory:

```python
@router.get("/api/datasets/{project_id}/image/{filename}")
async def serve_image(project_id: str, filename: str):
    image_path = workspace / "projects" / project_id / "dataset" / "images" / filename
    return FileResponse(image_path)

@router.get("/api/datasets/{project_id}/thumbnail/{filename}")
async def serve_thumbnail(project_id: str, filename: str):
    thumb_path = workspace / "projects" / project_id / "dataset" / ".thumbs" / filename
    if not thumb_path.exists():
        generate_thumbnail(image_path, thumb_path)
    return FileResponse(thumb_path)
```

---

## 8. Related Documents

- Full endpoint reference → [01-endpoints.md](01-endpoints.md)
- Error handling → [02-error-handling.md](02-error-handling.md)
- GUI pages (consumers of these APIs) → [../10-gui/00-gui-overview.md](../10-gui/00-gui-overview.md)
