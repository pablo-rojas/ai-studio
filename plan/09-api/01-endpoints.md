# API — Endpoint Reference

Complete reference of all API endpoints.

---

## 1. Projects

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/projects` | List all projects |
| `POST` | `/api/projects` | Create a new project |
| `GET` | `/api/projects/{id}` | Get project details |
| `PATCH` | `/api/projects/{id}` | Update project (name, task) |
| `DELETE` | `/api/projects/{id}` | Delete project and all its data |

### `POST /api/projects` — Create Project

**Request:**
```json
{
  "name": "My Classification Project",
  "task": "classification"
}
```

**Response:**
```json
{
  "status": "ok",
  "data": {
    "id": "proj-a1b2c3d4",
    "name": "My Classification Project",
    "task": "classification",
    "created_at": "2026-02-19T10:00:00Z"
  }
}
```

---

## 2. Datasets

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/datasets/{project_id}` | Get dataset info (stats, class list) |
| `POST` | `/api/datasets/{project_id}/import/local` | Import from local path |
| `POST` | `/api/datasets/{project_id}/import/upload` | Import via ZIP upload |
| `GET` | `/api/datasets/{project_id}/images` | List images (paginated) |
| `GET` | `/api/datasets/{project_id}/images/{filename}` | Get image file |
| `GET` | `/api/datasets/{project_id}/thumbnails/{filename}` | Get thumbnail |
| `GET` | `/api/datasets/{project_id}/images/{filename}/info` | Get image annotation info |
| `DELETE` | `/api/datasets/{project_id}` | Clear dataset |

### Query Parameters for `GET /api/datasets/{project_id}/images`

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `page` | int | 1 | Page number |
| `page_size` | int | 50 | Items per page |
| `sort_by` | string | `"filename"` | Sort field: `filename`, `class`, `size` |
| `sort_order` | string | `"asc"` | `asc` or `desc` |
| `filter_class` | string | — | Filter by class name |
| `search` | string | — | Filename search |

---

## 3. Splits

Splits are stored inline inside `dataset.json` (`split_names` + per-image `split` lists). These endpoints read/write that file directly.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/splits/{project_id}` | List all splits (names + stats) |
| `POST` | `/api/splits/{project_id}` | Create a new split |
| `GET` | `/api/splits/{project_id}/{split_name}` | Get split details (stats, per-class distribution) |
| `DELETE` | `/api/splits/{project_id}/{split_name}` | Delete split |
| `GET` | `/api/splits/{project_id}/preview` | Preview a split before creating |

### `POST /api/splits/{project_id}` — Create Split

**Request:**
```json
{
  "name": "80-10-10",
  "ratios": { "train": 0.8, "val": 0.1, "test": 0.1 },
  "seed": 42
}
```

Backend appends the name to `split_names`, appends a value (`"train"`, `"val"`, `"test"`, or `"none"`) to each image's `split` list, and saves `dataset.json`.

**Response:**
```json
{
  "status": "ok",
  "data": {
    "split_name": "80-10-10",
    "stats": {
      "train": 800, "val": 100, "test": 100
    }
  }
}
```

---

## 4. Training (Experiments)

### Experiments

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/training/{project_id}/experiments` | List experiments |
| `POST` | `/api/training/{project_id}/experiments` | Create experiment |
| `GET` | `/api/training/{project_id}/experiments/{exp_id}` | Get experiment config |
| `PATCH` | `/api/training/{project_id}/experiments/{exp_id}` | Update experiment config |
| `DELETE` | `/api/training/{project_id}/experiments/{exp_id}` | Delete experiment + all results |
| `POST` | `/api/training/{project_id}/experiments/{exp_id}/train` | Start training |
| `POST` | `/api/training/{project_id}/experiments/{exp_id}/stop` | Stop training |
| `POST` | `/api/training/{project_id}/experiments/{exp_id}/resume` | Resume interrupted training |
| `POST` | `/api/training/{project_id}/experiments/{exp_id}/restart` | Restart experiment (deletes results, resets to created) |
| `GET` | `/api/training/{project_id}/experiments/{exp_id}/metrics` | Get metrics JSON |
| `GET` | `/api/training/{project_id}/experiments/{exp_id}/stream` | SSE event stream |

### `POST .../train` — Start Training

**Response:**
```json
{
  "status": "ok",
  "data": {
    "experiment_id": "exp-a1b2c3d4",
    "status": "pending"
  }
}
```

---

## 5. Evaluation

Evaluation is **1:1 with an experiment** — each experiment can have at most one evaluation at a time. There is no separate evaluation ID; the experiment ID is the identifier. Evaluation data lives inside the experiment folder at `experiments/<exp-id>/evaluation/`.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/evaluation/{project_id}/{experiment_id}` | Start evaluation for an experiment |
| `GET` | `/api/evaluation/{project_id}/{experiment_id}` | Get evaluation status + aggregate metrics |
| `DELETE` | `/api/evaluation/{project_id}/{experiment_id}` | Reset (delete) evaluation data |
| `GET` | `/api/evaluation/{project_id}/{experiment_id}/results` | Per-image results (paginated) |
| `GET` | `/api/evaluation/{project_id}/{experiment_id}/checkpoints` | List available checkpoint files |

### Query Parameters for Results

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `page` | int | 1 | Page number |
| `page_size` | int | 50 | Items per page |
| `sort_by` | string | `"filename"` | `filename`, `confidence`, `error` |
| `filter_correct` | bool | — | `true` for correct only, `false` for errors |
| `filter_class` | string | — | Filter by ground truth class |
| `filter_subset` | string | — | Filter by split subset (e.g., `"test"`, `"val"`) |

### `POST /api/evaluation/{project_id}/{experiment_id}` — Start Evaluation

Requires the experiment to have `status == "completed"`. Fails if an evaluation is already running.

**Request:**
```json
{
  "checkpoint": "best",
  "split_subsets": ["test"],
  "batch_size": 32,
  "device": "cuda:0"
}
```

**Response:**
```json
{
  "status": "ok",
  "data": { "status": "pending" }
}
```

### `DELETE /api/evaluation/{project_id}/{experiment_id}` — Reset Evaluation

Immediately deletes the `evaluation/` subfolder inside the experiment. No confirmation required.

### `GET /api/evaluation/{project_id}/{experiment_id}/checkpoints` — List Checkpoints

Returns only checkpoint files that actually exist in the experiment's `checkpoints/` directory.

**Response:**
```json
{
  "status": "ok",
  "data": { "checkpoints": ["best", "last"] }
}
```

---

## 6. Export

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/export/{project_id}` | List exports |
| `POST` | `/api/export/{project_id}` | Create + start export |
| `GET` | `/api/export/{project_id}/{export_id}` | Get export status |
| `DELETE` | `/api/export/{project_id}/{export_id}` | Delete export |
| `GET` | `/api/export/{project_id}/{export_id}/download` | Download exported model |
| `GET` | `/api/export/formats` | List available formats |

### `POST /api/export/{project_id}` — Create Export

**Request:**
```json
{
  "experiment_id": "exp-a1b2c3d4",
  "checkpoint": "best",
  "format": "onnx",
  "options": {
    "opset_version": 17,
    "input_shape": [1, 3, 224, 224],
    "dynamic_axes": { "input": { "0": "batch_size" }, "output": { "0": "batch_size" } },
    "simplify": true
  }
}
```

---

## 7. System

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/system/health` | Health check |
| `GET` | `/api/system/devices` | List available devices (CPU + GPUs) |
| `GET` | `/api/system/workspace` | Workspace stats (disk usage) |

### `GET /api/system/devices`

Returns all available training devices. The frontend uses this to populate the device selector.

**Response:**
```json
{
  "status": "ok",
  "data": {
    "devices": [
      { "id": "cpu", "name": "CPU", "type": "cpu" },
      { "id": "gpu:0", "name": "NVIDIA RTX 4090", "type": "gpu", "memory_total_mb": 24564, "memory_free_mb": 22100 },
      { "id": "gpu:1", "name": "NVIDIA RTX 4090", "type": "gpu", "memory_total_mb": 24564, "memory_free_mb": 24000 }
    ],
    "cuda_available": true,
    "cuda_version": "12.1",
    "default_selected": ["gpu:0"]
  }
}
```

`default_selected` contains the recommended initial selection: `["gpu:0"]` if a GPU is available, otherwise `["cpu"]`.

---

## 8. Related Documents

- API overview → [00-api-overview.md](00-api-overview.md)
- Error handling → [02-error-handling.md](02-error-handling.md)
