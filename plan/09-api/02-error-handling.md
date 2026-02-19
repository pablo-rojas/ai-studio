# API — Error Handling

This document describes the error handling strategy for the FastAPI API layer.

---

## 1. Exception Hierarchy

```python
# app/core/exceptions.py

class AIStudioError(Exception):
    """Base exception for all AI Studio errors."""
    status_code: int = 500
    error_code: str = "INTERNAL_ERROR"
    
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class NotFoundError(AIStudioError):
    status_code = 404
    error_code = "NOT_FOUND"


class ConflictError(AIStudioError):
    status_code = 409
    error_code = "CONFLICT"


class ValidationError(AIStudioError):
    status_code = 422
    error_code = "VALIDATION_ERROR"


class TrainingInProgressError(ConflictError):
    error_code = "TRAINING_IN_PROGRESS"
    
    def __init__(self):
        super().__init__("An experiment is already training")


class DatasetNotImportedError(ValidationError):
    error_code = "DATASET_NOT_IMPORTED"
    
    def __init__(self, project_id: str):
        super().__init__(f"Project {project_id} has no imported dataset")


class SplitNotFoundError(NotFoundError):
    error_code = "SPLIT_NOT_FOUND"


class ExperimentNotFoundError(NotFoundError):
    error_code = "EXPERIMENT_NOT_FOUND"


class RunNotFoundError(NotFoundError):
    error_code = "RUN_NOT_FOUND"


class CheckpointNotFoundError(NotFoundError):
    error_code = "CHECKPOINT_NOT_FOUND"


class ExportFormatUnavailableError(ValidationError):
    error_code = "EXPORT_FORMAT_UNAVAILABLE"
    
    def __init__(self, format: str):
        super().__init__(f"Export format '{format}' is not yet available")


class GPUNotAvailableError(ValidationError):
    error_code = "GPU_NOT_AVAILABLE"
```

---

## 2. Global Exception Handler

```python
# app/main.py

from fastapi import Request
from fastapi.responses import JSONResponse

@app.exception_handler(AIStudioError)
async def aistudio_error_handler(request: Request, exc: AIStudioError):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "error": {
                "code": exc.error_code,
                "message": exc.message
            }
        }
    )

@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    # Log the full traceback
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred"
            }
        }
    )
```

---

## 3. Pydantic Validation

FastAPI automatically validates request bodies using Pydantic schemas. Validation errors return 422 with details:

```json
{
  "detail": [
    {
      "loc": ["body", "task"],
      "msg": "value is not a valid enumeration member",
      "type": "type_error.enum",
      "ctx": { "enum_values": ["classification", "anomaly_detection", "..."] }
    }
  ]
}
```

### Custom Validators

```python
from pydantic import BaseModel, validator

class CreateSplitRequest(BaseModel):
    name: str
    ratios: dict[str, float]
    seed: int = 42
    
    @validator("ratios")
    def ratios_must_sum_to_one(cls, v):
        total = sum(v.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Ratios must sum to 1.0, got {total}")
        return v
    
    @validator("ratios")
    def must_have_train_and_val(cls, v):
        if "train" not in v or "val" not in v:
            raise ValueError("Ratios must include 'train' and 'val'")
        return v
```

---

## 4. Error Scenarios

| Scenario | Exception | HTTP | Error Code |
|----------|-----------|------|------------|
| Project not found | `NotFoundError` | 404 | `NOT_FOUND` |
| Create split without dataset | `DatasetNotImportedError` | 422 | `DATASET_NOT_IMPORTED` |
| Start training while another is running | `TrainingInProgressError` | 409 | `TRAINING_IN_PROGRESS` |
| Evaluate with missing checkpoint | `CheckpointNotFoundError` | 404 | `CHECKPOINT_NOT_FOUND` |
| Export to unavailable format | `ExportFormatUnavailableError` | 422 | `EXPORT_FORMAT_UNAVAILABLE` |
| Invalid split ratios | Pydantic validation | 422 | `VALIDATION_ERROR` |
| No GPU available | `GPUNotAvailableError` | 422 | `GPU_NOT_AVAILABLE` |
| Disk full during training | `AIStudioError` | 500 | `INTERNAL_ERROR` |
| Corrupted checkpoint file | `AIStudioError` | 500 | `INTERNAL_ERROR` |

---

## 5. HTMX Error Display

When HTMX requests fail, the error must be displayed in the GUI:

```python
@app.exception_handler(AIStudioError)
async def aistudio_error_handler(request: Request, exc: AIStudioError):
    if request.headers.get("HX-Request"):
        # Return HTML fragment for HTMX
        return templates.TemplateResponse("fragments/error_toast.html", {
            "request": request,
            "error_message": exc.message
        }, status_code=exc.status_code)
    
    # Return JSON for API calls
    return JSONResponse(...)
```

The `error_toast.html` fragment renders a dismissible error toast notification that appears at the top of the page.

---

## 6. Training Error Handling

Training can fail due to various reasons. These are captured in `experiment.json`:

```json
{
  "status": "failed",
  "error": {
    "type": "RuntimeError",
    "message": "CUDA out of memory. Tried to allocate 2.00 GiB",
    "traceback": "..."
  }
}
```

The training wrapper catches all exceptions:

```python
async def run_training(experiment_id: str):
    try:
        update_experiment_status(experiment_id, "training")
        trainer.fit(module, datamodule)
        update_experiment_status(experiment_id, "completed")
    except Exception as e:
        update_experiment_status(experiment_id, "failed", error={
            "type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc()
        })
    finally:
        training_lock.release()
```

---

## 7. Related Documents

- API overview → [00-api-overview.md](00-api-overview.md)
- Endpoint reference → [01-endpoints.md](01-endpoints.md)
