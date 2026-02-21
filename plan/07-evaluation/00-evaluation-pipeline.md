# Evaluation — Evaluation Pipeline

This document describes the end-to-end evaluation pipeline: loading a trained model, running inference on selected split subsets, and storing results.

---

## 1. Key Design Decision: One Evaluation Per Experiment

Evaluation is **1:1 with an experiment**. Each experiment can have at most one evaluation at a time. There is no separate evaluation ID, no evaluation index file, and no standalone `evaluations/` folder. Evaluation data lives inside the experiment folder at `experiments/<experiment-id>/evaluation/`.

To re-evaluate an experiment with different settings, the user must first **reset** (delete) the existing evaluation, then run a new one. Reset is immediate — no confirmation dialog.

---

## 2. Pipeline Overview

```
User selects an experiment (must be completed)
User configures: Checkpoint + Split subsets + Batch size + Device
           │
           ▼
   Load model from checkpoint
   Build combined dataloader from selected split subsets
           │
           ▼
   Run batch inference (no gradients)
   Collect predictions for each image (tagged with subset)
           │
           ▼
   Compute per-image results
   (prediction, confidence, correct/incorrect, subset)
           │
           ▼
   Compute aggregate metrics over the combined pool
   (accuracy, mAP, mIoU, MAE, etc.)
           │
           ▼
   Store results inside experiment folder:
   experiments/<exp-id>/evaluation/
   ├── evaluation.json    (config + status + progress)
   ├── aggregate.json     (aggregate metrics)
   └── results.json       (all per-image predictions)
```

---

## 3. Evaluation Configuration

The evaluation config is stored in `evaluation.json` inside `experiments/<exp-id>/evaluation/`. There is no separate evaluation ID — the parent experiment ID is the identifier.

```json
{
  "checkpoint": "best",
  "split_subsets": ["test"],
  "batch_size": 32,
  "device": "cuda:0",
  "status": "completed",
  "progress": { "processed": 120, "total": 120 },
  "created_at": "2026-02-19T14:00:00Z",
  "started_at": "2026-02-19T14:00:30Z",
  "completed_at": "2026-02-19T14:02:15Z",
  "error": null
}
```

| Field | Description |
|-------|-------------|
| `checkpoint` | Which checkpoint file to use. Only checkpoints that actually exist in the `checkpoints/` folder are offered (e.g., `"best"`, `"last"`) |
| `split_subsets` | List of subsets to evaluate on (e.g., `["test"]`, `["test", "val"]`). Images from all selected subsets are pooled into a single combined evaluation |
| `batch_size` | Inference batch size |
| `device` | Device for inference (e.g., `"cuda:0"`, `"cpu"`) |
| `status` | Current status: `"pending"`, `"running"`, `"completed"`, `"failed"` |
| `progress` | Images processed so far out of total |
| `error` | Error message if status is `"failed"`, otherwise `null` |

**Note:** `split_name` is not stored here — it is inherited from the parent experiment's `split_name` field in `experiment.json`.

---

## 4. Evaluator Implementation

```python
# app/evaluation/evaluator.py

class Evaluator:
    def __init__(self, config: EvaluationConfig, project_id: str, experiment_id: str):
        self.config = config
        self.project_id = project_id
        self.experiment_id = experiment_id
    
    def run(self) -> EvaluationResult:
        # 1. Load model from checkpoint
        model = self._load_model()
        model.eval()
        model.to(self.config.device)
        
        # 2. Build combined dataloader from ALL selected subsets
        #    Images are tagged with their source subset
        dataloader = self._build_combined_dataloader()
        
        # 3. Run inference
        per_image_results = []
        with torch.no_grad():
            for batch_idx, (images, targets, filenames, subsets) in enumerate(dataloader):
                images = images.to(self.config.device)
                predictions = model(images)
                
                # Process each image in the batch
                for i in range(len(filenames)):
                    result = self._process_prediction(
                        filename=filenames[i],
                        prediction=predictions[i],
                        target=targets[i],
                        subset=subsets[i]
                    )
                    per_image_results.append(result)
                
                # Update progress
                self._update_progress(len(per_image_results))
        
        # 4. Compute aggregate metrics over the combined pool
        aggregate = compute_aggregate_metrics(per_image_results, task=self.task)
        
        # 5. Store results
        self._save_results(per_image_results, aggregate)
        
        return EvaluationResult(per_image=per_image_results, aggregate=aggregate)
```

---

## 5. Background Execution

Evaluation runs as a background task:

```python
@router.post("/api/evaluation/{project_id}/{experiment_id}")
async def start_evaluation(
    project_id: str, experiment_id: str,
    config: EvaluationConfig,
    background_tasks: BackgroundTasks
):
    # Validates experiment is completed, no existing evaluation running
    evaluation_service.create_evaluation(project_id, experiment_id, config)
    background_tasks.add_task(run_evaluation, project_id, experiment_id)
    return {"status": "ok", "data": {"status": "pending"}}
```

Progress is reported via polling (percentage of images processed).

---

## 6. Evaluation Status

```
pending → running → completed
                  → failed
```

Status is stored in `evaluation.json` inside the experiment's `evaluation/` subfolder.

---

## 7. Storage Layout

Evaluation data lives **inside the experiment folder** — there is no standalone `evaluations/` directory or index file:

```
projects/<project-id>/experiments/<experiment-id>/
├── experiment.json
├── metrics.json
├── checkpoints/
│   ├── best.ckpt
│   └── last.ckpt
├── logs/
│   └── training.log
└── evaluation/
    ├── evaluation.json   # Config + status + progress
    ├── aggregate.json    # Aggregate metrics
    └── results.json      # All per-image predictions (with subset tag)
```

The `evaluation/` subfolder is created when evaluation starts and deleted entirely on reset.

---

## 8. Checkpoint Discovery

The checkpoint selector in the GUI only shows checkpoints that actually exist on disk. The service scans the `checkpoints/` directory and returns the list of available checkpoint names (e.g., `["best", "last"]` or just `["best"]`).

```python
def list_checkpoints(self, project_id: str, experiment_id: str) -> list[str]:
    """Scan checkpoints/ dir and return available checkpoint names."""
    ckpt_dir = self.paths.experiment_checkpoints_dir(project_id, experiment_id)
    return [p.stem for p in sorted(ckpt_dir.glob("*.ckpt"))]
```

---

## 9. Reset Behavior

Resetting an evaluation:
1. Deletes the entire `evaluation/` subfolder inside the experiment
2. Is immediate — no confirmation dialog
3. Allows the user to configure and run a new evaluation with different settings

This is exposed via `DELETE /api/evaluation/{project_id}/{experiment_id}`.

---

## 10. Related Documents

- Per-image results → [01-per-image-results.md](01-per-image-results.md)
- Aggregate metrics → [02-aggregate-metrics.md](02-aggregate-metrics.md)
- Checkpoints (input) → [../06-experiment-tracking/02-checkpoints.md](../06-experiment-tracking/02-checkpoints.md)
- Evaluation GUI page → [../10-gui/01-pages/05-evaluation-page.md](../10-gui/01-pages/05-evaluation-page.md)
