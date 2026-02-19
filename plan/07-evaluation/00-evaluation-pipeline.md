# Evaluation — Evaluation Pipeline

This document describes the end-to-end evaluation pipeline: loading a trained model, running inference on a test split, and storing results.

---

## 1. Pipeline Overview

```
User selects: Checkpoint + Split (test set)
           │
           ▼
   Load model from checkpoint
   Load dataset for selected split
           │
           ▼
   Run batch inference (no gradients)
   Collect predictions for each image
           │
           ▼
   Compute per-image results
   (prediction, confidence, correct/incorrect)
           │
           ▼
   Compute aggregate metrics
   (accuracy, mAP, mIoU, MAE, etc.)
           │
           ▼
   Store results:
   ├── evaluation.json    (config + aggregate metrics)
   └── results.json       (all per-image predictions in a single file)
```

---

## 2. Evaluation Configuration

```json
{
  "id": "eval-abc123",
  "name": "Test set evaluation",
  "experiment_id": "exp-a1b2c3d4",
  "checkpoint": "best",
  "split_name": "80-10-10",
  "split_subset": "test",
  "batch_size": 32,
  "device": "cuda:0",
  "created_at": "2026-02-19T14:00:00Z"
}
```

| Field | Description |
|-------|-------------|
| `experiment_id` | Which experiment to load the checkpoint from |
| `checkpoint` | `"best"` or `"last"` |
| `split_name` | Name matching an entry in `dataset.json` `split_names` — identifies which split to use |
| `split_subset` | Which subset: `"test"` (default), `"val"`, or `"train"` |
| `batch_size` | Inference batch size |
| `device` | GPU device for inference |

---

## 3. Evaluator Implementation

```python
# app/evaluation/evaluator.py

class Evaluator:
    def __init__(self, config: EvaluationConfig, project_path: Path):
        self.config = config
        self.project_path = project_path
    
    def run(self) -> EvaluationResult:
        # 1. Load model from checkpoint
        model = self._load_model()
        model.eval()
        model.to(self.config.device)
        
        # 2. Build dataset for the selected split subset
        dataloader = self._build_dataloader()
        
        # 3. Run inference
        per_image_results = []
        with torch.no_grad():
            for batch_idx, (images, targets, filenames) in enumerate(dataloader):
                images = images.to(self.config.device)
                predictions = model(images)
                
                # Process each image in the batch
                for i in range(len(filenames)):
                    result = self._process_prediction(
                        filename=filenames[i],
                        prediction=predictions[i],
                        target=targets[i]
                    )
                    per_image_results.append(result)
        
        # 4. Compute aggregate metrics
        aggregate = compute_aggregate_metrics(per_image_results, task=self.task)
        
        # 5. Store results
        self._save_results(per_image_results, aggregate)
        
        return EvaluationResult(per_image=per_image_results, aggregate=aggregate)
```

---

## 4. Background Execution

Like training, evaluation runs as a background task:

```python
@router.post("/api/evaluation/run")
async def start_evaluation(config: EvaluationConfig, background_tasks: BackgroundTasks):
    eval_id = create_evaluation(config)
    background_tasks.add_task(run_evaluation, eval_id)
    return {"evaluation_id": eval_id, "status": "running"}
```

Progress is reported via polling (percentage of images processed).

---

## 5. Evaluation Status

```
pending → running → completed
                  → failed
                  → cancelled
```

Stored in `evaluation.json`:
```json
{
  "status": "completed",
  "progress": { "processed": 120, "total": 120 },
  "started_at": "2026-02-19T14:00:30Z",
  "completed_at": "2026-02-19T14:02:15Z"
}
```

---

## 6. Storage Layout

```
projects/<project-id>/evaluations/
├── evaluations_index.json
└── eval-abc123/
    ├── evaluation.json        # Config + status + aggregate metrics
    └── results.json           # All per-image predictions
```

---

## 7. Related Documents

- Per-image results → [01-per-image-results.md](01-per-image-results.md)
- Aggregate metrics → [02-aggregate-metrics.md](02-aggregate-metrics.md)
- Checkpoints (input) → [../06-experiment-tracking/02-checkpoints.md](../06-experiment-tracking/02-checkpoints.md)
- Evaluation GUI page → [../10-gui/01-pages/05-evaluation-page.md](../10-gui/01-pages/05-evaluation-page.md)
