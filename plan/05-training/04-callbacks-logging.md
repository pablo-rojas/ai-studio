# Training — Callbacks & Logging

This document describes the Lightning callbacks used during training and how metrics/logs are persisted.

---

## 1. Callback Stack

Each training run uses the following callbacks:

| Callback | Source | Purpose |
|----------|--------|---------|
| `ModelCheckpoint` | Lightning built-in | Save best and last model checkpoints |
| `EarlyStopping` | Lightning built-in | Stop training when validation metric plateaus |
| `LearningRateMonitor` | Lightning built-in | Log learning rate per epoch |
| `JSONMetricLogger` | **Custom** | Write per-epoch metrics to `metrics.json` |
| `RunStatusUpdater` | **Custom** | Update `run.json` status on train start/end/error |
| `SSENotifier` | **Custom** | Push training progress events for live GUI updates |

---

## 2. ModelCheckpoint

```python
ModelCheckpoint(
    dirpath=run_dir / "checkpoints",
    filename="best",
    monitor="val_loss",    # Can be overridden per task (e.g., "val_mAP")
    mode="min",            # "min" for loss, "max" for accuracy/mAP
    save_last=True,        # Always save last epoch as last.ckpt
    save_top_k=1,          # Keep only the single best checkpoint
    verbose=True,
)
```

### Produced Files

- `checkpoints/best.ckpt` — best model based on monitored metric.
- `checkpoints/last.ckpt` — most recent epoch.

### Per-Task Monitor Metric

| Task | Monitor | Mode |
|------|---------|------|
| Classification | `val_accuracy` | max |
| Anomaly Detection | `val_auroc` | max |
| Object Detection | `val_mAP` | max |
| Oriented OD | `val_mAP` | max |
| Segmentation | `val_mIoU` | max |
| Instance Segmentation | `val_mAP_mask` | max |
| Regression | `val_mae` | min |

---

## 3. EarlyStopping

```python
EarlyStopping(
    monitor="val_loss",
    patience=config["early_stopping_patience"],  # default: 10
    mode="min",
    min_delta=0.0001,
    verbose=True,
)
```

- Can be disabled by setting `patience = 0` (no early stopping).
- Always monitors `val_loss` regardless of the primary metric (loss is universal).

---

## 4. LearningRateMonitor

```python
LearningRateMonitor(logging_interval="epoch")
```

Logs the current learning rate to Lightning's log system. The `JSONMetricLogger` picks this up and includes it in `metrics.json`.

---

## 5. JSONMetricLogger (Custom)

This is the core logging callback. It writes training progress to a JSON file that the GUI reads.

```python
class JSONMetricLogger(pl.Callback):
    """Write per-epoch metrics to a JSON file."""
    
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.epochs = []
    
    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        
        epoch_data = {
            "epoch": trainer.current_epoch + 1,
            "lr": trainer.optimizers[0].param_groups[0]["lr"],
            "duration_s": ...,  # measured via timer
        }
        
        # Collect all logged metrics
        for key, value in trainer.callback_metrics.items():
            epoch_data[key] = float(value)
        
        self.epochs.append(epoch_data)
        
        # Write atomically
        self._write_json()
    
    def _write_json(self):
        data = {"epochs": self.epochs}
        tmp = self.output_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(self.output_path)
```

### Output Format (`metrics.json`)

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

## 6. RunStatusUpdater (Custom)

Updates `run.json` at key lifecycle points:

```python
class RunStatusUpdater(pl.Callback):
    def __init__(self, run_json_path: Path):
        self.run_json_path = run_json_path
    
    def on_train_start(self, trainer, pl_module):
        self._update_status("running", started_at=now())
    
    def on_train_end(self, trainer, pl_module):
        best_metric = self._get_best_metric(trainer)
        self._update_status("completed", completed_at=now(),
                          best_epoch=trainer.checkpoint_callback.best_model_score,
                          final_metrics=self._collect_final_metrics(trainer))
    
    def on_exception(self, trainer, pl_module, exception):
        self._update_status("failed", error=str(exception))
```

---

## 7. SSENotifier (Custom)

Pushes real-time events to the frontend via Server-Sent Events:

```python
class SSENotifier(pl.Callback):
    """Push training events to an SSE queue for live GUI updates."""
    
    def __init__(self, event_queue):
        self.event_queue = event_queue
    
    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        self.event_queue.put({
            "type": "epoch_end",
            "epoch": trainer.current_epoch + 1,
            "max_epochs": trainer.max_epochs,
            "metrics": {k: float(v) for k, v in trainer.callback_metrics.items()},
        })
    
    def on_train_end(self, trainer, pl_module):
        self.event_queue.put({"type": "training_complete"})
```

The API layer reads from the queue and streams events to the connected GUI:

```python
@router.get("/api/training/{run_id}/stream")
async def training_stream(run_id: str):
    async def event_generator():
        while True:
            event = await queue.get()
            yield f"data: {json.dumps(event)}\n\n"
            if event["type"] == "training_complete":
                break
    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

---

## 8. Training Log File

A plain-text log file at `runs/<run-id>/logs/training.log` for debugging:

- Lightning's default logging output.
- Redirect Python's `logging` module to this file.
- Includes warnings, errors, and GPU memory stats.

---

## 9. Related Documents

- Training pipeline → [00-training-pipeline.md](00-training-pipeline.md)
- Experiment tracking → [../06-experiment-tracking/00-run-management.md](../06-experiment-tracking/00-run-management.md)
- Checkpoints → [../06-experiment-tracking/02-checkpoints.md](../06-experiment-tracking/02-checkpoints.md)
- Training GUI (live charts) → [../10-gui/01-pages/04-training-page.md](../10-gui/01-pages/04-training-page.md)
