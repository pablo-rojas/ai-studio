# Experiment Tracking — Experiment Management

This document describes how training experiments are organized, stored, and managed.

---

## 1. Concepts

| Entity | Description | Cardinality |
|--------|-------------|-------------|
| **Project** | A workspace for one task | Has one dataset, many experiments |
| **Experiment** | A training configuration + its single execution and results | Produces checkpoints + metrics |

An **experiment** is a self-contained unit combining configuration and training results. Each experiment has exactly one training execution. When the user creates an experiment, they configure parameters then click "Train". Training produces checkpoints and metrics stored directly within the experiment folder.

**Configuration is locked once training starts.** To change parameters, the user must **restart** the experiment, which deletes all results, checkpoints, and metrics. To try a different configuration without losing existing results, the user creates a new experiment (optionally by duplicating an existing one).

---

## 2. Experiment Lifecycle

```
                    ┌────────────┐
                    │   created  │  ← user creates experiment, configures hparams
                    └─────┬──────┘
                          │  user clicks "Train"
                          ▼
                    ┌────────────┐
                    │  training  │  ← trainer.fit() running in background
                    └─────┬──────┘
                    ┌─────┴─────┐
                    │           │
                    ▼           ▼
             ┌───────────┐ ┌────────┐
             │ completed │ │ failed │
             └───────────┘ └────┬───┘
                                │  user clicks "Resume"
                                ▼
                          ┌────────────┐
                          │  training  │
                          └────────────┘
```

At any point after training has started, the user may **restart** the experiment:
- All results (metrics, checkpoints, logs) are deleted.
- Configuration is unlocked for editing.
- Status resets to `"created"`.

Valid `status` values: `"created"`, `"pending"`, `"training"`, `"completed"`, `"failed"`, `"cancelled"`.

---

## 3. Storage Layout

```
projects/<project-id>/experiments/
├── experiments_index.json          # List of all experiments
├── exp-a1b2c3d4/
│   ├── experiment.json             # Experiment configuration + status + results
│   ├── metrics.json                # Per-epoch metrics (created when training starts)
│   ├── checkpoints/                # (created when training starts)
│   │   ├── best.ckpt
│   │   └── last.ckpt
│   └── logs/
│       └── training.log
└── exp-m3n4o5p6/
    └── ...
```

### `experiments_index.json`

```json
{
  "experiments": [
    {
      "id": "exp-a1b2c3d4",
      "name": "ResNet50 baseline",
      "created_at": "2026-02-19T11:00:00Z",
      "status": "completed",
      "best_metric_value": 0.956
    }
  ]
}
```

### `experiment.json`

See [02-data-layer/00-storage-layout.md](../02-data-layer/00-storage-layout.md) for the full schema. Contains both configuration and execution state:

**Configuration sections** (editable only while status is `"created"`):
- `description`: optional free-text notes about the experiment (what changed, purpose, etc.).
- `model`: backbone + head config.
- `hyperparameters`: optimizer, LR, scheduler, etc.
- `augmentations`: train + val pipelines.
- `hardware`: device selection (multi-select) + precision.
- `split_name`: which split to use (must match an entry in `dataset.json` `split_names`).

**Execution state** (updated by the training pipeline):
- `status`: current experiment status.
- `started_at` / `completed_at`: timestamps.
- `best_epoch`, `best_metric`: best checkpoint info.
- `final_metrics`: metrics from the last epoch.
- `error`: error message if failed.

```json
{
  "id": "exp-a1b2c3d4",
  "name": "ResNet50 baseline",
  "description": "Baseline experiment with default hyperparameters",
  "created_at": "2026-02-19T11:00:00Z",
  "status": "completed",
  "started_at": "2026-02-19T11:05:00Z",
  "completed_at": "2026-02-19T11:45:00Z",
  "split_name": "80-10-10",
  "model": { "backbone": "resnet50", "head": "classification", "pretrained": true, "freeze_backbone": false },
  "hyperparameters": { "optimizer": "adam", "learning_rate": 0.001, "weight_decay": 0.0001, "scheduler": "cosine", "batch_size": 32, "batch_multiplier": 1, "max_epochs": 50, "early_stopping_patience": 10 },
  "augmentations": { "train": [ "..." ], "val": [ "..." ] },
  "hardware": { "selected_devices": ["gpu:0"], "precision": "32" },
  "best_epoch": 38,
  "best_metric": { "val_accuracy": 0.956 },
  "final_metrics": {
    "train_loss": 0.042,
    "val_loss": 0.118,
    "val_accuracy": 0.956,
    "val_f1": 0.951
  },
  "error": null
}
```

---

## 4. Experiment List (Left Column)

The Training page's left column shows a list of all experiments for the current project:

| Column | Content |
|--------|---------|
| Name | User-given name (editable) |
| Status icon | ● Training (blue spinner), ✓ Completed (green), ✗ Failed (red), ⊘ Cancelled (orange), ○ Not trained (grey) |
| Best metric | Best metric value (if training completed) |

Controls:
- **"New Experiment"** button at the top.
- **Click** to select an experiment and populate the center + right columns.
- **Right-click / menu**: rename, duplicate, delete.

---

## 5. Creating an Experiment

1. User clicks "New Experiment".
2. A new experiment is created with:
   - Auto-generated ID.
   - Default name (e.g., "Experiment 3").
   - Task-specific default hyperparameters, augmentations, and first available split.
   - Status: `"created"`.
3. The experiment appears selected in the left column.
4. Center column shows the config form, pre-populated with defaults.
5. User modifies config as desired.
6. Changes are saved automatically (debounced) or on explicit "Save" button.

---

## 6. Training an Experiment

1. User clicks "Train" button (in center or right column).
2. Configuration is locked (form becomes read-only).
3. `experiment.json` is updated with `status: "pending"`.
4. Training starts as a background task.
5. Right column switches to live training view (progress bar + metric charts).
6. `metrics.json` is created and updated after each epoch.
7. On completion, `experiment.json` is updated with final metrics, status, and best epoch info.

---

## 7. Viewing Results (Right Column)

When an experiment is selected, the right column shows:

### If not yet trained (status: `"created"`)
- Message: "Not yet trained. Click 'Train' to start training."

### If training (status: `"training"`)
- **Training progress**: epoch progress bar, ETA, live loss chart.
- **"Stop Training"** button to cancel.

### If completed (status: `"completed"`)
- **Final metrics**: summary table of all metrics.
- **Loss curves**: train_loss and val_loss over epochs (Chart.js line chart).
- **Metric curves**: additional metric curves (accuracy, mAP, etc.).
- **Learning rate curve**: LR schedule over epochs.
- **Checkpoint info**: best epoch, best metric value, file sizes.
- **Quick action buttons**: [Evaluate] [Export].

### If failed (status: `"failed"`)
- **Error message** and traceback.
- **"Resume"** button (if `last.ckpt` exists).

---

## 8. Restarting an Experiment

If the user wants to change configuration after training has started or completed:

1. User clicks "Restart" button.
2. A confirmation dialog warns: "This will delete all training results, checkpoints, and metrics. Continue?"
3. On confirmation:
   - `metrics.json`, `checkpoints/`, and `logs/` are deleted.
   - `experiment.json` execution state is cleared (status reset to `"created"`, timestamps/metrics removed).
   - Configuration fields become editable again.
4. User can now modify parameters and re-train.

---

## 9. Comparing Experiments

Users compare results across experiments by:
1. Looking at the experiment list (which shows best metric per experiment).
2. Selecting different experiments and comparing their right-column metrics.
3. (Future) A dedicated comparison view overlaying loss/metric curves from multiple experiments.

---

## 10. Deleting Experiments

- **Delete an experiment**: removes the experiment folder (config, checkpoints, metrics, logs). Warns if the experiment's checkpoint is referenced by an evaluation or export.

---

## 11. Related Documents

- Training pipeline → [../05-training/00-training-pipeline.md](../05-training/00-training-pipeline.md)
- Metrics logging → [01-metrics-logging.md](01-metrics-logging.md)
- Checkpoints → [02-checkpoints.md](02-checkpoints.md)
- Training GUI page → [../10-gui/01-pages/04-training-page.md](../10-gui/01-pages/04-training-page.md)
