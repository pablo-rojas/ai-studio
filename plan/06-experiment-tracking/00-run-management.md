# Experiment Tracking — Run Management

This document describes how training experiments and runs are organized, stored, and managed.

---

## 1. Concepts

| Entity | Description | Cardinality |
|--------|-------------|-------------|
| **Project** | A workspace for one task | Has one dataset, many experiments |
| **Experiment** | A training configuration (model + hparams + augmentations + split) | Has many runs |
| **Run** | A single execution of an experiment | Produces checkpoints + metrics |

An **experiment** is a reusable configuration template. Each time the user clicks "Run", a new **run** is created under that experiment with a frozen copy of the config. This allows:
- Re-running the same experiment (e.g., after code changes).
- Comparing runs with different configs by creating new experiments.

---

## 2. Experiment Lifecycle

```
                    ┌────────────┐
                    │   created  │  ← user creates experiment, configs hparams
                    └─────┬──────┘
                          │  user clicks "Run"
                          ▼
                    ┌────────────┐
                    │   active   │  ← has at least one run
                    └─────┬──────┘
                          │  training completes/fails
                          ▼
                    ┌────────────┐
                    │   active   │  ← experiment stays active, runs have their own status
                    └────────────┘
```

Experiments are never "completed" — they can always be re-run or reconfigured.

---

## 3. Storage Layout

```
projects/<project-id>/experiments/
├── experiments_index.json          # List of all experiments
├── exp-a1b2c3d4/
│   ├── experiment.json             # Experiment configuration (mutable)
│   └── runs/
│       ├── run-e5f6g7h8/
│       │   ├── run.json            # Run metadata + status
│       │   ├── config.json         # Frozen config snapshot
│       │   ├── metrics.json        # Per-epoch metrics
│       │   ├── checkpoints/
│       │   │   ├── best.ckpt
│       │   │   └── last.ckpt
│       │   └── logs/
│       │       └── training.log
│       └── run-i9j0k1l2/
│           └── ...
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
      "run_count": 2,
      "latest_run_status": "completed"
    }
  ]
}
```

### `experiment.json`

See [02-data-layer/00-storage-layout.md](../02-data-layer/00-storage-layout.md) for the full schema. Key sections:
- `model`: backbone + head config.
- `hyperparameters`: optimizer, LR, scheduler, etc.
- `augmentations`: train + val pipelines.
- `hardware`: device selection (multi-select) + precision.
- `split_index`: which split to use (index into `dataset.json` `split_names`).

### `run.json`

```json
{
  "id": "run-e5f6g7h8",
  "experiment_id": "exp-a1b2c3d4",
  "status": "completed",
  "started_at": "2026-02-19T11:05:00Z",
  "completed_at": "2026-02-19T11:45:00Z",
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

### `config.json`

A frozen copy of `experiment.json` at the time the run was started. This ensures reproducibility even if the experiment is later reconfigured.

---

## 4. Experiment List (Left Column)

The Training page's left column shows a list of all experiments for the current project:

| Column | Content |
|--------|---------|
| Name | User-given name (editable) |
| Status icon | ● Running (blue spinner), ✓ Completed (green), ✗ Failed (red), ○ No runs (grey) |
| Run count | Number of runs |
| Best metric | Best result across all runs |

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
3. The experiment appears selected in the left column.
4. Center column shows the config form, pre-populated with defaults.
5. User modifies config as desired.
6. Changes are saved automatically (debounced) or on explicit "Save" button.

---

## 6. Running an Experiment

1. User clicks "Run" button (in center or right column).
2. A new run is created under the experiment: `runs/<run-id>/`.
3. `config.json` is written as a frozen snapshot.
4. `run.json` is created with `status: "pending"`.
5. Training starts as a background task.
6. Right column switches to live training view (progress bar + metric charts).
7. On completion, `run.json` is updated with final metrics and status.

---

## 7. Viewing Run Results (Right Column)

When an experiment is selected, the right column shows:

### If no runs exist
- Message: "No runs yet. Click 'Run' to start training."

### If runs exist
- **Run selector** dropdown (if multiple runs): shows run ID, date, status.
- **Training progress** (if running): epoch progress bar, ETA, live loss chart.
- **Final metrics** (if completed): summary table of all metrics.
- **Loss curves**: train_loss and val_loss over epochs (Chart.js line chart).
- **Metric curves**: additional metric curves (accuracy, mAP, etc.).
- **Learning rate curve**: LR schedule over epochs.
- **Checkpoint info**: best epoch, best metric value, file sizes.

---

## 8. Comparing Runs

Users may want to compare results across experiments. This is primarily done by:
1. Looking at the experiment list (which shows best metric per experiment).
2. Selecting different experiments and comparing their right-column metrics.
3. (Future) A dedicated comparison view overlaying loss/metric curves from multiple runs.

---

## 9. Deleting Experiments & Runs

- **Delete a run**: removes the run folder. Warns if the run's checkpoint is referenced by an evaluation or export.
- **Delete an experiment**: removes the experiment folder and all its runs. Warns about dependent evaluations/exports.

---

## 10. Related Documents

- Training pipeline → [../05-training/00-training-pipeline.md](../05-training/00-training-pipeline.md)
- Metrics logging → [01-metrics-logging.md](01-metrics-logging.md)
- Checkpoints → [02-checkpoints.md](02-checkpoints.md)
- Training GUI page → [../10-gui/01-pages/04-training-page.md](../10-gui/01-pages/04-training-page.md)
