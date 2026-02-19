# Experiment Tracking — Checkpoints

This document describes checkpoint saving, management, and resumption.

---

## 1. Checkpoint Files

Each experiment's training saves two checkpoints:

| File | Description | When Updated |
|------|-------------|-------------|
| `best.ckpt` | Model with best monitored metric | Whenever a new best is achieved |
| `last.ckpt` | Model at the most recent epoch | Every epoch |

Location: `projects/<project-id>/experiments/<exp-id>/checkpoints/`

---

## 2. Checkpoint Contents

Lightning checkpoints (`.ckpt`) contain:

| Content | Purpose |
|---------|---------|
| `state_dict` | Model weights |
| `optimizer_states` | Optimizer state (for resumption) |
| `lr_schedulers` | Scheduler state (for resumption) |
| `epoch` | Current epoch number |
| `global_step` | Global training step |
| `callbacks` | Callback states (EarlyStopping counter, etc.) |
| `hyper_parameters` | LightningModule hyperparameters |

---

## 3. Checkpoint Selection

Different downstream operations use different checkpoints:

| Operation | Which Checkpoint |
|-----------|-----------------|
| **Evaluation** | `best.ckpt` (default) or user-selected |
| **Export** | `best.ckpt` (default) or user-selected |
| **Resume training** | `last.ckpt` |

---

## 4. Resume from Checkpoint

If training is interrupted (crash, cancellation, manual stop):

1. The `last.ckpt` contains the full training state.
2. User can resume by clicking "Resume" on a failed or cancelled experiment.
3. The Trainer is created with `ckpt_path=last.ckpt`:

```python
trainer.fit(module, datamodule, ckpt_path=str(exp_dir / "checkpoints" / "last.ckpt"))
```

4. Training continues from the last saved epoch with all states restored.

### Resume UI

- If an experiment has status `"failed"` or `"cancelled"` and has a `last.ckpt`, show a **"Resume"** button.
- Resuming continues from the last checkpoint (same experiment, appends to `metrics.json`).

---

## 5. Checkpoint Size

Checkpoint sizes vary by model:

| Model | Approx. Size |
|-------|-------------|
| ResNet-18 | ~45 MB |
| ResNet-50 | ~100 MB |
| EfficientNet-B0 | ~20 MB |
| Faster R-CNN (ResNet-50) | ~160 MB |
| DeepLabV3 (ResNet-50) | ~165 MB |
| Mask R-CNN (ResNet-50) | ~170 MB |

Both `best.ckpt` and `last.ckpt` are the same size. Total per experiment: ~2× model size.

---

## 6. Checkpoint Metadata

For quick reference without loading the full `.ckpt`, metadata is stored in `experiment.json`:

```json
{
  "best_epoch": 38,
  "best_metric": { "val_accuracy": 0.956 },
  "checkpoints": {
    "best": {
      "path": "checkpoints/best.ckpt",
      "size_mb": 100.2,
      "epoch": 38
    },
    "last": {
      "path": "checkpoints/last.ckpt",
      "size_mb": 100.2,
      "epoch": 50
    }
  }
}
```

---

## 7. Cleanup

- Checkpoints are kept indefinitely unless the user deletes or restarts an experiment.
- No automatic cleanup — storage management is left to the user.
- The GUI shows checkpoint sizes in the experiment detail view to help users manage disk space.

---

## 8. Related Documents

- Experiment management → [00-run-management.md](00-run-management.md)
- ModelCheckpoint callback → [../05-training/04-callbacks-logging.md](../05-training/04-callbacks-logging.md)
- Evaluation (uses checkpoints) → [../07-evaluation/00-evaluation-pipeline.md](../07-evaluation/00-evaluation-pipeline.md)
- Export (uses checkpoints) → [../08-export/00-export-overview.md](../08-export/00-export-overview.md)
