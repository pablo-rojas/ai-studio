# Experiment Tracking — Metrics Logging

This document describes the per-epoch metric logging format and how metrics are visualized.

---

## 1. Metrics JSON Format

Each training run produces a `metrics.json` file updated after every epoch:

```json
{
  "epochs": [
    {
      "epoch": 1,
      "train_loss": 2.3100,
      "val_loss": 1.8900,
      "val_accuracy": 0.3200,
      "val_f1": 0.2800,
      "lr": 0.001000,
      "duration_s": 45.2
    },
    {
      "epoch": 2,
      "train_loss": 1.4500,
      "val_loss": 1.1200,
      "val_accuracy": 0.5800,
      "val_f1": 0.5400,
      "lr": 0.000990,
      "duration_s": 44.8
    }
  ]
}
```

---

## 2. Logged Metrics by Task

### Always Logged

| Key | Description |
|-----|-------------|
| `epoch` | Epoch number (1-based) |
| `train_loss` | Average training loss |
| `val_loss` | Average validation loss |
| `lr` | Current learning rate |
| `duration_s` | Epoch wall-clock duration in seconds |

### Task-Specific

| Task | Additional Metrics |
|------|-------------------|
| Classification | `val_accuracy`, `val_f1`, `val_precision`, `val_recall` |
| Anomaly Detection | `val_auroc`, `val_f1` |
| Object Detection | `val_mAP`, `val_mAP_50`, `val_mAP_75` |
| Oriented OD | `val_mAP`, `val_mAP_50` |
| Segmentation | `val_mIoU`, `val_pixel_acc`, `val_dice` |
| Instance Segmentation | `val_mAP_mask`, `val_mAP_box` |
| Regression | `val_mae`, `val_mse`, `val_rmse`, `val_r2` |

---

## 3. Visualization: Charts

### 3.1 Loss Curves

- **Chart type**: Line chart (Chart.js).
- **X-axis**: Epoch.
- **Y-axis**: Loss value.
- **Series**: `train_loss` (blue), `val_loss` (orange).
- Shows best epoch marker (vertical dashed line at epoch with lowest val_loss).

### 3.2 Metric Curves

- **Chart type**: Line chart.
- **X-axis**: Epoch.
- **Y-axis**: Metric value.
- **Series**: one line per metric (e.g., accuracy, F1, mAP).
- Separate chart(s) from loss to allow different Y-axis scales.

### 3.3 Learning Rate Schedule

- **Chart type**: Line chart.
- **X-axis**: Epoch.
- **Y-axis**: Learning rate (log scale if values span orders of magnitude).

---

## 4. Live Updates

During training, charts update in real-time:

1. The `JSONMetricLogger` callback writes after each epoch.
2. The GUI either:
   - **Polls** the `/api/training/{run_id}/metrics` endpoint every 2–5 seconds.
   - **SSE** stream receives epoch-end events with new data points.
3. Chart.js charts are updated by appending new data points without full re-render.

---

## 5. Metrics Summary Table

After training completes, the right column shows a summary:

| Metric | Value |
|--------|-------|
| Train Loss | 0.042 |
| Val Loss | 0.118 |
| Val Accuracy | 95.6% |
| Val F1 | 95.1% |
| Best Epoch | 38 / 50 |
| Training Time | 40m 12s |
| Epochs (actual) | 50 |

---

## 6. Related Documents

- Callbacks & JSON logger → [../05-training/04-callbacks-logging.md](../05-training/04-callbacks-logging.md)
- Run management → [00-run-management.md](00-run-management.md)
- Checkpoints → [02-checkpoints.md](02-checkpoints.md)
