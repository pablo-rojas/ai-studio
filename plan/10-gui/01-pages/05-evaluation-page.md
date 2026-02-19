# GUI — Evaluation Page

**Route**: `/projects/{id}/evaluation`

---

## 1. Purpose

Run model evaluation on test sets, view aggregate metrics, browse per-image predictions, and debug model errors.

---

## 2. Layout

```
┌───────────────────────────────────────────────────────────────┐
│  AI Studio     │  Project: "Cats vs Dogs"    │  Classification │
├───────────────────────────────────────────────────────────────┤
│  [Proj] [Data] [Split] [Train] [Eval] [Export]                │
├───────────────────────────────────────────────────────────────┤
│  Evaluation                        [+ New Evaluation]         │
│                                                               │
│  ┌──────────────┬──────────────────────────────────────────┐  │
│  │  Eval List   │        Results                           │  │
│  │              │                                          │  │
│  │ ┌──────────┐ │  ── Aggregate Metrics ──                 │  │
│  │ │►Test eval│ │  Accuracy: 95.6%                         │  │
│  │ │ ✓ 95.6%  │ │  F1: 95.1%                               │  │
│  │ └──────────┘ │  Precision: 94.8%                        │  │
│  │              │  Recall: 95.3%                           │  │
│  │ ┌──────────┐ │                                          │  │
│  │ │ Val eval │ │  ┌───────────────────────────────────┐   │  │
│  │ │ ✓ 94.2%  │ │  │   Confusion Matrix                │   │  │
│  │ └──────────┘ │  │   (heatmap)                       │   │  │
│  │              │  └───────────────────────────────────┘   │  │
│  │              │                                          │  │
│  │              │  ── Per-Image Results ──                 │  │
│  │              │  Filter: [All ▼] Sort: [Conf ▼]          │  │
│  │              │  ┌────┐ ┌────┐ ┌────┐ ┌────┐             │  │
│  │              │  │img1│ │img2│ │img3│ │img4│             │  │
│  │              │  │ ✓  │ │ ✗ │ │ ✓  │ │ ✓  │             │  │
│  │              │  └────┘ └────┘ └────┘ └────┘             │  │
│  │              │  ◀ 1 2 3 4 5 ▶                          │  │
│  └──────────────┴──────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────┘
```

---

## 3. New Evaluation Modal

```
┌──────────────────────────────────────┐
│  New Evaluation                      │
├──────────────────────────────────────┤
│                                      │
│  Experiment: [ResNet50 Baseline ▼]   │
│  Run:        [run-e5f6g7h8 (best)▼]  │
│  Checkpoint: [best ▼]                │
│                                      │
│  Split:      [80-10-10 ▼]             │
│  Subset:     [test ▼]                │
│                                      │
│  Batch Size: [32___]                 │
│  Device:     [cuda:0 ▼]              │
│                                      │
│             [Cancel]  [Evaluate]     │
└──────────────────────────────────────┘
```

- Experiment dropdown → filters available runs.
- Run dropdown → shows each run with its best metric.
- Subset dropdown: test (default), val, train.
- "Evaluate" starts background evaluation.

---

## 4. Evaluation List (Left Panel)

Each evaluation card:

| Element | Content |
|---------|---------|
| Name | Auto-generated or user-given |
| Experiment | Source experiment name |
| Status | ● Running, ✓ Completed, ✗ Failed |
| Key metric | Primary metric value |
| Date | Creation date |

Click to select and show results in the right panel.

---

## 5. Results Panel (Right Side)

### 5.1 Aggregate Metrics Card

- Table of all aggregate metrics (see [../07-evaluation/02-aggregate-metrics.md](../../07-evaluation/02-aggregate-metrics.md)).
- Task-specific visualizations:

| Task | Visualization |
|------|---------------|
| Classification | Confusion matrix heatmap + per-class bar chart |
| Anomaly Detection | ROC curve + threshold analysis |
| Object Detection | Per-class AP bar chart |
| Segmentation | Per-class IoU bar chart |
| Regression | Scatter plot (predicted vs actual) |

### 5.2 Per-Image Results Grid

Below the aggregate metrics, a paginated grid of per-image results:

- **Thumbnail** with overlay:
  - ✓ green border = correct prediction.
  - ✗ red border = incorrect prediction.
  - Confidence score overlay.
- **Filters**:
  - All / Correct / Incorrect.
  - By class (ground truth or predicted).
- **Sort**:
  - By confidence (ascending → see least confident).
  - By error magnitude (for regression).
  - By filename.

### 5.3 Per-Image Detail

Clicking an image in the results grid opens a detail view:

```
┌──────────────────────────────────────┐
│  ◀ img_0042.jpg                  ✕  │
├──────────────────────────────────────┤
│                                      │
│      [Image with prediction overlay] │
│                                      │
├──────────────────────────────────────┤
│  Ground Truth: cat                   │
│  Prediction:   cat  (94% confidence) │
│  Result:       ✓ Correct             │
│                                      │
│  Class Probabilities:                │
│  cat  ████████████████████░ 94%      │
│  dog  ██░░░░░░░░░░░░░░░░░░  4%       │
│  bird █░░░░░░░░░░░░░░░░░░░  2%       │
│                                      │
│       [◀ Previous] [Next ▶]         │
└──────────────────────────────────────┘
```

For detection tasks: shows predicted boxes (solid, colored) overlaid with ground truth boxes (dashed).

---

## 6. Running Evaluation Progress

While evaluation is in progress:

```
┌─────────────────────────────┐
│  Evaluation: eval-abc123    │
│  Status: ● Running          │
│  Progress: 65/120 images    │
│  ████████████░░░░░░░ 54%    │
└─────────────────────────────┘
```

---

## 7. Related Documents

- Evaluation pipeline → [../../07-evaluation/00-evaluation-pipeline.md](../../07-evaluation/00-evaluation-pipeline.md)
- Per-image results → [../../07-evaluation/01-per-image-results.md](../../07-evaluation/01-per-image-results.md)
- Aggregate metrics → [../../07-evaluation/02-aggregate-metrics.md](../../07-evaluation/02-aggregate-metrics.md)
- Evaluation API → [../../09-api/01-endpoints.md](../../09-api/01-endpoints.md#5-evaluation)
