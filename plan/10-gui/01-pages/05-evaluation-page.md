# GUI — Evaluation Page

**Route**: `/projects/{id}/evaluation`

---

## 1. Purpose

Evaluate trained models on test/validation sets, view aggregate metrics, browse per-image predictions, and debug model errors. Evaluation is **1:1 with an experiment** — each completed experiment can have at most one evaluation at a time.

---

## 2. Layout

Two-column grid (`grid-cols-[260px_1fr]`), identical structure to the Training page:

```
┌───────────────────────────────────────────────────────────────┐
│  AI Studio     │  Project: "Cats vs Dogs"    │  Classification │
├───────────────────────────────────────────────────────────────┤
│  [Proj] [Data] [Split] [Train] [Eval] [Export]                │
├───────────────────────────────────────────────────────────────┤
│  Evaluation                                                   │
│                                                               │
│  ┌──────────────┬──────────────────────────────────────────┐  │
│  │  Experiments  │   Right Panel                           │  │
│  │  (completed)  │                                         │  │
│  │               │  ┌─ Hardware & Config ─────────────┐  │  │
│  │ ┌───────────┐ │  │ Checkpoint: [best ▼]              │  │  │
│  │ │►ResNet50  │ │  │ Subsets:  ☑ test ☐ val ☐ train    │  │  │
│  │ │ ✓ 95.6%   │ │  │ Batch:   [32___]                  │  │  │
│  │ └───────────┘ │  │ Device:  [cuda:0 ▼]               │  │  │
│  │               │  │          [Evaluate] [Reset]        │  │  │
│  │ ┌───────────┐ │  └───────────────────────────────────┘  │  │
│  │ │ EfficNet  │ │                                         │  │
│  │ │ ✓ 94.2%   │ │  ┌─ Metrics & Visualizations ──────┐  │  │
│  │ └───────────┘ │  │ Accuracy: 95.6%  F1: 95.1%        │  │  │
│  │               │  │ Precision: 94.8% Recall: 95.3%    │  │  │
│  │               │  │                                    │  │  │
│  │               │  │ ┌──────────────────────────────┐   │  │  │
│  │               │  │ │  Confusion Matrix (heatmap)  │   │  │  │
│  │               │  │ └──────────────────────────────┘   │  │  │
│  │               │  │ ┌──────────────────────────────┐   │  │  │
│  │               │  │ │  Per-class bar chart         │   │  │  │
│  │               │  │ └──────────────────────────────┘   │  │  │
│  │               │  └───────────────────────────────────┘  │  │
│  │               │                                         │  │
│  │               │  ┌─ § Per-Image Results ───── [▼] ───┐  │  │
│  │               │  │ Filter: [All ▼] Sort: [Conf ▼]    │  │  │
│  │               │  │ ┌────┐ ┌────┐ ┌────┐ ┌────┐       │  │  │
│  │               │  │ │img1│ │img2│ │img3│ │img4│       │  │  │
│  │               │  │ │ ✓  │ │ ✗  │ │ ✓  │ │ ✓  │       │  │  │
│  │               │  │ └────┘ └────┘ └────┘ └────┘       │  │  │
│  │               │  │ ◀ 1 2 3 4 5 ▶                     │  │  │
│  │               │  └───────────────────────────────────┘  │  │
│  └───────────────┴──────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────┘
```

---

## 3. Left Panel — Experiment List (Completed Only)

Identical visual style to the Training page's experiment list fragment, but **only shows experiments with `status == "completed"`**. No "create experiment" form — experiments are created on the Training page.

Each experiment card:

| Element | Content |
|---------|---------|
| Name | Experiment name (truncated) |
| Status | Always ✓ Completed (green pill) |
| Best metric | Primary metric value (e.g., `95.6%`) |
| ID | Experiment ID in small text |

Click to select → loads right panel content via HTMX (`hx-get` targeting `#evaluation-workspace`, push URL).

**Empty state**: "No completed experiments yet. Train a model on the Training page."

---

## 4. Right Panel — Two Top Cards + One Collapsible Section

The right panel (`id="evaluation-workspace"`) has:

1. A responsive top row (`data-eval-top-cards`) rendered as two **non-collapsible** cards:
   - Card 1: `data-eval-card="hardware"` (hardware + evaluation controls)
   - Card 2: `data-eval-card="metrics"` (aggregate metrics + visualizations)
2. A bottom **collapsible** per-image results section.

Top row layout classes:

```html
<div data-eval-top-cards class="grid grid-cols-1 gap-4 xl:grid-cols-2">
```

### 4.1 Section 1: Evaluation Configuration (always visible)

Configuration form for setting up and running evaluation:

| Field | Input Type | Description |
|-------|-----------|-------------|
| **Checkpoint** | Dropdown | Populated from existing `.ckpt` files only (scans `checkpoints/` dir). If only `best.ckpt` exists, only "best" appears. |
| **Split subsets** | Multi-select checklist | Shows all subsets from the experiment's split (e.g., ☑ test ☐ val ☐ train). Default: only "test" checked. Multiple can be selected — images are pooled into a combined evaluation. |
| **Batch size** | Number input | Default: 32 |
| **Device** | Dropdown | Available devices from hardware detection (e.g., `cuda:0`, `cpu`) |

**Action buttons**:
- **"Evaluate"** — `hx-post` to `/api/evaluation/{project_id}/{experiment_id}`. Disabled while running. Hidden when evaluation is completed.
- **"Reset"** — `hx-delete` to `/api/evaluation/{project_id}/{experiment_id}`. Immediately deletes evaluation data (no confirmation). Only visible when an evaluation exists (completed or failed). After reset, the config section returns to its editable default state.

**Progress indicator** (visible only while status is `running`):
```
Evaluating: 65/120 images
████████████░░░░░░░ 54%
```

When evaluation is already completed, the config section displays the settings used (read-only) plus the Reset button.

### 4.2 Section 2: Metrics & Visualizations (always visible)

This is a non-collapsible card (`data-eval-card="metrics"`). If no evaluation exists yet, it shows an empty informational state.

Contents:
- **Aggregate metrics table**: Accuracy, F1, Precision, Recall (see [../../07-evaluation/02-aggregate-metrics.md](../../07-evaluation/02-aggregate-metrics.md))
- **Confusion matrix heatmap** (classification): Rendered via Chart.js or HTML table with color intensity. Rows = ground truth, columns = predicted. Diagonal in green, off-diagonal in red.
- **Per-class bar chart** (Chart.js): Horizontal bars for per-class F1/precision/recall, sorted best to worst.
- **Task-specific visualizations**:

| Task | Visualization |
|------|---------------|
| Classification | Confusion matrix heatmap + per-class bar chart |
| Anomaly Detection | ROC curve + threshold analysis |
| Object Detection | Per-class AP bar chart |
| Segmentation | Per-class IoU bar chart |
| Regression | Scatter plot (predicted vs actual) |

### 4.3 Section 3: Per-Image Results Grid (collapsed when no results)

Expanded by default when evaluation results exist. Paginated grid of per-image results loaded via HTMX.

- **Grid/card style** matches the Dataset page visual density (`dataset-image-grid`, `gap-3`, `h-32` image area, compact metadata).
- **Thumbnail status**:
  - ✓ green border + badge = correct prediction.
  - ✗ red border + badge = incorrect prediction.
- **List-level metadata only** in card:
  - Filename
  - Correct/Incorrect badge
  - Subset badge
- Detailed fields (ground truth, prediction, confidence, probabilities) are hidden from the list and shown in the detail view only.
- **Filters**:
  - All / Correct / Incorrect.
  - By class (ground truth or predicted).
  - By subset (test, val, train) — useful when multiple subsets were evaluated together.
- **Sort**:
  - By confidence (ascending → see least confident).
  - By error magnitude (for regression).
  - By filename.
- **Pagination**: 50 per page, HTMX-loaded pages.

### Per-Image Detail View

Clicking an image in the results grid opens a **modal** detail view:

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
│  Subset:       test                  │
│                                      │
│  Class Probabilities:                │
│  cat  ████████████████████░ 94%      │
│  dog  ██░░░░░░░░░░░░░░░░░░  4%      │
│  bird █░░░░░░░░░░░░░░░░░░░  2%      │
│                                      │
│       [◀ Previous] [Next ▶]         │
└──────────────────────────────────────┘
```

The modal combines evaluation metadata + dataset metadata:
- Evaluation: ground truth, prediction, confidence, subset, result.
- Dataset: image dimensions, class label (if available), annotation count, split assignments.
- Class probabilities: progress bars (when available) or empty-state text.

Navigation:
- **Previous/Next** moves within the **currently loaded page** of filtered results.
- If current item is first/last in the page, corresponding nav button is disabled.

For detection tasks: modal shows predicted boxes (solid, colored) overlaid with ground truth boxes (dashed).

---

## 5. HTMX / Alpine.js Interaction Patterns

### Page Load
1. Page route loads project, dataset, list of completed experiments, and optionally a selected experiment (from `?experiment_id=` query param).
2. Left panel renders experiment list fragment.
3. Right panel renders either the detail fragment (if experiment selected) or the empty state.

### Select Experiment
```html
<button hx-get="/api/evaluation/{project_id}/{experiment_id}"
        hx-target="#evaluation-workspace"
        hx-swap="outerHTML"
        hx-push-url="/projects/{project_id}/evaluation?experiment_id={experiment_id}">
```

### Start Evaluation
```html
<form hx-post="/api/evaluation/{project_id}/{experiment_id}"
      hx-target="#evaluation-workspace"
      hx-swap="outerHTML">
```

### Reset Evaluation
```html
<button hx-delete="/api/evaluation/{project_id}/{experiment_id}"
        hx-target="#evaluation-workspace"
        hx-swap="outerHTML">
```

### Per-Image Collapsible Section (Alpine.js)
```html
<div x-data="{ open: true }">
  <button @click="open = !open" class="...">
    <span>Per-Image Results</span>
    <svg :class="{ 'rotate-180': open }" ...>▼</svg>
  </button>
  <div x-show="open" x-collapse>
    <!-- section content -->
  </div>
</div>
```

### Per-Image Detail Modal
```html
<button
  @click="detailOpen = true"
  hx-get="/api/evaluation/{project_id}/{experiment_id}/results/{filename}/info"
  hx-target="#evaluation-result-detail"
  hx-swap="innerHTML"
  hx-include="#evaluation-result-detail-query">
```

### Per-Image Grid Pagination
```html
<div hx-get="/api/evaluation/{project_id}/{experiment_id}/results?page=2&filter_correct=true"
     hx-target="#per-image-grid"
     hx-swap="innerHTML">
```

---

## 6. Template Files

| Template | Purpose |
|----------|---------|
| `pages/evaluation.html` | Full page (extends `base.html`), 2-column grid |
| `fragments/evaluation_experiment_list.html` | Left panel: completed experiments list |
| `fragments/evaluation_detail.html` | Right panel: top non-collapsible cards + collapsible per-image section |
| `fragments/evaluation_empty.html` | Right panel empty state |
| `fragments/evaluation_results_grid.html` | Per-image thumbnail grid (HTMX-loaded) |
| `fragments/evaluation_result_detail.html` | Modal body for per-image detail (HTMX-loaded) |

---

## 7. Related Documents

- Evaluation pipeline → [../../07-evaluation/00-evaluation-pipeline.md](../../07-evaluation/00-evaluation-pipeline.md)
- Per-image results → [../../07-evaluation/01-per-image-results.md](../../07-evaluation/01-per-image-results.md)
- Aggregate metrics → [../../07-evaluation/02-aggregate-metrics.md](../../07-evaluation/02-aggregate-metrics.md)
- Evaluation API → [../../09-api/01-endpoints.md](../../09-api/01-endpoints.md#5-evaluation)
