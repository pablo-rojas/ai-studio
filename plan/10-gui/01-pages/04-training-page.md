# GUI — Training Page

**Route**: `/projects/{id}/training`

---

## 1. Purpose

Configure experiments, set hyperparameters and augmentations, launch training, and view live results. This is the most complex page, using a **3-column layout**.

---

## 2. Layout — 3-Column Design

```
┌───────────────────────────────────────────────────────────────────┐
│  AI Studio     │  Project: "Cats vs Dogs"    │  Classification    │
├───────────────────────────────────────────────────────────────────┤
│  [Proj] [Data] [Split] [Train] [Eval] [Export]                    │
├──────────────┬─────────────────────┬──────────────────────────────┤
│  Experiments │   Configuration     │     Results                  │
│  (Left Col)  │   (Center Col)      │     (Right Col)              │
│              │                     │                              │
│ [+ New]      │  ── Model ──        │  Status: ✓ Completed           │
│              │  Backbone: [Res50▼] │                              │
│ ┌──────────┐ │  Pretrained: [✓]    │                              │
│ │►Baseline │ │                     │  ── Metrics ──               │
│ │ ✓  95.6% │ │  ── Optimizer ──    │  Accuracy: 95.6%             │
│ └──────────┘ │  Type: [AdamW ▼]    │  F1: 95.1%                   │
│ ┌──────────┐ │  LR: [0.001____]    │  Loss: 0.118                 │
│ │ EfficNet │ │  Weight Decay:      │                              │
│ │ ○  ---   │ │     [0.01______]    │ ┌────────────────────────┐   │
│ └──────────┘ │                     │ │  Loss Curves           │   │
│              │  ── Scheduler ──    │ │  ╱‾‾‾‾‾╲___            │   │
│              │  Type: [Cosine ▼]   │ │ ╱           ──         │   │
│              │                     │ └────────────────────────┘   │
│              │  ── Training ──     │                              │
│              │  Epochs: [50___]    │ ┌────────────────────────┐   │
│              │  Batch: [32____]    │ │ Accuracy Curve         │   │
│              │  Workers: [4___]    │ │     ╱‾‾‾‾‾‾──          │   │
│              │                     │ │   ╱                    │   │
│              │  ── Augmentations ──│ └────────────────────────┘   │
│              │  [Resize 224×224]   │                              │
│              │  [RandomHFlip 0.5]  │  [Evaluate]                  │
│              │  [ColorJitter ...]  │  [Export]                    │
│              │  [+ Add Transform]  │                              │
│              │                     │                              │
│              │  ── Hardware ──     │                              │
│              │  Device: [☑GPU 0 ▼] │                              │
│              │                     │                              │
│              │  [Save] [Train ▶]   │                              │
└──────────────┴─────────────────────┴──────────────────────────────┘
```

---

## 3. Left Column — Experiment List

**Width**: ~220px fixed.

### Content

- **"+ New Experiment"** button at top.
- List of experiment cards (scrollable).

### Experiment Card

```
┌─────────────────────┐
│ ► ResNet50 Baseline │
│ ✓ 95.6%             │
└─────────────────────┘
```

| Element | Description |
|---------|-------------|
| Arrow ► | Selected indicator |
| Name | Editable name |
| Status icon | ● Training, ✓ Completed, ✗ Failed, ○ Not trained |
| Best metric | Best metric value (if completed) |

### Actions

- **Click**: Select experiment → populate center + right columns.
- **Double-click name**: Rename inline.
- **Right-click menu**: Rename, Duplicate, Delete.
- **Delete**: Confirm dialog. Warns if has evaluations/exports.

---

## 4. Center Column — Configuration

**Width**: ~40% of remaining space.

Scrollable form with sections:

### 4.1 Model Section

| Control | Type | Description |
|---------|------|-------------|
| Architecture | Dropdown | Task-specific options (ResNet-18/34/50, EfficientNet, etc.) |
| Pretrained | Checkbox | Use ImageNet pretrained weights |
| Freeze backbone | Checkbox | Freeze backbone, only train head |

### 4.2 Split Section

| Control | Type | Description |
|---------|------|-------------|
| Split | Dropdown | List of available splits |

### 4.3 Optimizer Section

| Control | Type | Description |
|---------|------|-------------|
| Optimizer | Dropdown | Adam, AdamW, SGD |
| Learning Rate | Number input | Default from task defaults |
| Weight Decay | Number input | |
| Momentum | Number input | Only visible for SGD |

### 4.4 Scheduler Section

| Control | Type | Description |
|---------|------|-------------|
| Scheduler | Dropdown | Cosine, StepLR, MultiStepLR, PolynomialLR, None |
| Warmup Epochs | Number input | |
| Scheduler-specific params | Dynamic | Show/hide based on scheduler type |

### 4.5 Loss Section

| Control | Type | Description |
|---------|------|-------------|
| Loss Function | Dropdown | Task-specific options |
| Loss-specific params | Dynamic | e.g., label_smoothing for CE |

### 4.6 Training Section

| Control | Type | Description |
|---------|------|-------------|
| Epochs | Number input | |
| Batch Size | Number input | |
| Batch Multiplier | Number input | 1–64. Accumulates gradients over N batches to simulate a larger effective batch size without extra GPU memory. |
| Num Workers | Number input | DataLoader workers |
| Early Stopping | Checkbox | Enable/disable |
| Patience | Number input | Visible when early stopping enabled |

### 4.7 Augmentation Section

A live-editable list of transforms:

```
┌─────────────────────────────────┐
│  Augmentations (Train)          │
├─────────────────────────────────┤
│  ≡ Resize        224 × 224      │
│  ≡ RandomHFlip   p=0.5          │
│  ≡ ColorJitter   b=0.2 c=0.2    │
│  ≡ Normalize     ImageNet       │
│                                 │
│  [+ Add Transform]              │
└─────────────────────────────────┘
```

- **Drag handles** (≡) for reordering.
- **Click** on a transform to expand its parameters.
- **Delete** button (×) to remove.
- **"+ Add Transform"** opens a dropdown of available transforms.
- **"Reset to Defaults"** button restores task-specific defaults.

### 4.8 Hardware Section

| Control | Type | Description |
|---------|------|-------------|
| Device | Multi-select dropdown with checkboxes | Lists all detected devices: CPU, GPU 0, GPU 1, … Each entry shows device name + memory. Default: GPU 0 selected if available, otherwise CPU. If both GPU(s) and CPU are selected, CPU is silently ignored. If exactly one GPU is selected, trains on that GPU. If 2+ GPUs are selected, DDP multi-GPU is used automatically. |
| Precision | Dropdown | 32, 16-mixed, bf16-mixed |

**Effective batch size note**: The GUI always shows a read-only label: *"Effective batch size: N (batch_size × batch_multiplier × num_gpus)"*. When `batch_multiplier` is 1 and only one GPU is selected, the label simply mirrors the batch size.

### 4.9 Action Buttons

- **Save**: Persist config to `experiment.json`. Only enabled while status is `"created"`.
- **Train ▶**: Save + start training. Configuration becomes read-only once training begins.

---

## 5. Right Column — Results

**Width**: ~40% of remaining space.

### 5.1 Not Trained State

```
┌─────────────────────┐
│                     │
│  Not yet trained.   │
│  Click "Train" to   │
│  start training.    │
│                     │
└─────────────────────┘
```

### 5.2 Running State

```
┌─────────────────────────────┐
│  Status: ● Training          │
│  Epoch: 23/50               │
│  ████████████░░░░░░░ 46%    │
│  ETA: 12m 30s               │
│                             │
│  ┌───────────────────────┐  │
│  │    Loss Curves        │  │
│  │  (live updating)      │  │
│  └───────────────────────┘  │
│                             │
│  [Stop Training]            │
└─────────────────────────────┘
```

- Progress bar with epoch count.
- Live-updating loss chart (via SSE or polling).
- "Stop Training" button cancels training.

### 5.3 Completed State

```
┌─────────────────────────────┐
│  Status: ✓ Completed         │
│  Duration: 40m 12s          │
│                             │
│  ── Metrics ──              │
│  Accuracy:  95.6%           │
│  F1:        95.1%           │
│  Precision: 94.8%           │
│  Recall:    95.3%           │
│  Best Epoch: 38/50          │
│                             │
│  ┌───────────────────────┐  │
│  │    Loss Curves        │  │
│  └───────────────────────┘  │
│                             │
│  ┌───────────────────────┐  │
│  │  Accuracy / F1 Curves │  │
│  └───────────────────────┘  │
│                             │
│  ┌───────────────────────┐  │
│  │   LR Schedule Curve   │  │
│  └───────────────────────┘  │
│                             │
│  [Evaluate] [Export] [Restart]│
└─────────────────────────────┘
```

- **Metric summary table** with all recorded metrics.
- **Three charts**: Loss curves, metric curves, LR schedule.
- **Restart button**: Deletes results and allows reconfiguration.
- **Quick action buttons**: Jump to Evaluation/Export pages with the experiment pre-selected.

### 5.4 Failed State

```
┌─────────────────────────────┐
│  Status: ✗ Failed            │
│                             │
│  Error: CUDA out of memory  │
│  [Show full traceback ▼]    │
│                             │
│  [Resume] [Restart]          │
└─────────────────────────────┘
```

---

## 6. HTMX Interactions

| Action | Trigger | Request | Target |
|--------|---------|---------|--------|
| Select experiment | Click card | `GET .../experiments/{id}` | `#config-panel`, `#results-panel` |
| Save config | Click "Save" | `PATCH .../experiments/{id}` | Toast notification |
| Start training | Click "Train" | `POST .../experiments/{id}/train` | `#results-panel` |
| Stop training | Click "Stop" | `POST .../experiments/{id}/stop` | `#results-panel` |
| Add transform | Click "Add" | Alpine.js local state | Re-render augmentation list |

---

## 7. Related Documents

- Experiment management → [../../06-experiment-tracking/00-run-management.md](../../06-experiment-tracking/00-run-management.md)
- Hyperparameters → [../../05-training/01-hyperparameters.md](../../05-training/01-hyperparameters.md)
- Augmentations → [../../05-training/02-augmentation.md](../../05-training/02-augmentation.md)
- Training API → [../../09-api/01-endpoints.md](../../09-api/01-endpoints.md#4-training)
