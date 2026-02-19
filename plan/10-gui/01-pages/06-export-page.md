# GUI — Export Page

**Route**: `/projects/{id}/export`

---

## 1. Purpose

Export trained models to deployable formats (ONNX initially). Configure export options, validate the exported model, and download.

---

## 2. Layout

```
┌────────────────────────────────────────────────────────────────┐
│  AI Studio     │  Project: "Cats vs Dogs"    │  Classification │
├────────────────────────────────────────────────────────────────┤
│  [Proj] [Data] [Split] [Train] [Eval] [Export]                 │
├────────────────────────────────────────────────────────────────┤
│  Export                               [+ New Export]           │
│                                                                │
│  ┌──────────────┬───────────────────────────────────────────┐  │
│  │ Export List  │       Export Detail                       │  │
│  │              │                                           │  │
│  │ ┌──────────┐ │  Source:                                  │  │
│  │ │►ONNX v1 │ │  Experiment: ResNet50 Baseline             │  │
│  │ │ ✓ 98MB   │ │  Run: run-e5f6 │ Checkpoint: best        │  │
│  │ └──────────┘ │                                           │  │
│  │              │  Format: ONNX                             │  │
│  │ ┌──────────┐ │  Opset: 17                                │  │
│  │ │ ONNX v2 │ │  Input: 1×3×224×224                        │  │
│  │ │ ✓ 98MB   │ │  Dynamic Batch: ✓                        │  │
│  │ └──────────┘ │  Simplified: ✓                            │  │
│  │              │                                           │  │
│  │              │  ── Validation ──                         │  │
│  │              │  Status: ✓ Passed                         │  │
│  │              │  Max Diff: 1.2×10⁻⁶                       │  │
│  │              │                                           │  │
│  │              │  ── File Info ──                          │  │
│  │              │  Size: 98.5 MB                            │  │
│  │              │  Created: 2 hours ago                     │  │
│  │              │                                           │  │
│  │              │  [⬇ Download]  [Delete]                   │  │
│  └──────────────┴───────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

---

## 3. New Export Modal

```
┌──────────────────────────────────────┐
│  New Export                          │
├──────────────────────────────────────┤
│                                      │
│  ── Source Model ──                  │
│  Experiment: [ResNet50 Baseline ▼]   │
│  Run:        [run-e5f6g7h8 ▼]        │
│  Checkpoint: [best ▼]                │
│                                      │
│  ── Export Format ──                 │
│  ┌──────────┐ ┌───────────┐          │
│  │  ONNX    │ │TorchScript│          │
│  │ (active) │ │(coming    │          │
│  │          │ │ soon)     │          │
│  └──────────┘ └───────────┘          │
│  ┌──────────┐ ┌───────────┐          │
│  │ TensorRT │ │ OpenVINO  │          │
│  │(coming   │ │(coming    │          │
│  │ soon)    │ │ soon)     │          │
│  └──────────┘ └───────────┘          │
│                                      │
│  ── ONNX Options ──                  │
│  Opset Version: [17 ▼]               │
│  Input Height:  [224___]             │
│  Input Width:   [224___]             │
│  Dynamic Batch: [✓]                  │
│  Simplify:      [✓]                  │
│                                      │
│             [Cancel]  [Export]       │
└──────────────────────────────────────┘
```

- Format cards: only ONNX is clickable; others show "Coming Soon" badge.
- Input dimensions pre-populated from the experiment's Resize transform.
- "Export" triggers background export.

---

## 4. Export List (Left Panel)

Each export card:

| Element | Content |
|---------|---------|
| Name | Format + version (e.g., "ONNX v1") |
| Status | ● Running, ✓ Completed, ✗ Failed |
| File size | e.g., "98.5 MB" |
| Source | Experiment name + run |
| Date | Creation date |

---

## 5. Export Detail (Right Panel)

### Completed Export

Shows:
- **Source**: experiment, run, checkpoint used.
- **Format & options**: all configured export options.
- **Validation**: pass/fail + max numerical difference.
- **File info**: size, creation date.
- **Download button**: triggers file download.
- **Delete button**: remove the export.

### Running Export

```
┌─────────────────────────────┐
│  Exporting...               │
│  ████████████████░░░░ 80%   │
│  Step: Validating export    │
└─────────────────────────────┘
```

Steps: Loading model → Exporting → Simplifying → Validating → Done.

### Failed Export

```
┌─────────────────────────────┐
│  Export Failed              │
│  Error: ONNX export failed  │
│  for operator "RotatedNMS"  │
│                             │
│  [Show Details]  [Retry]    │
└─────────────────────────────┘
```

---

## 6. Quick Export from Training Page

The Training page's "Export" quick action button:
1. Navigates to Export page.
2. Pre-fills experiment and run from the training page.
3. Opens the New Export modal.

---

## 7. Related Documents

- Export pipeline → [../../08-export/00-export-overview.md](../../08-export/00-export-overview.md)
- ONNX details → [../../08-export/01-onnx.md](../../08-export/01-onnx.md)
- Export API → [../../09-api/01-endpoints.md](../../09-api/01-endpoints.md#6-export)
