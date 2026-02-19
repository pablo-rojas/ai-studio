# GUI — Split Page

**Route**: `/projects/{id}/split`

---

## 1. Purpose

Create train/val/test data splits. View split statistics and class distribution per subset. Splits are stored inline in `dataset.json` (`split_names` list + per-image `split` lists), so this page reads and writes that file via the splits API.

---

## 2. Layout

```
┌────────────────────────────────────────────────────────────────┐
│  AI Studio     │  Project: "Cats vs Dogs"    │  Classification │
├────────────────────────────────────────────────────────────────┤
│  [Proj] [Data] [Split] [Train] [Eval] [Export]                 │
├────────────────────────────────────────────────────────────────┤
│  Splits                          [+ New Split]                 │
│                                                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ ● 80-10-10  (index 0)                                   │   │
│  │   Seed: 42                                              │   │
│  │   Train: 960  │  Val: 120  │  Test: 120                 │   │
│  │   Created: 2 days ago                                   │   │
│  │                                 [View Details] [Delete] │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ ○ 70-15-15  (index 1)                                   │   │
│  │   Seed: 123                                             │   │
│  │   Train: 840  │  Val: 180  │  Test: 180                 │   │
│  │   Created: 1 day ago                                    │   │
│  │                                 [View Details] [Delete] │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

Each card represents one entry in `split_names`. The index shown is the position in the list (used as `split_index` by experiments).

---

## 3. New Split Modal

```
┌──────────────────────────────────────────────┐
│  Create Split                                │
├──────────────────────────────────────────────┤
│                                              │
│  Name: [80-10-10________________________]    │
│                                              │
│  ── Ratios ──                                │
│  Train: [====●===========] 80%               │
│  Val:   [=●===============] 10%              │
│  Test:  [=●===============] 10%              │
│  (Total: 100%)                               │
│                                              │
│  Seed: [42___]                               │
│                                              │
│  ── Preview ──                               │
│  Train: 960 images                           │
│  Val: 120 images                             │
│  Test: 120 images                            │
│                                              │
│  Class Distribution (train/val/test):        │
│  cat:  ████████░░ 336/42/42                  │
│  dog:  ████████░░ 328/41/41                  │
│  bird: ███████░░░ 296/37/37                  │
│                                              │
│              [Cancel]  [Create]              │
└──────────────────────────────────────────────┘
```

### Ratio Sliders

- Three linked sliders that always sum to 100%.
- Adjusting one slider proportionally adjusts the other two.
- Minimum value per slider: 0% (test can be 0 if not needed).
- Implemented with Alpine.js reactive binding.

### Live Preview

- When the user adjusts ratios, the preview updates via:
  - `hx-get="/api/splits/{project_id}/preview"` with query params (`ratios`, `seed`).
  - Or computed client-side with Alpine.js for immediate feedback.
- Shows image counts and per-class distribution **before** creation.

### On Create

`POST /api/splits/{project_id}` appends the new name to `split_names` and populates each image's `split` list at the new index. The page refreshes the split list via HTMX.

---

## 4. Split Detail View

Clicking "View Details" on a split card opens an expanded view:

```
Split: 80-10-10  (index 0)
Seed: 42

┌──────────────────────────────────────┐
│  Subset │ Images │ % of total        │
├──────────────────────────────────────┤
│  Train  │   960  │ 80.0%             │
│  Val    │   120  │ 10.0%             │
│  Test   │   120  │ 10.0%             │
│  Total  │ 1,200  │ 100.0%            │
└──────────────────────────────────────┘

Class Distribution:
┌───────┬───────┬──────┬──────┐
│ Class │ Train │  Val │ Test │
├───────┼───────┼──────┼──────┤
│ cat   │  336  │  42  │  42  │
│ dog   │  328  │  41  │  41  │
│ bird  │  296  │  37  │  37  │
└───────┴───────┴──────┴──────┘

[Browse Train Images] [Browse Val Images] [Browse Test Images]
```

Clicking "Browse X Images" filters the Dataset page to show only images in that subset (via query param `?split_index=0&subset=train`).

---

## 5. Split Immutability

Splits are **immutable** once created. If the user wants to change ratios:
- They must create a new split.
- The old split can be deleted (unless referenced by experiments).

Deleting a split removes its name from `split_names` and the corresponding entry at that index in every image's `split` list. All higher-index splits shift down, so experiments referencing those higher indices must be updated (see [splits data model](../../02-data-layer/03-splits.md#7-deletion--re-indexing)).

A split referenced by an experiment shows a warning on delete:
> "This split is used by 2 experiments. Deleting it will invalidate those experiments."

---

## 6. Related Documents

- Split data model → [../../02-data-layer/03-splits.md](../../02-data-layer/03-splits.md)
- Split API → [../../09-api/01-endpoints.md](../../09-api/01-endpoints.md#3-splits)
