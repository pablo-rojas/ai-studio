# GUI — Project Page

**Route**: `/projects`

---

## 1. Purpose

The Project page is the entry point. Users create, select, and manage projects here. Each project is scoped to **one task type** (classification, anomaly detection, etc.).

---

## 2. Layout

```
┌─────────────────────────────────────────────────────────┐
│  AI Studio            Projects                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐  │
│  │  +   │  │ Project A │  │ Project B │  │ Project C │  │
│  │ New  │  │ Classif.  │  │ Obj. Det. │  │ Anomaly   │  │
│  │      │  │ 1200 imgs │  │ 500 imgs  │  │ 800 imgs  │  │
│  │      │  │ 3 exps    │  │ 1 exp     │  │ 0 exps    │  │
│  └──────┘  └───────────┘  └───────────┘  └───────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Project Card

Each project is displayed as a card:

| Element | Content |
|---------|---------|
| **Name** | Project name (editable inline) |
| **Task badge** | Colored pill: "Classification", "Object Detection", etc. |
| **Dataset summary** | Image count, class count, or "No dataset" |
| **Experiment count** | Number of experiments |
| **Last modified** | Relative date (e.g., "2 hours ago") |

### Card Actions (Hover / Right-click)

- **Open** (click): Navigate to Dataset page.
- **Rename** (inline edit): Double-click name to edit.
- **Duplicate**: Copy project config (not data).
- **Delete**: Confirm dialog → remove project folder.

---

## 4. New Project Modal

Clicking the "+" card opens a modal:

```
┌────────────────────────────────────┐
│  Create New Project                │
├────────────────────────────────────┤
│                                    │
│  Name: [________________________]  │
│                                    │
│  Task:                             │
│  ┌───────────────┐ ┌────────────┐  │
│  │ Classification│ │  Anomaly   │  │
│  │    (active)   │ │ Detection  │  │
│  └───────────────┘ └────────────┘  │
│  ┌───────────────┐ ┌────────────┐  │
│  │   Object      │ │ Oriented   │  │
│  │  Detection    │ │    OD      │  │
│  └───────────────┘ └────────────┘  │
│  ┌───────────────┐ ┌────────────┐  │
│  │ Segmentation  │ │  Instance  │  │
│  │               │ │   Seg.     │  │
│  └───────────────┘ └────────────┘  │
│  ┌──────────────┐                  │
│  │  Regression  │                  │
│  └──────────────┘                  │
│                                    │
│           [Cancel]  [Create]       │
└────────────────────────────────────┘
```

- Task selection is a grid of clickable cards (one selected at a time).
- Tasks not yet available in the current phase are shown as **disabled** with a "Coming Soon" label.
- "Create" button submits `POST /api/projects`.
- On success, navigate to the new project's Dataset page.

---

## 5. Interactions

| Action | Mechanism | Target |
|--------|-----------|--------|
| Load project list | HTMX `GET /api/projects` | `#project-grid` |
| Create project | HTMX `POST /api/projects` (via modal form) | Redirect to dataset page |
| Delete project | HTMX `DELETE /api/projects/{id}` | Remove card from grid |
| Rename project | HTMX `PATCH /api/projects/{id}` | Update card name |

---

## 6. Navigation

After selecting a project, the user is redirected to `/projects/{id}/dataset`. The navigation bar then shows all 6 navigation items.

---

## 7. Related Documents

- Projects API → [../../09-api/01-endpoints.md](../../09-api/01-endpoints.md#1-projects)
- GUI overview → [../00-gui-overview.md](../00-gui-overview.md)
