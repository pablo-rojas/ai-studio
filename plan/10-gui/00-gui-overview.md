# GUI — Overview

This document describes the overall GUI architecture, navigation, theming, and shared patterns.

---

## 1. Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Server templates | **Jinja2** | Render full HTML pages |
| Partial updates | **HTMX** | Swap page fragments without full reload |
| Client logic | **Alpine.js** | Lightweight reactive state (modals, toggles, form binding) |
| Charts | **Chart.js** | Loss curves, metric visualizations |
| CSS | **Tailwind CSS** (via CDN) or custom CSS | Responsive utility-first styling |
| Icons | **Heroicons** or **Lucide** | SVG icon set |

No SPA framework. Every page is a full Jinja2 template; interactivity is added progressively with HTMX and Alpine.js.

---

## 2. Page Map

AI Studio has **6 pages**, all scoped to a project except the Project page:

| # | Page | Route | Description |
|---|------|-------|-------------|
| 1 | Project | `/projects` | CRUD projects, select task |
| 2 | Dataset | `/projects/{id}/dataset` | Import, browse, view dataset |
| 3 | Split | `/projects/{id}/split` | Create and view data splits |
| 4 | Training | `/projects/{id}/training` | Configure experiments, run training |
| 5 | Evaluation | `/projects/{id}/evaluation` | Evaluate models on test set |
| 6 | Export | `/projects/{id}/export` | Export models to ONNX |

---

## 3. Layout Structure

```
┌─────────────────────────────────────────────────────┐
│ Top Bar: AI Studio logo │ Project name │ Task badge │
├─────────────────────────────────────────────────────┤
│ Nav: [Proj] [Data] [Split] [Train] [Eval] [Export]  │
├─────────────────────────────────────────────────────┤
│                                                     │
│                   Page Content                      │
│                                                     │
│                                                     │
│                                                     │
│                                                     │
│                                                     │
│                                                     │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### Top Bar
- **Left**: AI Studio logo + app name.
- **Center**: Current project name (clickable → back to project list).
- **Right**: Task type badge (e.g., "Classification" in a colored pill).

### Navigation Bar (Horizontal)
- **Horizontal bar** below the top bar.
- 6 nav items displayed inline, one per page.
- Highlighting for active page (bottom border or background).
- Icons + short labels, laid out left-to-right.
- On the Project page, only the Project nav item is shown (no dataset/split/etc. until project is selected).

---

## 4. Template Hierarchy

```
app/templates/
├── base.html                  # <html> skeleton, CSS/JS imports, top bar, nav bar
├── pages/
│   ├── project.html           # Project page
│   ├── dataset.html           # Dataset page
│   ├── split.html             # Split page
│   ├── training.html          # Training page
│   ├── evaluation.html        # Evaluation page
│   └── export.html            # Export page
├── fragments/                 # HTMX partial fragments
│   ├── project_list.html
│   ├── project_card.html
│   ├── image_grid.html
│   ├── image_detail.html
│   ├── split_list.html
│   ├── split_preview.html
│   ├── experiment_list.html
│   ├── hparam_form.html
│   ├── training_results.html
│   ├── eval_results.html
│   ├── eval_image_detail.html
│   ├── export_list.html
│   ├── error_toast.html
│   └── ...
└── components/               # Reusable Jinja2 macros
    ├── pagination.html
    ├── modal.html
    ├── navbar.html
    ├── metric_card.html
    ├── chart.html
    └── ...
```

### `base.html` Skeleton

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Studio — {% block title %}{% endblock %}</title>
    <link rel="stylesheet" href="/static/css/tailwind.min.css">
    <link rel="stylesheet" href="/static/css/app.css">
    <script src="/static/js/htmx.min.js"></script>
    <script src="/static/js/alpine.min.js" defer></script>
    <script src="/static/js/chart.min.js"></script>
    {% block head %}{% endblock %}
</head>
<body class="bg-gray-50 text-gray-900">
    <!-- Top Bar -->
    {% include "components/topbar.html" %}
    
    <!-- Horizontal Navigation Bar -->
    {% include "components/navbar.html" %}
    
    <!-- Main Content -->
    <main class="p-6">
        {% block content %}{% endblock %}
    </main>
    
    <!-- Toast container -->
    <div id="toast-container" class="fixed top-4 right-4 z-50"></div>
    
    {% block scripts %}{% endblock %}
</body>
</html>
```

---

## 5. HTMX Patterns

### Partial Page Updates

Most interactions use HTMX to swap a portion of the page:

```html
<!-- Click a project card to load its details -->
<div hx-get="/api/projects/{{ project.id }}" 
     hx-target="#project-detail" 
     hx-swap="innerHTML">
    {{ project.name }}
</div>
```

### Form Submission

```html
<form hx-post="/api/splits/{{ project_id }}" 
      hx-target="#split-list"
      hx-swap="beforeend">
    ...
</form>
```

### Polling

```html
<!-- Poll training status every 3 seconds -->
<div hx-get="/api/training/{{ experiment_id }}/status" 
     hx-trigger="every 3s"
     hx-target="#training-status">
</div>
```

### Loading States

```html
<button hx-post="/api/training/train" 
        hx-indicator="#spinner">
    <span id="spinner" class="htmx-indicator">⏳</span>
    Start Training
</button>
```

---

## 6. Alpine.js Patterns

### Modals

```html
<div x-data="{ open: false }">
    <button @click="open = true">New Project</button>
    <div x-show="open" x-cloak class="modal">
        <!-- modal content -->
        <button @click="open = false">Cancel</button>
    </div>
</div>
```

### Form State

```html
<div x-data="{ ratios: { train: 80, val: 10, test: 10 }, seed: 42 }">
    <!-- ratio sliders -->
</div>
```

---

## 7. Responsive Design

- **Minimum width**: 1024px (desktop app, not mobile-optimized).
- **Nav bar**: Full-width horizontal strip; items wrap if window is narrow.
- **Training page 3-column**: Flex layout with min-width constraints.

---

## 8. Related Documents

- Individual page docs → [01-pages/](01-pages/)
- Shared components → [02-components.md](02-components.md)
- Static assets → [03-static-assets.md](03-static-assets.md)
- API (data source) → [../09-api/00-api-overview.md](../09-api/00-api-overview.md)
