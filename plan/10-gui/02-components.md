# GUI — Shared Components

This document describes reusable UI components used across multiple pages.

---

## 1. Top Bar

```html
<!-- components/topbar.html -->
<header class="h-14 bg-white border-b flex items-center px-4">
    <a href="/projects" class="font-bold text-lg">AI Studio</a>
    {% if project %}
        <span class="mx-2 text-gray-400">›</span>
        <a href="/projects" class="text-gray-600 hover:text-gray-900">{{ project.name }}</a>
        <span class="ml-3 px-2 py-1 text-xs font-medium rounded-full 
                      bg-blue-100 text-blue-800">{{ project.task | title }}</span>
    {% endif %}
</header>
```

---

## 2. Horizontal Navigation Bar

```html
<!-- components/navbar.html -->
<nav class="bg-white border-b px-4">
    <div class="flex items-center gap-1">
        {% set nav_items = [
            ("projects", "Projects", "folder-icon"),
            ("dataset", "Dataset", "image-icon"),
            ("split", "Split", "scissors-icon"),
            ("training", "Training", "play-icon"),
            ("evaluation", "Evaluation", "chart-icon"),
            ("export", "Export", "download-icon"),
        ] %}
        {% for id, label, icon in nav_items %}
            <a href="{{ url_for_page(id) }}" 
               class="flex items-center gap-2 px-4 py-3 text-sm font-medium border-b-2 
                      {% if active_page == id %}border-blue-600 text-blue-700{% else %}border-transparent text-gray-600 hover:text-gray-900 hover:border-gray-300{% endif %}">
                {% include "icons/" + icon + ".html" %}
                <span>{{ label }}</span>
            </a>
        {% endfor %}
    </div>
</nav>
```

- When on the Project page (no project selected), only "Projects" is shown.
- Active page is highlighted with a blue bottom border.

---

## 3. Pagination

```html
<!-- components/pagination.html -->
<!-- Usage: {% include "components/pagination.html" with page=page, total_pages=total, target="#content" %} -->
<nav class="flex items-center gap-2 mt-4">
    <button hx-get="{{ base_url }}?page={{ page - 1 }}" 
            hx-target="{{ target }}"
            {% if page == 1 %}disabled{% endif %}
            class="px-3 py-1 rounded border">◀</button>
    
    {% for p in range(1, total_pages + 1) %}
        <button hx-get="{{ base_url }}?page={{ p }}" 
                hx-target="{{ target }}"
                class="px-3 py-1 rounded border {% if p == page %}bg-blue-600 text-white{% endif %}">
            {{ p }}
        </button>
    {% endfor %}
    
    <button hx-get="{{ base_url }}?page={{ page + 1 }}" 
            hx-target="{{ target }}"
            {% if page == total_pages %}disabled{% endif %}
            class="px-3 py-1 rounded border">▶</button>
</nav>
```

Truncation logic for large page counts (show first, last, and ±2 around current with `...`).

---

## 4. Modal

```html
<!-- components/modal.html -->
<div x-data="{ open: false }" @open-modal.window="open = true">
    <!-- Trigger -->
    <slot name="trigger"></slot>
    
    <!-- Backdrop + Dialog -->
    <div x-show="open" x-cloak 
         class="fixed inset-0 z-40 bg-black/30 flex items-center justify-center"
         @click.self="open = false">
        <div class="bg-white rounded-lg shadow-xl w-full max-w-md p-6"
             @keydown.escape.window="open = false">
            <slot name="content"></slot>
        </div>
    </div>
</div>
```

Jinja2 macro version:
```jinja2
{% macro modal(id, title) %}
<div x-data="{ open: false }" x-on:open-{{ id }}.window="open = true">
    <div x-show="open" x-cloak class="fixed inset-0 z-40 bg-black/30 flex items-center justify-center">
        <div class="bg-white rounded-lg shadow-xl max-w-md w-full p-6">
            <h2 class="text-lg font-semibold mb-4">{{ title }}</h2>
            {{ caller() }}
        </div>
    </div>
</div>
{% endmacro %}
```

---

## 5. Metric Card

```html
<!-- components/metric_card.html -->
{% macro metric_card(label, value, unit="", color="blue") %}
<div class="bg-white rounded-lg border p-4">
    <div class="text-sm text-gray-500">{{ label }}</div>
    <div class="text-2xl font-bold text-{{ color }}-600">{{ value }}{{ unit }}</div>
</div>
{% endmacro %}
```

Usage: `{{ metric_card("Accuracy", "95.6", "%") }}`

---

## 6. Toast Notifications

```html
<!-- fragments/error_toast.html -->
<div class="bg-red-50 border border-red-200 text-red-800 rounded-lg p-4 mb-2 flex items-start gap-3"
     x-data="{ show: true }" 
     x-show="show"
     x-init="setTimeout(() => show = false, 5000)">
    <span class="text-red-500">⚠</span>
    <div class="flex-1">{{ error_message }}</div>
    <button @click="show = false" class="text-red-400 hover:text-red-600">✕</button>
</div>
```

Toast types:
- **Error** (red): API errors, validation failures.
- **Success** (green): Successful operations (import complete, export done).
- **Info** (blue): Status updates (training started).
- **Warning** (yellow): Warnings (split in use, large dataset).

---

## 7. Chart Container

```html
<!-- components/chart.html -->
{% macro line_chart(id, height="300px") %}
<div class="bg-white rounded-lg border p-4">
    <canvas id="{{ id }}" style="height: {{ height }}"></canvas>
</div>
{% endmacro %}
```

Chart initialization pattern:
```javascript
const ctx = document.getElementById('loss-chart').getContext('2d');
const chart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: epochs,
        datasets: [
            { label: 'Train Loss', data: trainLoss, borderColor: '#3B82F6' },
            { label: 'Val Loss', data: valLoss, borderColor: '#F97316' }
        ]
    },
    options: { responsive: true, maintainAspectRatio: false }
});
```

---

## 8. Status Badge

```html
{% macro status_badge(status) %}
{% set colors = {
    "running": "bg-blue-100 text-blue-800",
    "completed": "bg-green-100 text-green-800",
    "failed": "bg-red-100 text-red-800",
    "pending": "bg-gray-100 text-gray-800"
} %}
<span class="px-2 py-0.5 text-xs font-medium rounded-full {{ colors.get(status, '') }}">
    {{ status | title }}
</span>
{% endmacro %}
```

---

## 9. Empty State

```html
{% macro empty_state(icon, title, description, action_label="", action_url="") %}
<div class="flex flex-col items-center justify-center py-20 text-gray-400">
    <div class="text-5xl mb-4">{{ icon }}</div>
    <div class="text-lg font-medium text-gray-600">{{ title }}</div>
    <div class="text-sm mt-1">{{ description }}</div>
    {% if action_label %}
        <a href="{{ action_url }}" class="mt-4 px-4 py-2 bg-blue-600 text-white rounded-md">
            {{ action_label }}
        </a>
    {% endif %}
</div>
{% endmacro %}
```

---

## 10. Related Documents

- GUI overview → [../00-gui-overview.md](../00-gui-overview.md)
- Static assets → [../03-static-assets.md](../03-static-assets.md)
