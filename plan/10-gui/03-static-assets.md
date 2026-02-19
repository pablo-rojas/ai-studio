# GUI — Static Assets

This document describes the static files served by the application.

---

## 1. Directory Layout

```
app/static/
├── css/
│   ├── tailwind.min.css        # Tailwind CSS (CDN fallback: local copy)
│   └── app.css                 # Custom styles (overrides, animations)
├── js/
│   ├── htmx.min.js             # HTMX library (~14 KB gzipped)
│   ├── alpine.min.js           # Alpine.js (~8 KB gzipped)
│   ├── chart.min.js            # Chart.js (~65 KB gzipped)
│   ├── app.js                  # Application JavaScript
│   ├── charts.js               # Chart initialization helpers
│   └── canvas-overlay.js       # Canvas overlay for annotation rendering
├── icons/
│   ├── folder.svg
│   ├── image.svg
│   ├── scissors.svg
│   ├── play.svg
│   ├── chart-bar.svg
│   └── download.svg
└── img/
    └── logo.svg                # AI Studio logo
```

---

## 2. CSS

### Tailwind CSS

Tailwind provides utility classes. Options for inclusion:

| Option | Pros | Cons |
|--------|------|------|
| **CDN** (`<script src="https://cdn.tailwindcss.com">`) | Zero build step, always latest | Requires internet, larger payload |
| **Pre-built CSS** (local file) | Offline, smaller | Needs build step for customization |
| **Tailwind CLI** (build step) | Full customization, purging | Requires Node.js at build time |

**Recommendation**: Start with CDN for development, switch to pre-built CSS for production.

### `app.css`

Custom styles for things difficult to do with utility classes:

```css
/* HTMX loading indicator */
.htmx-indicator { display: none; }
.htmx-request .htmx-indicator { display: inline-block; }
.htmx-request.htmx-indicator { display: inline-block; }

/* Alpine.js cloak (hide until Alpine initializes) */
[x-cloak] { display: none !important; }

/* Drag handle for augmentation list */
.drag-handle { cursor: grab; }
.drag-handle:active { cursor: grabbing; }

/* Confusion matrix heatmap cell */
.cm-cell {
    transition: background-color 0.2s;
}

/* Thumbnail grid */
.thumb-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 1rem;
}

/* Image overlay (for annotation rendering) */
.image-container {
    position: relative;
}
.image-container canvas {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
}
```

---

## 3. JavaScript

### `app.js`

Global application logic:

```javascript
// HTMX configuration
document.body.addEventListener('htmx:responseError', function(event) {
    // Show error toast for failed HTMX requests
    const toast = document.createElement('div');
    toast.innerHTML = `<div class="bg-red-50 border border-red-200 text-red-800 rounded-lg p-4">
        Request failed: ${event.detail.xhr.status}
    </div>`;
    document.getElementById('toast-container').appendChild(toast);
    setTimeout(() => toast.remove(), 5000);
});

// SSE helper for training updates
function connectTrainingSSE(runId, callbacks) {
    const source = new EventSource(`/api/training/runs/${runId}/stream`);
    source.addEventListener('epoch_end', (e) => callbacks.onEpoch(JSON.parse(e.data)));
    source.addEventListener('complete', (e) => { callbacks.onComplete(JSON.parse(e.data)); source.close(); });
    source.addEventListener('error', () => { callbacks.onError(); source.close(); });
    return source;
}
```

### `charts.js`

Chart creation helpers:

```javascript
function createLossChart(canvasId, epochs, trainLoss, valLoss) {
    return new Chart(document.getElementById(canvasId), {
        type: 'line',
        data: {
            labels: epochs,
            datasets: [
                { label: 'Train Loss', data: trainLoss, borderColor: '#3B82F6', fill: false },
                { label: 'Val Loss', data: valLoss, borderColor: '#F97316', fill: false }
            ]
        },
        options: {
            responsive: true,
            plugins: { legend: { position: 'top' } },
            scales: { y: { beginAtZero: false } }
        }
    });
}

function appendDataPoint(chart, label, datasetValues) {
    chart.data.labels.push(label);
    datasetValues.forEach((val, i) => chart.data.datasets[i].data.push(val));
    chart.update('none'); // No animation for live updates
}
```

### `canvas-overlay.js`

Draws annotation overlays (bounding boxes, masks) on images:

```javascript
function drawBoundingBoxes(canvas, boxes, options = {}) {
    const ctx = canvas.getContext('2d');
    const scaleX = canvas.width / options.imageWidth;
    const scaleY = canvas.height / options.imageHeight;
    
    boxes.forEach(box => {
        ctx.strokeStyle = box.color || '#00FF00';
        ctx.lineWidth = 2;
        ctx.strokeRect(
            box.x * scaleX, box.y * scaleY,
            box.w * scaleX, box.h * scaleY
        );
        // Label
        ctx.fillStyle = box.color || '#00FF00';
        ctx.font = '12px sans-serif';
        ctx.fillText(box.label, box.x * scaleX, box.y * scaleY - 4);
    });
}
```

---

## 4. Library Versions

| Library | Version | Size (gzipped) | Source |
|---------|---------|-----------------|--------|
| HTMX | 2.0.x | ~14 KB | https://unpkg.com/htmx.org |
| Alpine.js | 3.14.x | ~8 KB | https://unpkg.com/alpinejs |
| Chart.js | 4.4.x | ~65 KB | https://cdn.jsdelivr.net/npm/chart.js |
| Tailwind CSS | 3.4.x | ~300 KB (CDN) | https://cdn.tailwindcss.com |

Total JS payload: **~87 KB gzipped** (excluding Tailwind CSS).

---

## 5. Serving

```python
# app/main.py
from fastapi.staticfiles import StaticFiles

app.mount("/static", StaticFiles(directory="app/static"), name="static")
```

In templates: `<script src="{{ url_for('static', path='js/htmx.min.js') }}"></script>`

---

## 6. Related Documents

- GUI overview → [../00-gui-overview.md](../00-gui-overview.md)
- Components → [../02-components.md](../02-components.md)
