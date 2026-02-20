const THEME_STORAGE_KEY = "ai-studio-theme";

function getStoredTheme() {
  try {
    const storedTheme = localStorage.getItem(THEME_STORAGE_KEY);
    if (storedTheme === "light" || storedTheme === "dark") {
      return storedTheme;
    }
  } catch (_error) {
    return null;
  }

  return null;
}

function getSystemTheme() {
  return window.matchMedia("(prefers-color-scheme: dark)").matches
    ? "dark"
    : "light";
}

function setStoredTheme(theme) {
  try {
    localStorage.setItem(THEME_STORAGE_KEY, theme);
  } catch (_error) {
    // Ignore storage errors and keep theme only for the current session.
  }
}

function applyTheme(theme) {
  const isDark = theme === "dark";
  document.documentElement.classList.toggle("dark", isDark);

  const label = document.getElementById("theme-toggle-label");
  if (label) {
    label.textContent = isDark ? "Light mode" : "Dark mode";
  }

  const button = document.getElementById("theme-toggle");
  if (button) {
    button.setAttribute(
      "aria-label",
      isDark ? "Switch to light mode" : "Switch to dark mode",
    );
  }
}

function initializeThemeToggle() {
  const button = document.getElementById("theme-toggle");
  if (!button) {
    return;
  }

  const initialTheme = getStoredTheme() ?? getSystemTheme();
  applyTheme(initialTheme);

  button.addEventListener("click", () => {
    const nextTheme = document.documentElement.classList.contains("dark")
      ? "light"
      : "dark";
    applyTheme(nextTheme);
    setStoredTheme(nextTheme);
  });
}

document.addEventListener("DOMContentLoaded", initializeThemeToggle);

document.addEventListener("htmx:responseError", (event) => {
  const target = document.getElementById("toast-container");
  if (!target) {
    return;
  }

  const toast = document.createElement("div");
  toast.className =
    "pointer-events-auto mb-2 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700 shadow-sm dark:border-red-900 dark:bg-red-900/30 dark:text-red-200";
  toast.textContent = `Request failed (${event.detail.xhr.status}).`;
  target.appendChild(toast);
  setTimeout(() => toast.remove(), 5000);
});

let trainingEventSource = null;

function closeTrainingEventSource() {
  if (trainingEventSource !== null) {
    trainingEventSource.close();
    trainingEventSource = null;
  }
}

function findSeriesValue(epochPayload, keyFilters) {
  const keys = Object.keys(epochPayload);
  const candidate = keys.find((key) =>
    keyFilters.every((token) => key.toLowerCase().includes(token)),
  );
  if (!candidate) {
    return null;
  }
  const raw = epochPayload[candidate];
  if (typeof raw !== "number") {
    return null;
  }
  return raw;
}

function formatLossValue(value) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "—";
  }
  return value.toFixed(4);
}

function parseTrainingMetrics(workspace) {
  const metricsNode = workspace.querySelector("script[data-training-metrics]");
  if (!metricsNode) {
    return { epochs: [] };
  }

  try {
    const parsed = JSON.parse(metricsNode.textContent ?? "{}");
    if (!parsed || !Array.isArray(parsed.epochs)) {
      return { epochs: [] };
    }
    return parsed;
  } catch (_error) {
    return { epochs: [] };
  }
}

const SVG_NS = "http://www.w3.org/2000/svg";
const CHART_FRAME = {
  xStart: 64,
  xEnd: 620,
  yStart: 24,
  yEnd: 240,
};

function parseEpoch(epochPayload, fallbackEpoch) {
  if (!epochPayload || typeof epochPayload !== "object") {
    return fallbackEpoch;
  }
  const raw = epochPayload.epoch;
  const parsed =
    typeof raw === "number" ? raw : Number.parseInt(String(raw ?? fallbackEpoch), 10);
  if (!Number.isFinite(parsed)) {
    return fallbackEpoch;
  }
  return Math.max(1, Math.trunc(parsed));
}

function computeEpochDomain(epochs) {
  if (epochs.length === 0) {
    return { min: 1, max: 1 };
  }
  const maxEpoch = epochs.reduce((maxValue, epochPayload, index) => {
    const epoch = parseEpoch(epochPayload, index + 1);
    return Math.max(maxValue, epoch);
  }, 1);
  return { min: 1, max: maxEpoch };
}

function computeLossDomain(values) {
  if (values.length === 0) {
    return { min: 0, max: 1 };
  }
  const rawMin = Math.min(...values);
  const rawMax = Math.max(...values);
  if (!Number.isFinite(rawMin) || !Number.isFinite(rawMax)) {
    return { min: 0, max: 1 };
  }
  if (Math.abs(rawMax - rawMin) < 1e-9) {
    const pad = Math.max(Math.abs(rawMin) * 0.1, 0.1);
    return { min: Math.max(0, rawMin - pad), max: rawMax + pad };
  }
  const span = rawMax - rawMin;
  const pad = Math.max(span * 0.08, 0.02);
  return { min: Math.max(0, rawMin - pad), max: rawMax + pad };
}

function mapEpochToX(epoch, xDomain) {
  const xSpan = CHART_FRAME.xEnd - CHART_FRAME.xStart;
  const epochSpan = Math.max(xDomain.max - xDomain.min, 1);
  return CHART_FRAME.xStart + ((epoch - xDomain.min) / epochSpan) * xSpan;
}

function mapValueToY(value, yDomain) {
  const ySpan = CHART_FRAME.yEnd - CHART_FRAME.yStart;
  const domainSpan = Math.max(yDomain.max - yDomain.min, 1e-9);
  return CHART_FRAME.yEnd - ((value - yDomain.min) / domainSpan) * ySpan;
}

function buildPolylinePoints(points, xDomain, yDomain) {
  if (points.length === 0) {
    return "";
  }

  return [...points]
    .sort((left, right) => left.epoch - right.epoch)
    .map((point) => {
      const x = mapEpochToX(point.epoch, xDomain);
      const y = mapValueToY(point.value, yDomain);
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
}

function clearSvgNode(node) {
  while (node.firstChild) {
    node.removeChild(node.firstChild);
  }
}

function appendSvgLine(parent, attributes, className) {
  const line = document.createElementNS(SVG_NS, "line");
  line.setAttribute("stroke", "currentColor");
  if (className) {
    line.setAttribute("class", className);
  }
  for (const [name, value] of Object.entries(attributes)) {
    line.setAttribute(name, String(value));
  }
  parent.appendChild(line);
}

function appendSvgText(parent, attributes, text, className) {
  const node = document.createElementNS(SVG_NS, "text");
  node.setAttribute("fill", "currentColor");
  if (className) {
    node.setAttribute("class", className);
  }
  for (const [name, value] of Object.entries(attributes)) {
    node.setAttribute(name, String(value));
  }
  node.textContent = text;
  parent.appendChild(node);
}

function buildEpochTicks(maxEpoch) {
  if (maxEpoch <= 1) {
    return [1];
  }
  const maxTicks = 6;
  const step = Math.max(1, Math.ceil((maxEpoch - 1) / (maxTicks - 1)));
  const ticks = [1];
  for (let value = 1 + step; value < maxEpoch; value += step) {
    ticks.push(value);
  }
  if (ticks.at(-1) !== maxEpoch) {
    ticks.push(maxEpoch);
  }
  return ticks;
}

function buildLinearTicks(minValue, maxValue, tickCount) {
  if (!Number.isFinite(minValue) || !Number.isFinite(maxValue)) {
    return [0, 1];
  }
  if (tickCount <= 1 || Math.abs(maxValue - minValue) < 1e-9) {
    return [minValue];
  }
  const step = (maxValue - minValue) / (tickCount - 1);
  return Array.from({ length: tickCount }, (_, index) => minValue + index * step);
}

function formatLossTick(value) {
  const magnitude = Math.abs(value);
  if (magnitude >= 10) {
    return value.toFixed(1);
  }
  if (magnitude >= 1) {
    return value.toFixed(2);
  }
  return value.toFixed(3);
}

function renderChartTicks(
  workspace,
  {
    xTicksSelector,
    yTicksSelector,
    xDomain,
    yDomain,
    yTickCount,
    yFormatter,
  },
) {
  const xTicksNode = workspace.querySelector(xTicksSelector);
  const yTicksNode = workspace.querySelector(yTicksSelector);
  if (!xTicksNode || !yTicksNode) {
    return;
  }

  clearSvgNode(xTicksNode);
  clearSvgNode(yTicksNode);

  for (const tick of buildEpochTicks(xDomain.max)) {
    const x = mapEpochToX(tick, xDomain);
    appendSvgLine(
      xTicksNode,
      {
        x1: x.toFixed(2),
        y1: CHART_FRAME.yEnd,
        x2: x.toFixed(2),
        y2: CHART_FRAME.yEnd + 5,
      },
      "text-slate-400 dark:text-slate-500",
    );
    appendSvgText(
      xTicksNode,
      {
        x: x.toFixed(2),
        y: CHART_FRAME.yEnd + 18,
        "text-anchor": "middle",
      },
      String(tick),
      "text-[10px] text-slate-500 dark:text-slate-400",
    );
  }

  for (const tick of buildLinearTicks(yDomain.min, yDomain.max, yTickCount)) {
    const y = mapValueToY(tick, yDomain);
    appendSvgLine(
      yTicksNode,
      {
        x1: CHART_FRAME.xStart,
        y1: y.toFixed(2),
        x2: CHART_FRAME.xEnd,
        y2: y.toFixed(2),
      },
      "text-slate-200 dark:text-slate-800",
    );
    appendSvgLine(
      yTicksNode,
      {
        x1: CHART_FRAME.xStart - 5,
        y1: y.toFixed(2),
        x2: CHART_FRAME.xStart,
        y2: y.toFixed(2),
      },
      "text-slate-400 dark:text-slate-500",
    );
    appendSvgText(
      yTicksNode,
      {
        x: CHART_FRAME.xStart - 8,
        y: (y + 3).toFixed(2),
        "text-anchor": "end",
      },
      yFormatter(tick),
      "text-[10px] text-slate-500 dark:text-slate-400",
    );
  }
}

function renderTrainingLossChart(workspace, epochs) {
  const trainPoints = [];
  const valPoints = [];
  const allLossValues = [];

  for (const [index, epochPayload] of epochs.entries()) {
    if (!epochPayload || typeof epochPayload !== "object") {
      continue;
    }
    const epoch = parseEpoch(epochPayload, index + 1);

    const trainLoss = findSeriesValue(epochPayload, ["train", "loss"]);
    const valLoss = findSeriesValue(epochPayload, ["val", "loss"]);
    if (typeof trainLoss === "number") {
      trainPoints.push({ epoch, value: trainLoss });
      allLossValues.push(trainLoss);
    }
    if (typeof valLoss === "number") {
      valPoints.push({ epoch, value: valLoss });
      allLossValues.push(valLoss);
    }
  }

  const xDomain = computeEpochDomain(epochs);
  const yDomain = computeLossDomain(allLossValues);

  const trainPolyline = workspace.querySelector("[data-training-loss-train]");
  const valPolyline = workspace.querySelector("[data-training-loss-val]");
  const emptyState = workspace.querySelector("[data-training-loss-empty]");
  const trainLatest = workspace.querySelector("[data-training-loss-train-latest]");
  const valLatest = workspace.querySelector("[data-training-loss-val-latest]");
  const progressFill = workspace.querySelector("[data-training-progress-fill]");
  const epochLabel = workspace.querySelector("[data-training-epoch-label]");

  renderChartTicks(workspace, {
    xTicksSelector: "[data-training-loss-x-ticks]",
    yTicksSelector: "[data-training-loss-y-ticks]",
    xDomain,
    yDomain,
    yTickCount: 5,
    yFormatter: formatLossTick,
  });

  if (trainPolyline) {
    trainPolyline.setAttribute("points", buildPolylinePoints(trainPoints, xDomain, yDomain));
  }
  if (valPolyline) {
    valPolyline.setAttribute("points", buildPolylinePoints(valPoints, xDomain, yDomain));
  }

  const hasLossValues = trainPoints.length > 0 || valPoints.length > 0;
  if (emptyState) {
    emptyState.classList.toggle("hidden", hasLossValues);
  }

  if (trainLatest) {
    trainLatest.textContent = formatLossValue(trainPoints.at(-1)?.value);
  }
  if (valLatest) {
    valLatest.textContent = formatLossValue(valPoints.at(-1)?.value);
  }

  const maxEpochsRaw = workspace.getAttribute("data-training-max-epochs");
  const maxEpochs = maxEpochsRaw ? Number.parseInt(maxEpochsRaw, 10) : null;
  const currentEpochRaw = epochs.length > 0 ? epochs.at(-1)?.epoch : 0;
  const currentEpoch =
    typeof currentEpochRaw === "number"
      ? currentEpochRaw
      : Number.parseInt(String(currentEpochRaw ?? 0), 10);

  if (
    progressFill &&
    Number.isInteger(currentEpoch) &&
    maxEpochs !== null &&
    Number.isInteger(maxEpochs) &&
    maxEpochs > 0
  ) {
    const progressPct = Math.max(0, Math.min((currentEpoch / maxEpochs) * 100, 100));
    progressFill.style.width = `${progressPct.toFixed(1)}%`;
  }
  if (epochLabel && maxEpochs !== null && Number.isInteger(maxEpochs) && maxEpochs > 0) {
    epochLabel.textContent = `Epoch ${Math.max(currentEpoch, 0)} / ${maxEpochs}`;
  }
}

function renderTrainingMetricChart(workspace, epochs) {
  const trainF1Points = [];
  const valF1Points = [];

  for (const [index, epochPayload] of epochs.entries()) {
    if (!epochPayload || typeof epochPayload !== "object") {
      continue;
    }
    const epoch = parseEpoch(epochPayload, index + 1);

    const trainF1 = findSeriesValue(epochPayload, ["train", "f1"]);
    const valF1 = findSeriesValue(epochPayload, ["val", "f1"]);

    if (typeof trainF1 === "number") {
      trainF1Points.push({ epoch, value: trainF1 });
    }
    if (typeof valF1 === "number") {
      valF1Points.push({ epoch, value: valF1 });
    }
  }

  const xDomain = computeEpochDomain(epochs);
  const yDomain = { min: 0, max: 1 };

  const trainPolyline = workspace.querySelector("[data-training-metric-train]");
  const valPolyline = workspace.querySelector("[data-training-metric-val]");
  const emptyState = workspace.querySelector("[data-training-metric-empty]");
  const trainLatest = workspace.querySelector("[data-training-metric-train-latest]");
  const valLatest = workspace.querySelector("[data-training-metric-val-latest]");

  renderChartTicks(workspace, {
    xTicksSelector: "[data-training-metric-x-ticks]",
    yTicksSelector: "[data-training-metric-y-ticks]",
    xDomain,
    yDomain,
    yTickCount: 5,
    yFormatter: (value) => value.toFixed(2),
  });

  if (trainPolyline) {
    trainPolyline.setAttribute(
      "points",
      buildPolylinePoints(trainF1Points, xDomain, yDomain),
    );
  }
  if (valPolyline) {
    valPolyline.setAttribute("points", buildPolylinePoints(valF1Points, xDomain, yDomain));
  }

  const hasMetricValues = trainF1Points.length > 0 || valF1Points.length > 0;
  if (emptyState) {
    emptyState.classList.toggle("hidden", hasMetricValues);
  }

  if (trainLatest) {
    trainLatest.textContent = formatLossValue(trainF1Points.at(-1)?.value);
  }
  if (valLatest) {
    valLatest.textContent = formatLossValue(valF1Points.at(-1)?.value);
  }
}

function refreshTrainingExperimentList(projectId, experimentId) {
  if (!window.htmx) {
    return;
  }
  const queryString = new URLSearchParams({
    selected_experiment_id: experimentId,
  });
  window.htmx.ajax(
    "GET",
    `/api/training/${projectId}/experiments?${queryString.toString()}`,
    {
      target: "#training-experiment-list",
      swap: "outerHTML",
      headers: { "HX-Request": "true" },
    },
  );
}

function refreshTrainingWorkspace(projectId, experimentId) {
  if (!window.htmx) {
    return;
  }
  window.htmx.ajax("GET", `/api/training/${projectId}/experiments/${experimentId}`, {
    target: "#training-experiment-workspace",
    swap: "outerHTML",
    headers: { "HX-Request": "true" },
  });
}

function normalizeTrainingStatus(status) {
  const normalized = String(status ?? "").toLowerCase();
  const classMap = {
    created: "bg-slate-100 text-slate-800 dark:bg-slate-700 dark:text-slate-200",
    pending: "bg-sky-100 text-sky-800 dark:bg-sky-500/20 dark:text-sky-200",
    training: "bg-sky-100 text-sky-800 dark:bg-sky-500/20 dark:text-sky-200",
    completed: "bg-emerald-100 text-emerald-800 dark:bg-emerald-500/20 dark:text-emerald-200",
    failed: "bg-rose-100 text-rose-800 dark:bg-rose-500/20 dark:text-rose-200",
    cancelled: "bg-amber-100 text-amber-800 dark:bg-amber-500/20 dark:text-amber-200",
  };
  return {
    label: normalized || "created",
    classes: classMap[normalized] ?? classMap.created,
  };
}

function updateTrainingStatusLabel(workspace, status) {
  const statusNode = workspace.querySelector("[data-training-status-label]");
  if (!statusNode) {
    return;
  }

  const normalized = normalizeTrainingStatus(status);
  statusNode.textContent = normalized.label;
  statusNode.className = `rounded-full px-2 py-1 text-xs font-medium uppercase tracking-wide ${normalized.classes}`;
}

function initializeTrainingWorkspace() {
  closeTrainingEventSource();

  const workspace = document.getElementById("training-experiment-workspace");
  if (!workspace) {
    return;
  }

  const experimentId = workspace.getAttribute("data-training-experiment-id");
  const projectId = workspace.getAttribute("data-training-project-id");
  const streamUrl = workspace.getAttribute("data-training-stream-url");
  const status = workspace.getAttribute("data-training-status");

  if (!experimentId || !projectId || !streamUrl) {
    return;
  }

  const metrics = parseTrainingMetrics(workspace);
  const epochs = Array.isArray(metrics.epochs) ? [...metrics.epochs] : [];
  renderTrainingLossChart(workspace, epochs);
  renderTrainingMetricChart(workspace, epochs);
  updateTrainingStatusLabel(workspace, status);
  refreshTrainingExperimentList(projectId, experimentId);

  const shouldStream = status === "pending" || status === "training";
  if (!shouldStream) {
    return;
  }

  const source = new EventSource(streamUrl);
  trainingEventSource = source;

  source.addEventListener("status", (event) => {
    try {
      const payload = JSON.parse(event.data);
      updateTrainingStatusLabel(workspace, payload.status);
      refreshTrainingExperimentList(projectId, experimentId);
    } catch (_error) {
      // Ignore malformed payload and keep stream open.
    }
  });

  source.addEventListener("epoch_end", (event) => {
    try {
      const payload = JSON.parse(event.data);
      epochs.push(payload);
      renderTrainingLossChart(workspace, epochs);
      renderTrainingMetricChart(workspace, epochs);
    } catch (_error) {
      // Ignore malformed payload and keep stream open.
    }
  });

  source.addEventListener("complete", () => {
    closeTrainingEventSource();
    refreshTrainingWorkspace(projectId, experimentId);
    refreshTrainingExperimentList(projectId, experimentId);
  });
}

document.addEventListener("DOMContentLoaded", initializeTrainingWorkspace);

document.body.addEventListener("htmx:afterSwap", (event) => {
  const target = event.detail?.target;
  if (!(target instanceof HTMLElement)) {
    return;
  }

  if (
    target.id === "training-experiment-workspace" ||
    target.querySelector("#training-experiment-workspace")
  ) {
    initializeTrainingWorkspace();
  }
});

window.addEventListener("beforeunload", closeTrainingEventSource);
