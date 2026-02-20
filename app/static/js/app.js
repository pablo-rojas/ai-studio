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
let trainingChartResizeObserver = null;
let trainingChartResizeRaf = null;
const TRAINING_SECTION_STORAGE_PREFIX = "ai-studio-training-sections";

function closeTrainingEventSource() {
  if (trainingEventSource !== null) {
    trainingEventSource.close();
    trainingEventSource = null;
  }
}

function closeTrainingChartResizeObserver() {
  if (trainingChartResizeObserver !== null) {
    trainingChartResizeObserver.disconnect();
    trainingChartResizeObserver = null;
  }
  if (trainingChartResizeRaf !== null) {
    window.cancelAnimationFrame(trainingChartResizeRaf);
    trainingChartResizeRaf = null;
  }
}

function parseTrainingSectionState(rawState) {
  if (!rawState) {
    return {};
  }
  try {
    const parsed = JSON.parse(rawState);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return {};
    }
    return parsed;
  } catch (_error) {
    return {};
  }
}

function readTrainingSectionState(storageKey) {
  try {
    return parseTrainingSectionState(localStorage.getItem(storageKey));
  } catch (_error) {
    return {};
  }
}

function initializeTrainingSectionState(workspace) {
  const sectionNodes = workspace.querySelectorAll("[data-training-config-section]");
  if (sectionNodes.length === 0) {
    return;
  }

  const projectId = workspace.getAttribute("data-training-project-id");
  if (!projectId) {
    return;
  }

  const storageKey = `${TRAINING_SECTION_STORAGE_PREFIX}:${projectId}`;
  const sectionState = readTrainingSectionState(storageKey);

  sectionNodes.forEach((node) => {
    if (!(node instanceof HTMLDetailsElement)) {
      return;
    }

    const sectionId = node.getAttribute("data-training-config-section");
    if (!sectionId) {
      return;
    }

    const defaultOpen = node.getAttribute("data-default-open") === "true";
    if (typeof sectionState[sectionId] === "boolean") {
      node.open = Boolean(sectionState[sectionId]);
    } else {
      node.open = defaultOpen;
    }

    node.addEventListener("toggle", () => {
      const latestState = readTrainingSectionState(storageKey);
      latestState[sectionId] = node.open;
      try {
        localStorage.setItem(storageKey, JSON.stringify(latestState));
      } catch (_error) {
        // Ignore localStorage failures and keep session-only behavior.
      }
    });
  });
}

function parsePositiveInteger(value, fallback = 1) {
  const parsed = Number.parseInt(String(value ?? ""), 10);
  if (!Number.isFinite(parsed) || parsed < 1) {
    return fallback;
  }
  return parsed;
}

function countSelectedGpus(workspace) {
  const selectedDeviceInputs = workspace.querySelectorAll(
    "input[name='hardware.selected_devices[]'][data-training-device-toggle]:checked",
  );
  let count = 0;
  selectedDeviceInputs.forEach((node) => {
    if (!(node instanceof HTMLInputElement)) {
      return;
    }
    const value = node.value.toLowerCase();
    if (value === "gpu" || value === "cuda" || value.startsWith("gpu:") || value.startsWith("cuda:")) {
      count += 1;
    }
  });
  return count;
}

function updateTrainingEffectiveBatchLabel(workspace) {
  const label = workspace.querySelector("[data-training-effective-batch]");
  if (!(label instanceof HTMLElement)) {
    return;
  }

  const batchSizeInput = workspace.querySelector("[name='hyperparameters.batch_size']");
  const batchMultiplierInput = workspace.querySelector(
    "[name='hyperparameters.batch_multiplier']",
  );
  if (!(batchSizeInput instanceof HTMLInputElement) || !(batchMultiplierInput instanceof HTMLInputElement)) {
    return;
  }

  const batchSize = parsePositiveInteger(batchSizeInput.value, 1);
  const batchMultiplier = parsePositiveInteger(batchMultiplierInput.value, 1);
  const selectedGpuCount = countSelectedGpus(workspace);
  const numGpus = Math.max(selectedGpuCount, 1);
  const effectiveBatchSize = batchSize * batchMultiplier * numGpus;

  label.textContent = `Effective batch size: ${effectiveBatchSize} (${batchSize} × ${batchMultiplier} × ${numGpus})`;
}

function updateTrainingConditionalFields(workspace) {
  const optimizerSelect = workspace.querySelector("[data-training-input='optimizer']");
  const schedulerSelect = workspace.querySelector("[data-training-input='scheduler']");
  const lossSelect = workspace.querySelector("[data-training-input='loss']");

  const momentumField = workspace.querySelector("[data-training-field='momentum']");
  if (momentumField instanceof HTMLElement && optimizerSelect instanceof HTMLSelectElement) {
    momentumField.classList.toggle("hidden", optimizerSelect.value !== "sgd");
  }

  const stepSizeField = workspace.querySelector("[data-training-field='step_size']");
  const gammaField = workspace.querySelector("[data-training-field='gamma']");
  const polyPowerField = workspace.querySelector("[data-training-field='poly_power']");
  if (schedulerSelect instanceof HTMLSelectElement) {
    const scheduler = schedulerSelect.value;
    const showStep = scheduler === "step" || scheduler === "multistep";
    const showGamma = scheduler === "step" || scheduler === "multistep";
    const showPolyPower = scheduler === "poly";

    if (stepSizeField instanceof HTMLElement) {
      stepSizeField.classList.toggle("hidden", !showStep);
    }
    if (gammaField instanceof HTMLElement) {
      gammaField.classList.toggle("hidden", !showGamma);
    }
    if (polyPowerField instanceof HTMLElement) {
      polyPowerField.classList.toggle("hidden", !showPolyPower);
    }
  }

  const labelSmoothingField = workspace.querySelector("[data-training-field='label_smoothing']");
  if (labelSmoothingField instanceof HTMLElement && lossSelect instanceof HTMLSelectElement) {
    const showLabelSmoothing =
      lossSelect.value === "cross_entropy" ||
      lossSelect.value === "label_smoothing_cross_entropy";
    labelSmoothingField.classList.toggle("hidden", !showLabelSmoothing);
  }
}

const TRAINING_AUGMENTATION_EDITABLE_ORDER = [
  "RandomResizedCrop",
  "RandomHorizontalFlip",
  "RandomRotation",
  "ColorJitter",
];
const TRAINING_AUGMENTATION_REQUIRED_ORDER = ["ToImage", "Normalize"];
const TRAINING_AUGMENTATION_DEFAULTS = {
  normalize: {
    mean: [0.485, 0.456, 0.406],
    std: [0.229, 0.224, 0.225],
  },
  randomResizedCrop: {
    size: [224, 224],
    scale: [0.8, 1.0],
  },
};

function isPlainObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function cloneValue(value) {
  if (typeof structuredClone === "function") {
    return structuredClone(value);
  }
  return JSON.parse(JSON.stringify(value));
}

function cloneStep(step) {
  const name = String(step?.name ?? "").trim();
  const params = isPlainObject(step?.params) ? cloneValue(step.params) : {};
  return {
    name,
    params,
  };
}

function parseTrainingAugmentations(workspace) {
  const augmentationsNode = workspace.querySelector("script[data-training-augmentations]");
  if (!augmentationsNode) {
    return { train: [], val: [] };
  }

  try {
    const parsed = JSON.parse(augmentationsNode.textContent ?? "{}");
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return { train: [], val: [] };
    }
    const train = Array.isArray(parsed.train) ? parsed.train : [];
    const val = Array.isArray(parsed.val) ? parsed.val : [];
    return {
      train: train.map(cloneStep).filter((step) => step.name !== ""),
      val: val.map(cloneStep).filter((step) => step.name !== ""),
    };
  } catch (_error) {
    return { train: [], val: [] };
  }
}

function parseNumberInRange(rawValue, { fallback, min, max }) {
  const parsed = Number.parseFloat(String(rawValue ?? ""));
  const numeric = Number.isFinite(parsed) ? parsed : fallback;
  if (!Number.isFinite(numeric)) {
    return fallback;
  }
  return Math.min(max, Math.max(min, numeric));
}

function resolveNumericValue(rawValue, fallback) {
  const parsed = Number.parseFloat(String(rawValue ?? ""));
  return Number.isFinite(parsed) ? parsed : fallback;
}

function toProbabilityDecimal(percentValue) {
  return Number((percentValue / 100).toFixed(4));
}

function getStepByName(steps, name) {
  return steps.find((step) => step.name === name) ?? null;
}

function resolveRotationRange(rawDegrees) {
  if (Array.isArray(rawDegrees) && rawDegrees.length >= 2) {
    const min = Number.parseFloat(String(rawDegrees[0]));
    const max = Number.parseFloat(String(rawDegrees[1]));
    if (Number.isFinite(min) && Number.isFinite(max)) {
      return [min, max];
    }
  }

  const bound = Number.parseFloat(String(rawDegrees ?? ""));
  if (!Number.isFinite(bound)) {
    return [-15, 15];
  }
  const absoluteBound = Math.abs(bound);
  return [-absoluteBound, absoluteBound];
}

function resolveScaleRange(rawScale) {
  if (Array.isArray(rawScale) && rawScale.length >= 2) {
    const min = Number.parseFloat(String(rawScale[0]));
    const max = Number.parseFloat(String(rawScale[1]));
    if (Number.isFinite(min) && Number.isFinite(max)) {
      return [min, max];
    }
  }
  return [...TRAINING_AUGMENTATION_DEFAULTS.randomResizedCrop.scale];
}

function updateAugmentationRowState(row) {
  if (!(row instanceof HTMLElement)) {
    return;
  }

  const enabledInput = row.querySelector("input[data-augmentation-enabled]");
  if (!(enabledInput instanceof HTMLInputElement)) {
    return;
  }

  const controls = row.querySelectorAll("input:not([data-augmentation-enabled])");
  controls.forEach((node) => {
    if (node instanceof HTMLInputElement) {
      node.disabled = !enabledInput.checked;
    }
  });
  row.classList.toggle("opacity-70", !enabledInput.checked);
}

function initializeAugmentationRows(workspace) {
  const rows = workspace.querySelectorAll("[data-augmentation-row]");
  rows.forEach((node) => {
    if (!(node instanceof HTMLElement)) {
      return;
    }

    updateAugmentationRowState(node);
    const enabledInput = node.querySelector("input[data-augmentation-enabled]");
    if (enabledInput instanceof HTMLInputElement) {
      enabledInput.addEventListener("change", () => {
        updateAugmentationRowState(node);
      });
    }
  });
}

function buildRandomResizedCropStep(row, existingStep) {
  const existingParams = isPlainObject(existingStep?.params) ? existingStep.params : {};
  const probabilityInput = row.querySelector("input[data-augmentation-probability]");
  const scaleMinInput = row.querySelector("input[data-augmentation-scale-min-pct]");
  const scaleMaxInput = row.querySelector("input[data-augmentation-scale-max-pct]");
  const [fallbackScaleMin, fallbackScaleMax] = resolveScaleRange(existingParams.scale);

  const probabilityPct = parseNumberInRange(probabilityInput?.value, {
    fallback: resolveNumericValue(existingParams.apply_p, 1) * 100,
    min: 0,
    max: 100,
  });
  const scaleMinPct = parseNumberInRange(scaleMinInput?.value, {
    fallback: fallbackScaleMin * 100,
    min: 1,
    max: 100,
  });
  const scaleMaxPct = parseNumberInRange(scaleMaxInput?.value, {
    fallback: fallbackScaleMax * 100,
    min: 1,
    max: 100,
  });
  const lowerScale = Math.min(scaleMinPct, scaleMaxPct);
  const upperScale = Math.max(scaleMinPct, scaleMaxPct);

  const params = {};
  params.size = Array.isArray(existingParams.size)
    ? [...existingParams.size]
    : [...TRAINING_AUGMENTATION_DEFAULTS.randomResizedCrop.size];
  params.scale = [
    Number((lowerScale / 100).toFixed(4)),
    Number((upperScale / 100).toFixed(4)),
  ];

  for (const passthroughKey of ["ratio", "interpolation", "antialias"]) {
    if (Object.hasOwn(existingParams, passthroughKey)) {
      params[passthroughKey] = cloneValue(existingParams[passthroughKey]);
    }
  }

  const probability = toProbabilityDecimal(probabilityPct);
  if (probability < 1) {
    params.apply_p = probability;
  }
  return { name: "RandomResizedCrop", params };
}

function buildRandomHorizontalFlipStep(row, existingStep) {
  const existingParams = isPlainObject(existingStep?.params) ? existingStep.params : {};
  const probabilityInput = row.querySelector("input[data-augmentation-probability]");
  const probabilityPct = parseNumberInRange(probabilityInput?.value, {
    fallback:
      resolveNumericValue(existingParams.p, resolveNumericValue(existingParams.apply_p, 0.5))
      * 100,
    min: 0,
    max: 100,
  });

  const params = cloneValue(existingParams);
  params.p = toProbabilityDecimal(probabilityPct);
  delete params.apply_p;
  return { name: "RandomHorizontalFlip", params };
}

function buildRandomRotationStep(row, existingStep) {
  const existingParams = isPlainObject(existingStep?.params) ? existingStep.params : {};
  const probabilityInput = row.querySelector("input[data-augmentation-probability]");
  const degreesMinInput = row.querySelector("input[data-augmentation-degrees-min]");
  const degreesMaxInput = row.querySelector("input[data-augmentation-degrees-max]");
  const [fallbackDegreesMin, fallbackDegreesMax] = resolveRotationRange(existingParams.degrees);

  const probabilityPct = parseNumberInRange(probabilityInput?.value, {
    fallback: resolveNumericValue(existingParams.apply_p, 1) * 100,
    min: 0,
    max: 100,
  });
  const degreesMin = parseNumberInRange(degreesMinInput?.value, {
    fallback: fallbackDegreesMin,
    min: -180,
    max: 180,
  });
  const degreesMax = parseNumberInRange(degreesMaxInput?.value, {
    fallback: fallbackDegreesMax,
    min: -180,
    max: 180,
  });

  const lowerDegrees = Math.min(degreesMin, degreesMax);
  const upperDegrees = Math.max(degreesMin, degreesMax);
  const params = cloneValue(existingParams);
  params.degrees = [lowerDegrees, upperDegrees];

  const probability = toProbabilityDecimal(probabilityPct);
  delete params.apply_p;
  if (probability < 1) {
    params.apply_p = probability;
  }

  return { name: "RandomRotation", params };
}

function buildColorJitterStep(row, existingStep) {
  const existingParams = isPlainObject(existingStep?.params) ? existingStep.params : {};
  const probabilityInput = row.querySelector("input[data-augmentation-probability]");
  const brightnessInput = row.querySelector("input[data-augmentation-brightness]");
  const contrastInput = row.querySelector("input[data-augmentation-contrast]");
  const saturationInput = row.querySelector("input[data-augmentation-saturation]");
  const hueInput = row.querySelector("input[data-augmentation-hue]");

  const probabilityPct = parseNumberInRange(probabilityInput?.value, {
    fallback: resolveNumericValue(existingParams.apply_p, 1) * 100,
    min: 0,
    max: 100,
  });
  const brightness = parseNumberInRange(brightnessInput?.value, {
    fallback: resolveNumericValue(existingParams.brightness, 0.2),
    min: 0,
    max: 2,
  });
  const contrast = parseNumberInRange(contrastInput?.value, {
    fallback: resolveNumericValue(existingParams.contrast, 0.2),
    min: 0,
    max: 2,
  });
  const saturation = parseNumberInRange(saturationInput?.value, {
    fallback: resolveNumericValue(existingParams.saturation, 0.1),
    min: 0,
    max: 2,
  });
  const hue = parseNumberInRange(hueInput?.value, {
    fallback: resolveNumericValue(existingParams.hue, 0.05),
    min: 0,
    max: 0.5,
  });

  const params = cloneValue(existingParams);
  params.brightness = Number(brightness.toFixed(4));
  params.contrast = Number(contrast.toFixed(4));
  params.saturation = Number(saturation.toFixed(4));
  params.hue = Number(hue.toFixed(4));

  const probability = toProbabilityDecimal(probabilityPct);
  delete params.apply_p;
  if (probability < 1) {
    params.apply_p = probability;
  }

  return { name: "ColorJitter", params };
}

function buildEditableAugmentationStep(stepName, row, existingTrain) {
  const enabledInput = row.querySelector("input[data-augmentation-enabled]");
  if (!(enabledInput instanceof HTMLInputElement) || !enabledInput.checked) {
    return null;
  }

  const existingStep = getStepByName(existingTrain, stepName);
  if (stepName === "RandomResizedCrop") {
    return buildRandomResizedCropStep(row, existingStep);
  }
  if (stepName === "RandomHorizontalFlip") {
    return buildRandomHorizontalFlipStep(row, existingStep);
  }
  if (stepName === "RandomRotation") {
    return buildRandomRotationStep(row, existingStep);
  }
  if (stepName === "ColorJitter") {
    return buildColorJitterStep(row, existingStep);
  }
  return null;
}

function resolveRequiredStep(existingTrain, stepName) {
  const existingStep = getStepByName(existingTrain, stepName);
  if (existingStep !== null) {
    return cloneStep(existingStep);
  }

  if (stepName === "Normalize") {
    return {
      name: "Normalize",
      params: cloneValue(TRAINING_AUGMENTATION_DEFAULTS.normalize),
    };
  }
  return { name: "ToImage", params: {} };
}

function buildAugmentationPayload(workspace) {
  const currentAugmentations = parseTrainingAugmentations(workspace);
  const existingTrain = currentAugmentations.train;
  const editableNames = new Set(TRAINING_AUGMENTATION_EDITABLE_ORDER);
  const requiredNames = new Set(TRAINING_AUGMENTATION_REQUIRED_ORDER);

  const editableTrainSteps = [];
  for (const stepName of TRAINING_AUGMENTATION_EDITABLE_ORDER) {
    const row = workspace.querySelector(`[data-augmentation-row='${stepName}']`);
    if (!(row instanceof HTMLElement)) {
      continue;
    }
    const step = buildEditableAugmentationStep(stepName, row, existingTrain);
    if (step !== null) {
      editableTrainSteps.push(step);
    }
  }

  const passthroughTrainSteps = existingTrain
    .filter((step) => !editableNames.has(step.name) && !requiredNames.has(step.name))
    .map(cloneStep);
  const requiredSteps = TRAINING_AUGMENTATION_REQUIRED_ORDER.map((stepName) =>
    resolveRequiredStep(existingTrain, stepName),
  );

  return {
    train: [...editableTrainSteps, ...passthroughTrainSteps, ...requiredSteps],
    val: currentAugmentations.val.map(cloneStep),
  };
}

function createHiddenAugmentationInput(container, name, value) {
  const input = document.createElement("input");
  input.type = "hidden";
  input.name = name;
  input.value = value;
  container.appendChild(input);
}

function syncAugmentationHiddenInputs(workspace, form) {
  const hiddenContainer = form.querySelector("[data-training-augmentation-hidden-fields]");
  if (!(hiddenContainer instanceof HTMLElement)) {
    return;
  }

  hiddenContainer.replaceChildren();
  const payload = buildAugmentationPayload(workspace);

  payload.train.forEach((step) => {
    createHiddenAugmentationInput(
      hiddenContainer,
      "augmentations.train[]",
      JSON.stringify(step),
    );
  });
  payload.val.forEach((step) => {
    createHiddenAugmentationInput(
      hiddenContainer,
      "augmentations.val[]",
      JSON.stringify(step),
    );
  });
}

function initializeTrainingConfigForm(workspace) {
  initializeTrainingSectionState(workspace);
  updateTrainingEffectiveBatchLabel(workspace);
  updateTrainingConditionalFields(workspace);
  initializeAugmentationRows(workspace);

  const batchSizeInput = workspace.querySelector("[name='hyperparameters.batch_size']");
  const batchMultiplierInput = workspace.querySelector(
    "[name='hyperparameters.batch_multiplier']",
  );
  const configForm = workspace.querySelector("form[data-training-config-form]");
  const deviceInputs = workspace.querySelectorAll(
    "input[name='hardware.selected_devices[]'][data-training-device-toggle]",
  );

  if (batchSizeInput instanceof HTMLInputElement) {
    batchSizeInput.addEventListener("input", () => {
      updateTrainingEffectiveBatchLabel(workspace);
    });
  }
  if (batchMultiplierInput instanceof HTMLInputElement) {
    batchMultiplierInput.addEventListener("input", () => {
      updateTrainingEffectiveBatchLabel(workspace);
    });
  }
  deviceInputs.forEach((node) => {
    if (node instanceof HTMLInputElement) {
      node.addEventListener("change", () => {
        updateTrainingEffectiveBatchLabel(workspace);
      });
    }
  });

  const optimizerSelect = workspace.querySelector("[data-training-input='optimizer']");
  if (optimizerSelect instanceof HTMLSelectElement) {
    optimizerSelect.addEventListener("change", () => {
      updateTrainingConditionalFields(workspace);
    });
  }

  const schedulerSelect = workspace.querySelector("[data-training-input='scheduler']");
  if (schedulerSelect instanceof HTMLSelectElement) {
    schedulerSelect.addEventListener("change", () => {
      updateTrainingConditionalFields(workspace);
    });
  }

  const lossSelect = workspace.querySelector("[data-training-input='loss']");
  if (lossSelect instanceof HTMLSelectElement) {
    lossSelect.addEventListener("change", () => {
      updateTrainingConditionalFields(workspace);
    });
  }

  if (configForm instanceof HTMLFormElement) {
    configForm.addEventListener("submit", () => {
      syncAugmentationHiddenInputs(workspace, configForm);
    });
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
const CHART_LAYOUT = {
  fallbackWidth: 640,
  fallbackHeight: 300,
  padding: {
    left: 64,
    right: 20,
    top: 24,
    bottom: 48,
  },
  minPlotWidth: 80,
  minPlotHeight: 80,
};

function resolveChartLayout(chart) {
  const measuredWidth = Math.round(chart.clientWidth ?? 0);
  const measuredHeight = Math.round(chart.clientHeight ?? 0);
  const width = measuredWidth > 0 ? measuredWidth : CHART_LAYOUT.fallbackWidth;
  const height = measuredHeight > 0 ? measuredHeight : CHART_LAYOUT.fallbackHeight;
  chart.setAttribute("viewBox", `0 0 ${width} ${height}`);

  const maxXStart = Math.max(
    0,
    width - CHART_LAYOUT.padding.right - CHART_LAYOUT.minPlotWidth,
  );
  const xStart = Math.min(CHART_LAYOUT.padding.left, maxXStart);
  const xEnd = Math.min(
    width,
    Math.max(xStart + CHART_LAYOUT.minPlotWidth, width - CHART_LAYOUT.padding.right),
  );

  const maxYStart = Math.max(
    0,
    height - CHART_LAYOUT.padding.bottom - CHART_LAYOUT.minPlotHeight,
  );
  const yStart = Math.min(CHART_LAYOUT.padding.top, maxYStart);
  const yEnd = Math.min(
    height,
    Math.max(
      yStart + CHART_LAYOUT.minPlotHeight,
      height - CHART_LAYOUT.padding.bottom,
    ),
  );

  return { xStart, xEnd, yStart, yEnd };
}

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

function mapEpochToX(epoch, xDomain, frame) {
  const xSpan = Math.max(frame.xEnd - frame.xStart, 1);
  const epochSpan = Math.max(xDomain.max - xDomain.min, 1);
  return frame.xStart + ((epoch - xDomain.min) / epochSpan) * xSpan;
}

function mapValueToY(value, yDomain, frame) {
  const ySpan = Math.max(frame.yEnd - frame.yStart, 1);
  const domainSpan = Math.max(yDomain.max - yDomain.min, 1e-9);
  return frame.yEnd - ((value - yDomain.min) / domainSpan) * ySpan;
}

function buildPolylinePoints(points, xDomain, yDomain, frame) {
  if (points.length === 0) {
    return "";
  }

  return [...points]
    .sort((left, right) => left.epoch - right.epoch)
    .map((point) => {
      const x = mapEpochToX(point.epoch, xDomain, frame);
      const y = mapValueToY(point.value, yDomain, frame);
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
    frame,
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
    const x = mapEpochToX(tick, xDomain, frame);
    appendSvgLine(
      xTicksNode,
      {
        x1: x.toFixed(2),
        y1: frame.yEnd,
        x2: x.toFixed(2),
        y2: frame.yEnd + 5,
      },
      "text-slate-400 dark:text-slate-500",
    );
    appendSvgText(
      xTicksNode,
      {
        x: x.toFixed(2),
        y: frame.yEnd + 18,
        "text-anchor": "middle",
      },
      String(tick),
      "text-[10px] text-slate-500 dark:text-slate-400",
    );
  }

  for (const tick of buildLinearTicks(yDomain.min, yDomain.max, yTickCount)) {
    const y = mapValueToY(tick, yDomain, frame);
    const yTickStartX = Math.max(frame.xStart - 5, 0);
    appendSvgLine(
      yTicksNode,
      {
        x1: frame.xStart,
        y1: y.toFixed(2),
        x2: frame.xEnd,
        y2: y.toFixed(2),
      },
      "text-slate-200 dark:text-slate-800",
    );
    appendSvgLine(
      yTicksNode,
      {
        x1: yTickStartX,
        y1: y.toFixed(2),
        x2: frame.xStart,
        y2: y.toFixed(2),
      },
      "text-slate-400 dark:text-slate-500",
    );
    appendSvgText(
      yTicksNode,
      {
        x: Math.max(frame.xStart - 8, 0),
        y: (y + 3).toFixed(2),
        "text-anchor": "end",
      },
      yFormatter(tick),
      "text-[10px] text-slate-500 dark:text-slate-400",
    );
  }
}

function updateAxisLines(workspace, { axisXSelector, axisYSelector }, frame) {
  const axisX = workspace.querySelector(axisXSelector);
  const axisY = workspace.querySelector(axisYSelector);

  if (axisY) {
    axisY.setAttribute("x1", frame.xStart.toFixed(2));
    axisY.setAttribute("y1", frame.yStart.toFixed(2));
    axisY.setAttribute("x2", frame.xStart.toFixed(2));
    axisY.setAttribute("y2", frame.yEnd.toFixed(2));
  }

  if (axisX) {
    axisX.setAttribute("x1", frame.xStart.toFixed(2));
    axisX.setAttribute("y1", frame.yEnd.toFixed(2));
    axisX.setAttribute("x2", frame.xEnd.toFixed(2));
    axisX.setAttribute("y2", frame.yEnd.toFixed(2));
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

  const lossChart = workspace.querySelector("[data-training-loss-chart]");
  const trainPolyline = workspace.querySelector("[data-training-loss-train]");
  const valPolyline = workspace.querySelector("[data-training-loss-val]");
  const emptyState = workspace.querySelector("[data-training-loss-empty]");
  const progressFill = workspace.querySelector("[data-training-progress-fill]");
  const epochLabel = workspace.querySelector("[data-training-epoch-label]");

  let frame = null;
  if (lossChart instanceof SVGSVGElement) {
    frame = resolveChartLayout(lossChart);
    updateAxisLines(
      workspace,
      {
        axisXSelector: "[data-training-loss-axis-x]",
        axisYSelector: "[data-training-loss-axis-y]",
      },
      frame,
    );
    renderChartTicks(workspace, {
      xTicksSelector: "[data-training-loss-x-ticks]",
      yTicksSelector: "[data-training-loss-y-ticks]",
      xDomain,
      yDomain,
      yTickCount: 5,
      yFormatter: formatLossTick,
      frame,
    });
  }

  if (trainPolyline && frame) {
    trainPolyline.setAttribute(
      "points",
      buildPolylinePoints(trainPoints, xDomain, yDomain, frame),
    );
  }
  if (valPolyline && frame) {
    valPolyline.setAttribute(
      "points",
      buildPolylinePoints(valPoints, xDomain, yDomain, frame),
    );
  }

  const hasLossValues = trainPoints.length > 0 || valPoints.length > 0;
  if (emptyState) {
    emptyState.classList.toggle("hidden", hasLossValues);
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

  const metricChart = workspace.querySelector("[data-training-metric-chart]");
  const trainPolyline = workspace.querySelector("[data-training-metric-train]");
  const valPolyline = workspace.querySelector("[data-training-metric-val]");
  const emptyState = workspace.querySelector("[data-training-metric-empty]");

  let frame = null;
  if (metricChart instanceof SVGSVGElement) {
    frame = resolveChartLayout(metricChart);
    updateAxisLines(
      workspace,
      {
        axisXSelector: "[data-training-metric-axis-x]",
        axisYSelector: "[data-training-metric-axis-y]",
      },
      frame,
    );
    renderChartTicks(workspace, {
      xTicksSelector: "[data-training-metric-x-ticks]",
      yTicksSelector: "[data-training-metric-y-ticks]",
      xDomain,
      yDomain,
      yTickCount: 5,
      yFormatter: (value) => value.toFixed(2),
      frame,
    });
  }

  if (trainPolyline && frame) {
    trainPolyline.setAttribute(
      "points",
      buildPolylinePoints(trainF1Points, xDomain, yDomain, frame),
    );
  }
  if (valPolyline && frame) {
    valPolyline.setAttribute(
      "points",
      buildPolylinePoints(valF1Points, xDomain, yDomain, frame),
    );
  }

  const hasMetricValues = trainF1Points.length > 0 || valF1Points.length > 0;
  if (emptyState) {
    emptyState.classList.toggle("hidden", hasMetricValues);
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

function initializeTrainingChartResizeObserver(workspace, renderCharts) {
  closeTrainingChartResizeObserver();
  if (typeof ResizeObserver === "undefined") {
    return;
  }

  const charts = workspace.querySelectorAll(
    "[data-training-loss-chart], [data-training-metric-chart]",
  );
  if (charts.length === 0) {
    return;
  }

  let scheduled = false;
  trainingChartResizeObserver = new ResizeObserver(() => {
    if (scheduled) {
      return;
    }
    scheduled = true;
    trainingChartResizeRaf = window.requestAnimationFrame(() => {
      scheduled = false;
      trainingChartResizeRaf = null;
      renderCharts();
    });
  });

  charts.forEach((chart) => {
    trainingChartResizeObserver?.observe(chart);
  });
}

function initializeTrainingWorkspace() {
  closeTrainingEventSource();
  closeTrainingChartResizeObserver();

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
  const renderCharts = () => {
    renderTrainingLossChart(workspace, epochs);
    renderTrainingMetricChart(workspace, epochs);
  };
  renderCharts();
  initializeTrainingChartResizeObserver(workspace, renderCharts);
  initializeTrainingConfigForm(workspace);
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
      renderCharts();
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
window.addEventListener("beforeunload", closeTrainingChartResizeObserver);
