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
