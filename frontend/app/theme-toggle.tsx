"use client";
import { useEffect, useState } from "react";

type Theme = "light" | "dark";
const KEY = "df-theme";

function current(): Theme {
  const set = document.documentElement.dataset.theme;
  if (set === "light" || set === "dark") return set;
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

/** Sidebar button that switches between the light and dark palette and remembers the choice. */
export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme | null>(null);
  useEffect(() => { setTheme(current()); }, []);

  function flip() {
    const next: Theme = current() === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem(KEY, next); } catch { /* private mode: the choice just isn't remembered */ }
    setTheme(next);
  }

  return (
    <button type="button" className="theme" onClick={flip} aria-label="Toggle light or dark theme">
      <span aria-hidden>{theme === "dark" ? "☀" : "☾"}</span>
      <span className="tl">{theme === "dark" ? "Light mode" : "Dark mode"}</span>
    </button>
  );
}
