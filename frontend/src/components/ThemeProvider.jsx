import { createContext, useContext, useLayoutEffect, useState } from "react";
import darkSlate  from "../themes/dark-slate";
import darkNavy   from "../themes/dark-navy";
import lightWarm  from "../themes/light-warm";

/**
 * Theme registry — add new themes here.
 * Each theme is a plain object: { id, label, vars: { "--c-*": value } }
 */
export const THEMES = {
  "light-warm": lightWarm,
  "dark-slate": darkSlate,
  "dark-navy":  darkNavy,
};

const DEFAULT_ID  = "dark-slate";
const STORAGE_KEY = "ailab-theme";

const ThemeCtx = createContext(null);

/**
 * ThemeProvider — writes the active theme as CSS custom properties on <html>.
 * Wrap once at the app root (outside BrowserRouter, inside StrictMode).
 *
 * useLayoutEffect fires before paint — zero flash of wrong theme.
 * Theme choice is persisted to localStorage automatically.
 */
export function ThemeProvider({ children }) {
  const [id, setId] = useState(
    () => localStorage.getItem(STORAGE_KEY) ?? DEFAULT_ID,
  );

  useLayoutEffect(() => {
    const theme = THEMES[id] ?? THEMES[DEFAULT_ID];
    const root  = document.documentElement;
    Object.entries(theme.vars).forEach(([k, v]) => root.style.setProperty(k, v));
    root.setAttribute("data-theme", id);
    localStorage.setItem(STORAGE_KEY, id);
  }, [id]);

  /** Cycle to the next available theme. */
  const toggle = () => {
    const ids  = Object.keys(THEMES);
    const next = ids[(ids.indexOf(id) + 1) % ids.length];
    setId(next);
  };

  return (
    <ThemeCtx.Provider value={{ id, setId, toggle, themes: THEMES }}>
      {children}
    </ThemeCtx.Provider>
  );
}

/**
 * useTheme — access the active theme id, switch function, and theme registry.
 *
 * @returns {{ id: string, setId: fn, toggle: fn, themes: object }}
 */
export const useTheme = () => useContext(ThemeCtx);
