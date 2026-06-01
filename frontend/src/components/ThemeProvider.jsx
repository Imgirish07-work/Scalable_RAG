import { createContext, useContext, useLayoutEffect, useState } from "react";
import claudeDark  from "../themes/claude-dark";
import claudeLight from "../themes/claude-light";

export const THEMES = {
  "claude-dark":  claudeDark,
  "claude-light": claudeLight,
};

const DEFAULT_ID  = "claude-dark";
const STORAGE_KEY = "scalable-rag-theme";

const ThemeCtx = createContext(null);

// useLayoutEffect fires before paint — zero flash of wrong theme
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

export const useTheme = () => useContext(ThemeCtx);
