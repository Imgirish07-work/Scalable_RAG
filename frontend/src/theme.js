/**
 * theme.js — design token references.
 *
 * C maps every token to its CSS custom property reference string.
 * Actual colour values are applied at runtime by ThemeProvider
 * (src/components/ThemeProvider.jsx), which writes them to <html>.
 *
 * This means:
 *   • style={{ background: C.bg }}  →  style={{ background: "var(--c-bg)" }}
 *   • Tailwind arbitrary: bg-[var(--c-bg)]
 *   • Both update instantly when the theme changes — zero component changes needed.
 *
 * To add a new token: add it here AND in every theme file under src/themes/.
 */

export const C = {
  /* ── Backgrounds ── */
  bg:           "var(--c-bg)",
  bgPanel:      "var(--c-bgPanel)",
  bgSoft:       "var(--c-bgSoft)",
  bgCard:       "var(--c-bgCard)",
  bgInput:      "var(--c-bgInput)",
  bgDeep:       "var(--c-bgDeep)",
  bgResponse:   "var(--c-bgResponse)",

  /* ── Text ── */
  ink:          "var(--c-ink)",
  inkSoft:      "var(--c-inkSoft)",
  inkMuted:     "var(--c-inkMuted)",

  /* ── Borders ── */
  line:         "var(--c-line)",
  lineSoft:     "var(--c-lineSoft)",
  lineCard:     "var(--c-lineCard)",

  /* ── Accent (cyan) ── */
  accent:       "var(--c-accent)",
  accentHover:  "var(--c-accentHover)",
  accentBg:     "var(--c-accentBg)",
  accentBorder: "var(--c-accentBorder)",
  accentGlow:   "var(--c-accentGlow)",

  /* ── Status ── */
  ok:           "var(--c-ok)",
  warn:         "var(--c-warn)",
  danger:       "var(--c-danger)",
};

/** Border-radius scale — not theme-dependent */
export const R = {
  sm: 4,
  md: 8,
  lg: 12,
};

/** Z-index scale — not theme-dependent */
export const Z = {
  modal:   1000,
  tooltip: 900,
  header:  800,
};
