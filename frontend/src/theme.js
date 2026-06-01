// design tokens — values applied at runtime by ThemeProvider via CSS custom props
export const C = {
  bg:           "var(--c-bg)",
  bgPanel:      "var(--c-bgPanel)",
  bgSoft:       "var(--c-bgSoft)",
  bgCard:       "var(--c-bgCard)",
  bgInput:      "var(--c-bgInput)",
  bgDeep:       "var(--c-bgDeep)",
  bgResponse:   "var(--c-bgResponse)",

  ink:          "var(--c-ink)",
  inkSoft:      "var(--c-inkSoft)",
  inkMuted:     "var(--c-inkMuted)",

  line:         "var(--c-line)",
  lineSoft:     "var(--c-lineSoft)",
  lineCard:     "var(--c-lineCard)",

  accent:       "var(--c-accent)",
  accentHover:  "var(--c-accentHover)",
  accentBg:     "var(--c-accentBg)",
  accentBorder: "var(--c-accentBorder)",
  accentGlow:   "var(--c-accentGlow)",

  ok:           "var(--c-ok)",
  warn:         "var(--c-warn)",
  danger:       "var(--c-danger)",
};

export const R = {
  sm: 4,
  md: 8,
  lg: 12,
};

export const Z = {
  modal:   1000,
  tooltip: 900,
  header:  800,
};
