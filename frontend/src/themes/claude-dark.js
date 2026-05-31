/**
 * Claude Dark — warm deep-gray with softened terracotta accent.
 * Mirrors Claude.ai's dark mode: low-chroma neutrals, content-first contrast.
 */
const claudeDark = {
  id:    "claude-dark",
  label: "Claude Dark",
  vars: {
    /* ── Backgrounds ── */
    "--c-bg":           "#1F1E1C",
    "--c-bgPanel":      "#262624",
    "--c-bgSoft":       "#2B2A28",
    "--c-bgCard":       "#302F2D",
    "--c-bgInput":      "#353432",
    "--c-bgDeep":       "#1A1917",
    "--c-bgResponse":   "#1A1917",

    /* ── Text ── */
    "--c-ink":          "#E8E6DF",
    "--c-inkSoft":      "#C8C6BF",
    "--c-inkMuted":     "#908E87",

    /* ── Borders ── */
    "--c-line":         "#3A3936",
    "--c-lineSoft":     "#353432",
    "--c-lineCard":     "#45433F",

    /* ── Accent (terracotta) ── */
    "--c-accent":       "#D08A6A",
    "--c-accentHover":  "#DA9A7C",
    "--c-accentBg":     "rgba(208,138,106,0.10)",
    "--c-accentBorder": "rgba(208,138,106,0.35)",
    "--c-accentGlow":   "rgba(208,138,106,0.15)",

    /* ── Status — universal scale, kept readable on warm-gray bg ── */
    "--c-ok":           "#7FBF8F",
    "--c-warn":         "#E6B266",
    "--c-danger":       "#E89A8A",

    "--c-bgSuccess":     "#1F3122",
    "--c-borderSuccess": "#365C3C",
    "--c-textSuccess":   "#9AD0A7",
    "--c-bgWarning":     "#2F2618",
    "--c-borderWarning": "#5C4A2A",
    "--c-textWarning":   "#D4B17A",
    "--c-bgError":       "#2F1F1C",
    "--c-borderError":   "#5C2F2A",
    "--c-textError":     "#E5A89C",
    "--c-bgInfo":        "#1E2530",
    "--c-borderInfo":    "#3A4C68",
    "--c-textInfo":      "#9AB6DC",
  },
};

export default claudeDark;
