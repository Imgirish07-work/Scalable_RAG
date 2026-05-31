/**
 * Claude Light — cream background with terracotta accent.
 * Mirrors Claude.ai's light mode: warm paper feel, high text contrast.
 */
const claudeLight = {
  id:    "claude-light",
  label: "Claude Light",
  vars: {
    /* ── Backgrounds ── */
    "--c-bg":           "#F5EFE6",
    "--c-bgPanel":      "#FAF5EC",
    "--c-bgSoft":       "#EDE7DC",
    "--c-bgCard":       "#FFFFFF",
    "--c-bgInput":      "#FFFFFF",
    "--c-bgDeep":       "#EDE7DC",
    "--c-bgResponse":   "#FAF5EC",

    /* ── Text ── */
    "--c-ink":          "#1A1915",
    "--c-inkSoft":      "#4A4843",
    "--c-inkMuted":     "#7C7972",

    /* ── Borders ── */
    "--c-line":         "#D7D0C3",
    "--c-lineSoft":     "#E2DBCE",
    "--c-lineCard":     "#CFC8BC",

    /* ── Accent (terracotta) ── */
    "--c-accent":       "#CC7A57",
    "--c-accentHover":  "#B86A48",
    "--c-accentBg":     "rgba(204,122,87,0.10)",
    "--c-accentBorder": "rgba(204,122,87,0.35)",
    "--c-accentGlow":   "rgba(204,122,87,0.18)",

    /* ── Status ── */
    "--c-ok":           "#3F8F58",
    "--c-warn":         "#B5781F",
    "--c-danger":       "#B83D2C",

    "--c-bgSuccess":     "#E5F1E8",
    "--c-borderSuccess": "#A8CDB1",
    "--c-textSuccess":   "#2D6E40",
    "--c-bgWarning":     "#F5E8CC",
    "--c-borderWarning": "#D4B97A",
    "--c-textWarning":   "#8A5A14",
    "--c-bgError":       "#F2DDD5",
    "--c-borderError":   "#D69E8E",
    "--c-textError":     "#8F2D1F",
    "--c-bgInfo":        "#DDE6F0",
    "--c-borderInfo":    "#A6BBD4",
    "--c-textInfo":      "#3A5A82",
  },
};

export default claudeLight;
