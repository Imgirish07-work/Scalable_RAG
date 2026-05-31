import { memo } from "react";

export default memo(function DocPreviewBadge({ label = "FILE", tone = "themed" }) {
  const bg =
    tone === "light" ? "rgba(255,255,255,0.92)"
      : tone === "dark" ? "rgba(58,57,54,0.92)"
      : "var(--c-bgSoft)";
  const fg =
    tone === "light" ? "#262624"
      : tone === "dark" ? "#ffffff"
      : "var(--c-inkSoft)";

  return (
    <div
      style={{
        position: "absolute",
        left: 5,
        bottom: 5,
        display: "inline-flex",
        alignItems: "center",
        gap: 3,
        padding: "2px 5px 2px 4px",
        borderRadius: 5,
        background: bg,
        backdropFilter: "blur(6px)",
        WebkitBackdropFilter: "blur(6px)",
        boxShadow: "0 1px 3px rgba(0,0,0,0.22)",
        pointerEvents: "none",
      }}
    >
      <span
        style={{
          color: fg,
          fontSize: 8,
          fontWeight: 600,
          letterSpacing: "0.04em",
          fontFamily: 'system-ui, -apple-system, sans-serif',
        }}
      >
        {label}
      </span>
    </div>
  );
});
