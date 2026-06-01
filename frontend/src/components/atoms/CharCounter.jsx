import { memo } from "react";

export default memo(function CharCounter({ current, max, warnAt, style, className = "" }) {
  const warn = warnAt ?? Math.max(max - 10, Math.floor(max * 0.9));
  const tone =
    current >= max  ? "text-red-400"
    : current >= warn ? "text-yellow-400"
    : "text-theme-ink-muted";
  return (
    <span
      className={`text-[11px] tabular-nums ${tone} ${className}`}
      style={style}
      aria-live="polite"
    >
      {current}/{max}
    </span>
  );
});
