import { memo } from "react";

/**
 * CharCounter — inline "current/max" indicator atom for length-limited inputs.
 *
 * Color thresholds:
 *   current >= max      → red    (limit hit)
 *   current >= warnAt   → yellow (approaching limit)
 *   otherwise           → muted
 *
 * Props:
 *   current   — current length
 *   max       — maximum allowed length
 *   warnAt    — yellow threshold (default: max - 10% rounded, min max - 10)
 *   style     — extra inline styles
 *   className — extra Tailwind classes
 */
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
