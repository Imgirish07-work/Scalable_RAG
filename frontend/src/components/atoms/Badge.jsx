import { memo } from "react";
import { C } from "../../theme";

/**
 * Badge — small label/tag atom.
 *
 * Props:
 *   children  — badge content
 *   variant   — "default" | "success" | "warning" | "danger" | "info"  (default: "default")
 *   size      — "sm" | "md"  (default: "sm")
 *   dot       — show a leading dot (boolean)
 *   style     — extra inline styles
 *   className — extra Tailwind classes
 */

const VARIANTS = {
  default: { background: C.accentBg,            color: C.accent,  border: `1px solid ${C.accentBorder}`, dot: C.accent  },
  success: { background: "rgba(63,185,80,0.1)",  color: C.ok,      border: "1px solid rgba(63,185,80,0.3)",  dot: C.ok   },
  warning: { background: "rgba(210,153,34,0.1)", color: C.warn,    border: "1px solid rgba(210,153,34,0.3)", dot: C.warn },
  danger:  { background: "rgba(220,38,38,0.1)",  color: C.danger,  border: "1px solid rgba(220,38,38,0.3)",  dot: C.danger },
  info:    { background: "rgba(99,102,241,0.1)", color: "#818cf8", border: "1px solid rgba(99,102,241,0.3)", dot: "#818cf8" },
};

const SIZES = {
  sm: { fontSize: 10, padding: "2px 8px",  borderRadius: 4 },
  md: { fontSize: 12, padding: "4px 10px", borderRadius: 6 },
};

export default memo(function Badge({
  children,
  variant   = "default",
  size      = "sm",
  dot       = false,
  style,
  className = "",
}) {
  const v = VARIANTS[variant] ?? VARIANTS.default;
  const s = SIZES[size]       ?? SIZES.sm;

  return (
    <span
      className={`inline-flex items-center gap-1.5 font-medium ${className}`}
      style={{ ...s, background: v.background, color: v.color, border: v.border, ...style }}
    >
      {dot && (
        <span
          style={{ width: 6, height: 6, borderRadius: "50%", background: v.dot, flexShrink: 0, display: "inline-block" }}
        />
      )}
      {children}
    </span>
  );
})
