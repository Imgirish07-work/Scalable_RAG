import { memo, useState } from "react";
import { C } from "../../theme";

/**
 * Button — universal button atom.
 *
 * Props:
 *   variant   — "primary" | "ghost" | "danger" | "outline"  (default: "primary")
 *   size      — "sm" | "md" | "lg"                          (default: "md")
 *   disabled  — boolean
 *   onClick   — handler
 *   style     — extra inline styles (merged last, highest priority)
 *   className — extra Tailwind classes
 *   children  — content
 *   title     — tooltip text
 *   type      — button type attr (default: "button")
 */
export default memo(function Button({
  variant   = "primary",
  size      = "md",
  disabled  = false,
  onClick,
  style,
  className = "",
  children,
  title,
  type      = "button",
}) {
  const [hovered, setHovered] = useState(false);

  const sizes = {
    sm: { padding: "4px 10px",  fontSize: 11, height: 26, borderRadius: 4  },
    md: { padding: "6px 14px",  fontSize: 12, height: 32, borderRadius: 6  },
    lg: { padding: "8px 20px",  fontSize: 13, height: 38, borderRadius: 8  },
  };

  const variants = {
    primary: {
      background: disabled
        ? C.bgSoft
        : hovered
        ? `linear-gradient(to right, ${C.accent}, ${C.accentHover})`
        : `linear-gradient(to right, ${C.accentHover}, ${C.accent})`,
      color:      disabled ? C.inkMuted : "#fff",
      border:     "none",
      boxShadow:  disabled ? "none" : hovered ? `0 0 14px ${C.accentGlow}` : "none",
    },
    ghost: {
      background: hovered ? C.accentBg : "transparent",
      color:      hovered ? C.accent : C.inkSoft,
      border:     "none",
      boxShadow:  "none",
    },
    outline: {
      background: hovered ? C.accentBg : "transparent",
      color:      hovered ? C.accent : C.inkSoft,
      border:     `1px solid ${hovered ? C.accent : C.lineSoft}`,
      boxShadow:  "none",
    },
    danger: {
      background: disabled ? C.bgSoft : hovered ? "#b91c1c" : C.danger,
      color:      disabled ? C.inkMuted : "#fff",
      border:     "none",
      boxShadow:  "none",
    },
  };

  const v = variants[variant] ?? variants.primary;
  const s = sizes[size] ?? sizes.md;

  return (
    <button
      type={type}
      title={title}
      disabled={disabled}
      aria-disabled={disabled}
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      className={`inline-flex items-center justify-center gap-1.5 font-medium transition-all ${className}`}
      style={{
        ...s,
        ...v,
        cursor:     disabled ? "not-allowed" : "pointer",
        opacity:    disabled ? 0.45 : 1,
        flexShrink: 0,
        transition: "background 0.15s, opacity 0.15s, box-shadow 0.15s, color 0.15s",
        ...style,
      }}
    >
      {children}
    </button>
  );
})
