import { memo, useState } from "react";
import { C } from "../../theme";

/**
 * IconButton — small icon-only button atom with hover highlight.
 *
 * Props:
 *   title     — tooltip text
 *   onClick   — click handler
 *   size      — width/height in px (default: 26)
 *   disabled  — disables interaction
 *   style     — extra inline styles
 *   className — extra Tailwind classes
 *   children  — icon content (SVG, emoji, etc.)
 */
export default memo(function IconButton({
  title,
  onClick,
  size      = 26,
  disabled  = false,
  style,
  className = "",
  children,
}) {
  const [hovered, setHovered] = useState(false);

  return (
    <button
      title={title}
      aria-label={title}
      onClick={onClick}
      disabled={disabled}
      aria-disabled={disabled}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      className={`inline-flex items-center justify-center rounded transition-colors ${className}`}
      style={{
        width:      size,
        height:     size,
        background: hovered && !disabled ? C.bgSoft : "transparent",
        border:     "none",
        cursor:     disabled ? "not-allowed" : "pointer",
        color:      hovered && !disabled ? C.inkSoft : C.inkMuted,
        opacity:    disabled ? 0.4 : 1,
        flexShrink: 0,
        padding:    0,
        ...style,
      }}
    >
      {children}
    </button>
  );
})
