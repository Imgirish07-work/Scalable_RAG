import { memo } from "react";
import { C } from "../../theme";

/**
 * Skeleton — animated placeholder atom for loading states.
 *
 * Props:
 *   width     — CSS width (default: "100%")
 *   height    — CSS height (default: 12)
 *   rounded   — border-radius variant: "sm" | "md" | "lg" | "full" (default: "md")
 *   style     — extra inline styles
 *   className — extra Tailwind classes
 */

const RADII = { sm: 4, md: 8, lg: 12, full: 9999 };

export default memo(function Skeleton({
  width     = "100%",
  height    = 12,
  rounded   = "md",
  style,
  className = "",
}) {
  return (
    <div
      aria-hidden="true"
      className={`animate-pulse ${className}`}
      style={{
        width,
        height,
        borderRadius: RADII[rounded] ?? RADII.md,
        background: C.lineSoft,
        ...style,
      }}
    />
  );
})
