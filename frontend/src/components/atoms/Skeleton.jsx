import { memo } from "react";
import { C } from "../../theme";

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
