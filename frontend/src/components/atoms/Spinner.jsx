import { memo } from "react";
import { C } from "../../theme";

const SIZES = { sm: 16, md: 24, lg: 32 };

export default memo(function Spinner({ size = "md", style, className = "" }) {
  const px = typeof size === "number" ? size : (SIZES[size] ?? SIZES.md);
  return (
    <div
      aria-label="Loading"
      className={`animate-spin rounded-full flex-shrink-0 ${className}`}
      style={{
        width: px,
        height: px,
        border: `2px solid ${C.accentBg}`,
        borderTopColor: C.accent,
        ...style,
      }}
    />
  );
});
