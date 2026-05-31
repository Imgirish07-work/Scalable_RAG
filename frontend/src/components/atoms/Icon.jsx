import { memo } from "react";

/**
 * Icon — renders an SVG as a CSS mask so it inherits color from `style.color`
 * or Tailwind `text-*` classes via currentColor.
 *
 * Props:
 *   src       — SVG URL (imported asset)
 *   className — Tailwind classes (w-4 h-4, text-cyan-400, etc.)
 *   alt       — accessibility label (defaults to empty)
 *   style     — extra inline styles
 */
export default memo(function Icon({ src, className = "w-4 h-4", alt = "", style }) {
  return (
    <span
      role={alt ? "img" : "presentation"}
      aria-label={alt || undefined}
      aria-hidden={!alt ? "true" : undefined}
      className={`inline-block flex-shrink-0 ${className}`}
      style={{
        maskImage: `url(${src})`,
        WebkitMaskImage: `url(${src})`,
        maskSize: "contain",
        WebkitMaskSize: "contain",
        maskRepeat: "no-repeat",
        WebkitMaskRepeat: "no-repeat",
        maskPosition: "center",
        WebkitMaskPosition: "center",
        backgroundColor: "currentColor",
        ...style,
      }}
    />
  );
});
