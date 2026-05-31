import { memo } from "react";
import { C } from "../../theme";

/**
 * ProgressBar — determinate or indeterminate progress bar atom.
 *
 * Props:
 *   value          — current progress (0–100); omit for indeterminate
 *   height         — bar height in px (default: 6)
 *   gradient       — CSS gradient or color for the fill (default: cyan gradient)
 *   indeterminate  — force indeterminate mode (default: false)
 *   animated       — smooth width transition (default: true)
 *   style          — extra inline styles on the outer track
 *   className      — extra Tailwind classes on the outer track
 */
export default memo(function ProgressBar({
  value,
  height        = 6,
  gradient      = `linear-gradient(90deg, ${C.accent}, #14b8a6)`,
  indeterminate = false,
  animated      = true,
  style,
  className     = "",
}) {
  const isIndet = indeterminate || value == null;

  return (
    <div
      role="progressbar"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={isIndet ? undefined : Math.min(100, Math.max(0, value))}
      className={`rounded-full overflow-hidden ${className}`}
      style={{ height, background: C.lineSoft, ...style }}
    >
      <div
        className="h-full rounded-full"
        style={
          isIndet
            ? {
                width: "40%",
                background: gradient,
                animation: "progress-slide 1.2s ease-in-out infinite",
              }
            : {
                width: `${Math.min(100, Math.max(0, value))}%`,
                background: gradient,
                transition: animated ? "width 0.5s ease" : "none",
              }
        }
      />
      {isIndet && (
        <style>{`@keyframes progress-slide { 0% { margin-left: -40%; } 100% { margin-left: 100%; } }`}</style>
      )}
    </div>
  );
})
