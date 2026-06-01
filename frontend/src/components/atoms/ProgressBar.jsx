import { memo } from "react";
import { C } from "../../theme";

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
