import { memo } from "react";
import { C } from "../../theme";

/**
 * MetricChip — value + label stat display atom.
 *
 * Props:
 *   value     — primary display value (string or number)
 *   label     — description below the value
 *   style     — extra inline styles
 *   className — extra Tailwind classes
 */
export default memo(function MetricChip({
  value,
  label,
  style,
  className = "",
}) {
  return (
    <div
      className={`flex-1 text-center rounded ${className}`}
      style={{
        padding:    "8px 12px",
        background: C.bgSoft,
        border:     `1px solid ${C.lineSoft}`,
        ...style,
      }}
    >
      <div
        className="font-mono font-semibold text-sm"
        style={{ color: C.ink, lineHeight: 1.2 }}
      >
        {value}
      </div>
      <div
        className="font-mono uppercase tracking-widest mt-1"
        style={{ fontSize: 9.5, color: C.inkMuted }}
      >
        {label}
      </div>
    </div>
  );
})
