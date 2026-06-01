import { memo } from "react";
import { C } from "../../theme";

export default memo(function Tooltip({
  text,
  icon      = "ⓘ",
  width     = 224,
  style,
  className = "",
}) {
  return (
    <span className={`ml-1 group relative cursor-help hover:opacity-80 ${className}`} style={{ color: C.inkMuted }}>
      <span className="text-xs" aria-hidden="true">{icon}</span>
      <span
        role="tooltip"
        className="pointer-events-none absolute left-0 top-5 z-10 rounded-lg p-2 text-xs leading-relaxed opacity-0 group-hover:opacity-100 transition-opacity shadow-xl"
        style={{ width, background: C.bgCard, border: `1px solid ${C.line}`, color: C.inkSoft, ...style }}
      >
        {text}
      </span>
    </span>
  );
})
