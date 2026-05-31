import { memo, useState } from "react";
import { C } from "../../theme";

/**
 * SourceChip — clickable file-source badge molecule.
 *
 * Props:
 *   filename  — displayed label
 *   onClick   — click handler (e.g. open SAS URL)
 *   style     — extra inline styles
 *   className — extra Tailwind classes
 */
export default memo(function SourceChip({
  filename,
  onClick,
  style,
  className = "",
}) {
  const [hovered, setHovered] = useState(false);

  return (
    <button
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      className={`inline-flex items-center gap-1.5 text-xs transition-colors ${className}`}
      style={{
        padding:      "4px 10px",
        background:   hovered ? C.accentBg : C.bgSoft,
        border:       `1px solid ${hovered ? C.accent : C.lineSoft}`,
        borderRadius: 4,
        color:        hovered ? C.accent : C.inkSoft,
        cursor:       onClick ? "pointer" : "default",
        ...style,
      }}
    >
      <span
        style={{
          width:        5,
          height:       5,
          borderRadius: "50%",
          background:   C.accent,
          flexShrink:   0,
          display:      "inline-block",
        }}
      />
      {filename}
    </button>
  );
})
