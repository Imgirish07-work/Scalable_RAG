import { memo } from "react";
import { C } from "../../theme";
import lightbulbIcon from "../../Assets/svg/lightbulb.svg";
import warningIcon  from "../../Assets/svg/warning.svg";
import checkmarkIcon from "../../Assets/svg/checkmark.svg";

export default memo(function ParamInfoPanel({ meta, color, style, className = "" }) {
  if (!meta?.detail) return null;

  const avoidIcon = meta.avoid?.startsWith("Avoid") ? warningIcon : checkmarkIcon;

  return (
    <div
      className={`mt-2 rounded-lg p-3 space-y-1.5 text-xs transition-all ${className}`}
      style={{ background: C.bgDeep, border: `1px solid ${C.line}`, ...style }}
    >
      <p className="leading-relaxed" style={{ color: C.ink }}>{meta.detail}</p>
      <p className={`${color} font-medium flex items-center gap-1`}>
        <img src={lightbulbIcon} alt="" width={12} height={12} style={{ filter: "invert(1)", opacity: 0.8 }} />
        {meta.example}
      </p>
      {meta.avoid && (
        <p className="leading-relaxed flex items-center gap-1" style={{ color: C.ink }}>
          <img src={avoidIcon} alt="" width={12} height={12} style={{ filter: "invert(1)", opacity: 0.7 }} />
          {meta.avoid}
        </p>
      )}
    </div>
  );
})
