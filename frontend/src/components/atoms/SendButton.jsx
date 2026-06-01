import { memo, useState } from "react";
import { C } from "../../theme";

export default memo(function SendButton({ onClick, disabled }) {
  const [hovered, setHovered] = useState(false);
  // theme-aware arrow color: accent for dark, ink for light
  const isLight = document?.documentElement?.dataset?.theme?.startsWith('light');
  const arrowColor = isLight ? C.ink : "white";
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      aria-disabled={disabled}
      title="Send (Enter)"
      aria-label="Send message"
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      className="flex items-center justify-center flex-shrink-0 rounded-lg transition-all"
      style={{
        width: 34,
        height: 34,
        background: disabled
          ? C.lineCard
          : hovered
          ? `linear-gradient(to right, ${C.accent}, ${C.accentHover})`
          : `linear-gradient(to right, ${C.accentHover}, ${C.accent})`,
        color: arrowColor,
        border: "none",
        cursor: disabled ? "not-allowed" : "pointer",
        opacity: disabled ? 0.35 : 1,
        boxShadow: disabled
          ? "none"
          : hovered
          ? `0 0 16px ${C.accentGlow}`
          : `0 0 10px ${C.accentGlow}`,
        transition: "background 0.15s, opacity 0.15s, box-shadow 0.15s",
      }}
    >
      <svg
        viewBox="0 0 24 24"
        fill="none"
        stroke={arrowColor}
        strokeWidth="2.2"
        strokeLinecap="round"
        strokeLinejoin="round"
        style={{ width: 14, height: 14 }}
      >
        <line x1="5" y1="12" x2="19" y2="12" />
        <polyline points="12 5 19 12 12 19" />
      </svg>
    </button>
  );
})
