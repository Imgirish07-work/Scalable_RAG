import { memo } from "react";
import { C } from "../../theme";

export default memo(function StopButton({ onClick, style }) {
  return (
    <button
      onClick={onClick}
      title="Stop generation"
      aria-label="Stop generation"
      className="flex items-center justify-center flex-shrink-0 rounded-lg transition-colors"
      style={{
        width: 34,
        height: 34,
        background: C.danger,
        border: "none",
        cursor: "pointer",
        color: "white",
        ...style,
      }}
    >
      <span
        style={{
          width: 10,
          height: 10,
          background: "currentColor",
          borderRadius: 2,
          display: "inline-block",
        }}
      />
    </button>
  );
})
