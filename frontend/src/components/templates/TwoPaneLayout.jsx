import { C } from "../../theme";

export default function TwoPaneLayout({
  left,
  right,
  leftWidth   = "1fr",
  rightWidth  = 300,
  gap         = 0,
  rightBorder = true,
  rightStyle,
  leftStyle,
  style,
  className   = "",
}) {
  return (
    <div
      className={`flex h-full overflow-hidden ${className}`}
      style={{ background: C.bg, gap, ...style }}
    >
      <div
        className="flex flex-col min-h-0 overflow-hidden"
        style={{ flex: 1, minWidth: 0, ...leftStyle }}
      >
        {left}
      </div>

      {right && (
        <div
          className="flex flex-col min-h-0 overflow-hidden flex-shrink-0"
          style={{
            width:      rightWidth,
            borderLeft: rightBorder ? `1px solid ${C.lineSoft}` : "none",
            background: C.bgPanel,
            ...rightStyle,
          }}
        >
          {right}
        </div>
      )}
    </div>
  );
}
