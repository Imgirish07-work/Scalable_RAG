import { C } from "../../theme";

/**
 * TwoPaneLayout — full-height two-column page template.
 *
 * Used by: ModelPlayground, RAGPlayground, PromptPlayground, AgentPlayground
 *
 * Props:
 *   left           — React node for the main (wider) left pane
 *   right          — React node for the config/settings right pane
 *   leftWidth      — CSS flex-basis / width for the left pane  (default: "1fr")
 *   rightWidth     — CSS width for the right pane              (default: 300)
 *   gap            — gap between panes in px                   (default: 0)
 *   rightBorder    — show left border on the right pane        (default: true)
 *   rightStyle     — extra inline styles on the right pane
 *   leftStyle      — extra inline styles on the left pane
 *   style          — extra inline styles on the root wrapper
 *   className      — extra Tailwind classes on the root wrapper
 */
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
      {/* Left pane */}
      <div
        className="flex flex-col min-h-0 overflow-hidden"
        style={{ flex: 1, minWidth: 0, ...leftStyle }}
      >
        {left}
      </div>

      {/* Right pane */}
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
