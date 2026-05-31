import { memo } from "react";
import { C } from "../../theme";

/**
 * PanelHeader — titled section header with an optional action link/button.
 *
 * Props:
 *   title      — section label (string or node)
 *   action     — label for the right-side action (string, optional)
 *   onAction   — click handler for the action
 *   border     — show a bottom border (default: false)
 *   style      — extra inline styles on the root wrapper
 *   className  — extra Tailwind classes
 *   children   — rendered to the right of the title (overrides action/onAction)
 */
export default memo(function PanelHeader({
  title,
  action,
  onAction,
  border    = false,
  style,
  className = "",
  children,
}) {
  return (
    <div
      className={`flex items-center justify-between flex-shrink-0 ${className}`}
      style={{
        padding:      "10px 16px 8px",
        borderBottom: border ? `1px solid ${C.lineSoft}` : "none",
        ...style,
      }}
    >
      <span className="text-sm font-semibold" style={{ color: C.ink }}>
        {title}
      </span>

      {children ?? (
        action && (
          <button
            onClick={onAction}
            className="text-xs font-medium hover:underline"
            style={{
              background: "none",
              border:     "none",
              cursor:     "pointer",
              color:      C.accent,
              padding:    0,
            }}
          >
            {action}
          </button>
        )
      )}
    </div>
  );
})
