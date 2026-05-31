import { memo } from "react";
import { C } from "../../theme";

/**
 * EmptyState — centred placeholder shown when a list or panel is empty.
 *
 * Props:
 *   icon      — emoji or React node displayed large above the text
 *   title     — primary label (string)
 *   desc      — secondary description (string, optional)
 *   action    — React node (e.g. a Button) rendered below the desc (optional)
 *   style     — extra inline styles on the root wrapper
 *   className — extra Tailwind classes on the root wrapper
 */
export default memo(function EmptyState({ icon, title, desc, action, style, className = "" }) {
  return (
    <div
      className={`flex flex-col items-center justify-center text-center gap-3 ${className}`}
      style={{ padding: "48px 24px", ...style }}
    >
      {icon && (
        <div style={{ fontSize: 32, lineHeight: 1 }}>
          {icon}
        </div>
      )}
      <p className="text-sm font-medium" style={{ color: C.inkSoft }}>
        {title}
      </p>
      {desc && (
        <p className="text-xs leading-relaxed" style={{ color: C.inkMuted, maxWidth: 280 }}>
          {desc}
        </p>
      )}
      {action && (
        <div className="mt-1">{action}</div>
      )}
    </div>
  );
})
