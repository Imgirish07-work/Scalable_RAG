import { memo } from "react";
import Icon from "./Icon";
import checkIcon from "../../Assets/svg/checkmark.svg";

/**
 * FieldSuccess — inline validation success message atom.
 *
 * Props:
 *   show      — renders only when truthy
 *   message   — text to display (default: "Looks good!")
 *   style     — extra inline styles
 *   className — extra Tailwind classes
 */
export default memo(function FieldSuccess({
  show,
  message   = "Looks good!",
  style,
  className = "",
}) {
  if (!show) return null;
  return (
    <p
      role="status"
      className={`flex items-center gap-1 text-[11px] text-emerald-400 mt-1 ${className}`}
      style={style}
    >
      <Icon src={checkIcon} className="w-3 h-3 text-[var(--c-ok)]" /> {message}
    </p>
  );
})
