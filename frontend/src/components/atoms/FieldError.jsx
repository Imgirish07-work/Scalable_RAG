import { memo } from "react";
import Icon from "./Icon";
import warningIcon from "../../Assets/svg/warning.svg";

/**
 * FieldError — inline validation error message atom.
 *
 * Props:
 *   message   — error text; renders nothing when falsy
 *   style     — extra inline styles
 *   className — extra Tailwind classes
 */
export default memo(function FieldError({ message, style, className = "" }) {
  if (!message) return null;
  return (
    <p
      role="alert"
      className={`flex items-center gap-1 text-[11px] text-red-400 mt-1 ${className}`}
      style={style}
    >
      <Icon src={warningIcon} className="w-3 h-3 text-[var(--c-danger)]" /> {message}
    </p>
  );
})
