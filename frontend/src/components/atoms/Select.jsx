import { memo } from "react";
import { C } from "../../theme";

export default memo(function Select({
  value,
  onChange,
  options   = [],
  disabled  = false,
  width     = 150,
  style,
  className = "",
  "aria-label": ariaLabel,
}) {
  return (
    <select
      value={value}
      onChange={e => onChange(e.target.value)}
      disabled={disabled}
      aria-label={ariaLabel}
      className={`text-xs rounded outline-none focus:ring-1 focus:ring-[color:var(--c-accent)] focus:border-[color:var(--c-accent)] ${disabled ? "cursor-not-allowed" : "cursor-pointer"} ${className}`}
      style={{
        width,
        minWidth: width,
        padding: "4px 22px 4px 8px",
        border: `1px solid ${C.lineSoft}`,
        background: `var(--c-bgInput) url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='8' height='8' viewBox='0 0 8 8'%3E%3Cpath d='M1 2.5l3 3 3-3' stroke='%239ca3af' fill='none' stroke-width='1'/%3E%3C/svg%3E") no-repeat right 6px center`,
        color: C.ink,
        accentColor: "rgb(6 182 212)",
        appearance: "none",
        WebkitAppearance: "none",
        opacity: disabled ? 0.5 : 1,
        ...style,
      }}
    >
      {options.map(o => (
        <option key={o.value} value={o.value} style={{ backgroundColor: "var(--c-bgInput)" }}>
          {o.label}
        </option>
      ))}
    </select>
  );
})
