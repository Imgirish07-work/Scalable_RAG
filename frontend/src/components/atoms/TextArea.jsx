import { memo } from "react";
import { C } from "../../theme";

export default memo(function TextArea({
  label,
  value,
  onChange,
  rows        = 4,
  placeholder,
  disabled    = false,
  mono        = false,
  resize      = "vertical",
  style,
  className   = "",
  id,
}) {
  return (
    <div className="space-y-1">
      {label && (
        <label htmlFor={id} className="text-xs font-medium" style={{ color: C.inkSoft }}>
          {label}
        </label>
      )}
      <textarea
        id={id}
        value={value}
        onChange={e => onChange(e.target.value)}
        aria-label={label || placeholder}
        rows={rows}
        placeholder={placeholder}
        disabled={disabled}
        className={`w-full rounded-lg px-3 py-2 text-sm focus:outline-none
          focus:border-cyan-500/40 focus:ring-1 focus:ring-cyan-500/20
          placeholder-theme-ink-muted
          ${mono ? "font-mono" : ""}
          ${disabled ? "opacity-50 cursor-not-allowed" : ""}
          ${className}`}
        style={{
          background: C.bgInput,
          border:     `1px solid ${C.line}`,
          color:      C.ink,
          resize,
          ...style,
        }}
      />
    </div>
  );
})
