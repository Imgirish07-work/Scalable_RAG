import { memo } from "react";
import { C } from "../../theme";

export default memo(function Input({
  type       = "text",
  value,
  onChange,
  onBlur,
  placeholder,
  hasError   = false,
  disabled   = false,
  style,
  className  = "",
  id,
  ...rest
}) {
  return (
    <input
      id={id}
      type={type}
      value={value}
      onChange={onChange}
      onBlur={onBlur}
      placeholder={placeholder}
      disabled={disabled}
      aria-invalid={hasError || undefined}
      aria-disabled={disabled || undefined}
      className={`w-full rounded-lg px-3 py-2.5 text-sm outline-none transition-colors
        focus:ring-2
        ${hasError
          ? "border-red-500/60 focus:border-red-500 focus:ring-red-500/20"
          : "border-cyan-500/30 focus:border-cyan-400 focus:ring-cyan-500/20"}
        ${disabled ? "opacity-50 cursor-not-allowed" : ""}
        ${className}`}
      style={{
        background: C.accentBg,
        border:     `1px solid ${hasError ? C.danger : C.accentBorder}`,
        color:      C.ink,
        ...style,
      }}
      {...rest}
    />
  );
})
