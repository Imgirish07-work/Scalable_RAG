import { memo } from "react";
import { C } from "../../theme";

/**
 * Slider — reusable range input with cyan thumb and live value display.
 *
 * Props:
 *   min, max, step   — range config
 *   value            — controlled value (string or number)
 *   onChange(val)    — called on every tick with the raw string value
 *   onCommit(val)    — optional, called on mouseUp / touchEnd (use for API saves)
 *   disabled         — disables interaction
 *   trackWidth       — number (px) for fixed width; omit to use flex: 1
 *   formatValue(val) — display formatter; defaults to String
 */
export default memo(function Slider({
  min, max, step,
  value,
  onChange,
  onCommit,
  disabled = false,
  trackWidth,
  formatValue = String,
}) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        aria-valuemin={Number(min)}
        aria-valuemax={Number(max)}
        aria-valuenow={Number(value)}
        onChange={e => onChange(e.target.value)}
        onMouseUp={onCommit ? e => onCommit(e.target.value) : undefined}
        onTouchEnd={onCommit ? e => onCommit(e.target.value) : undefined}
        disabled={disabled}
        style={{
          WebkitAppearance: "none",
          appearance: "none",
          height: "2px",
          background: C.line,
          outline: "none",
          cursor: disabled ? "not-allowed" : "pointer",
          ...(trackWidth ? { width: trackWidth } : { flex: 1 }),
        }}
        className="disabled:opacity-40
          [&::-webkit-slider-thumb]:appearance-none
          [&::-webkit-slider-thumb]:w-3.5 [&::-webkit-slider-thumb]:h-3.5
          [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-[var(--c-accent)]
          [&::-webkit-slider-thumb]:cursor-pointer [&::-webkit-slider-thumb]:shadow-[0_0_6px_var(--c-accentGlow)]
          [&::-moz-range-thumb]:w-3.5 [&::-moz-range-thumb]:h-3.5
          [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:bg-[var(--c-accent)]
          [&::-moz-range-thumb]:border-0 [&::-moz-range-thumb]:cursor-pointer"
      />
      <span
        style={{
          color: C.accentHover,
          fontSize: 12,
          fontWeight: 600,
          fontVariantNumeric: "tabular-nums",
          minWidth: 32,
          textAlign: "right",
        }}
      >
        {formatValue(value)}
      </span>
    </div>
  );
})
