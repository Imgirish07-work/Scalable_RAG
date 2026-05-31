import { memo } from "react";
import { C } from "../../theme";
import Skeleton from "../atoms/Skeleton";
import Spinner from "../atoms/Spinner";

/**
 * ConfigRow — label + children layout for configuration panels.
 *
 * Props:
 *   label     — row label text
 *   help      — tooltip / title text on hover
 *   border    — show bottom border (default: true)
 *   loading   — initial-load placeholder: replaces the control with a Skeleton
 *   saving    — update-in-flight indicator: small spinner appears next to the
 *               label while the change is being persisted. Distinct from
 *               `loading` because the control still shows its current value.
 *   children  — control element(s) on the right
 *   style     — extra inline styles
 *   className — extra Tailwind classes
 */
export default memo(function ConfigRow({
  label,
  help,
  border = true,
  loading = false,
  saving = false,
  children,
  style,
  className = "",
}) {
  return (
    <div
      className={`grid items-center gap-3 py-2.5 ${className}`}
      style={{
        gridTemplateColumns: "1fr auto",
        borderBottom: border ? `1px solid ${C.lineSoft}` : "none",
        ...style,
      }}
    >
      <div className="flex items-center gap-1.5 text-xs" style={{ color: C.ink }} title={help}>
        {label}
        {saving && <Spinner size={12} />}
      </div>
      <div className="flex justify-end items-center" style={{ minHeight: 26 }}>
        {loading ? <Skeleton width={150} height={26} /> : children}
      </div>
    </div>
  );
})
