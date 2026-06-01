import { memo } from "react";

export default memo(function Avatar({
  name      = "",
  size      = 28,
  bg        = "#0d9488",
  style,
  className = "",
}) {
  const initial = (name || "?").charAt(0).toUpperCase();

  return (
    <div
      role="img"
      aria-label={`Avatar for ${name || "unknown user"}`}
      className={`flex items-center justify-center rounded-full text-white font-bold shrink-0 ${className}`}
      style={{
        width:      size,
        height:     size,
        fontSize:   size * 0.42,
        background: bg,
        ...style,
      }}
    >
      {initial}
    </div>
  );
})
