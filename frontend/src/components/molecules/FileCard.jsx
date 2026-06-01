import { memo } from "react";
import { Icon } from "../atoms";
import closeIcon from "../../Assets/svg/close.svg";
import { C } from "../../theme";

function getExt(filename) {
  return filename.split(".").pop().toLowerCase();
}

export default memo(function FileCard({ file, onRemove, style, className = "" }) {
  const ext = getExt(file.filename).toUpperCase().slice(0, 4);

  return (
    <div
      className={`flex flex-col flex-shrink-0 rounded-lg ${className}`}
      style={{
        width: 80,
        height: 90,
        border: `1px solid ${C.lineSoft}`,
        background: C.bgCard,
        padding: "8px 8px 6px",
        ...style,
      }}
    >
      <p
        className="flex-1 text-[10px] font-medium leading-tight min-h-0"
        style={{
          color: C.ink,
          display: "-webkit-box",
          WebkitLineClamp: 3,
          WebkitBoxOrient: "vertical",
          overflow: "hidden",
          wordBreak: "break-all",
        }}
        title={file.filename}
      >
        {file.filename}
      </p>

      <div className="flex items-center justify-between mt-2 flex-shrink-0">
        <span
          className="text-[9px] font-medium rounded px-1.5 py-0.5"
          style={{ background: C.accentBg, color: C.accent, letterSpacing: "0.03em" }}
        >
          {ext}
        </span>
        <button
          onClick={() => onRemove(file.filename)}
          title="Remove from session"
          className="leading-none transition-colors"
          style={{ background: "none", border: "none", cursor: "pointer", color: C.inkMuted, padding: "1px 2px", fontSize: 9 }}
          onMouseEnter={e => { e.currentTarget.style.color = "#f87171"; }}
          onMouseLeave={e => { e.currentTarget.style.color = C.inkMuted; }}
        >
          <Icon src={closeIcon} className="w-2.5 h-2.5 text-[var(--c-ink)]" />
        </button>
      </div>
    </div>
  );
})
