import { memo } from "react";
import IconButton from "../atoms/IconButton";

function IconCopy({ active }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke={active ? "#3fb950" : "currentColor"}
      strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ width: 13, height: 13 }}>
      <rect x="9" y="9" width="13" height="13" rx="2" />
      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
    </svg>
  );
}

function IconEdit() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ width: 13, height: 13 }}>
      <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
      <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
    </svg>
  );
}

function IconRetry() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ width: 13, height: 13 }}>
      <polyline points="1 4 1 10 7 10" />
      <path d="M3.51 15a9 9 0 1 0 .49-4.5" />
    </svg>
  );
}

export default memo(function MsgActionBar({
  visible,
  copiedId,
  msgId,
  onCopy,
  onEdit,
  onRetry,
  disabled  = false,
  style,
  className = "",
}) {
  return (
    <div
      className={`flex items-center gap-0.5 transition-opacity duration-150 ${className}`}
      style={{
        opacity:       visible && !disabled ? 1 : 0,
        pointerEvents: visible && !disabled ? "auto" : "none",
        ...style,
      }}
    >
      {onCopy && (
        <IconButton title="Copy" onClick={onCopy}>
          <IconCopy active={copiedId === msgId} />
        </IconButton>
      )}
      {onEdit && (
        <IconButton title="Edit" onClick={onEdit}>
          <IconEdit />
        </IconButton>
      )}
      {onRetry && (
        <IconButton title="Retry" onClick={onRetry}>
          <IconRetry />
        </IconButton>
      )}
    </div>
  );
})
