import { memo } from "react";
import { RefreshCw, X } from "lucide-react";
import { ProgressBar } from "../atoms";
import { C } from "../../theme";

const ACTIVE_STATUSES = new Set([
  "queued", "uploading", "finalizing", "processing",
]);

function resolvePercent(status, phase, progress, chunksProcessed, chunksTotal) {
  if (status === "uploading") return clampPct(progress);
  if (phase === "embedding" && chunksTotal > 0) {
    return clampPct(Math.round((chunksProcessed / chunksTotal) * 100));
  }
  return null;
}

function clampPct(n) {
  if (typeof n !== "number" || Number.isNaN(n)) return 0;
  return Math.min(100, Math.max(0, n));
}

function statusColor(status) {
  if (status === "failed") return "var(--c-textError)";
  return C.inkSoft;
}

export default memo(function UploadJobRow({
  filename,
  status,
  phase,
  progress,
  chunksProcessed = 0,
  chunksTotal = 0,
  message,
  onCancel,
  onRetry,
}) {
  const isActive = ACTIVE_STATUSES.has(status);
  const isFailed = status === "failed";
  const pct = resolvePercent(status, phase, progress, chunksProcessed, chunksTotal);
  const showBar = isActive;

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 4,
        padding: "6px 10px",
        background: C.bgSoft,
        borderRadius: 8,
        border: `1px solid ${isFailed ? "var(--c-textError)" : C.lineSoft}`,
        minWidth: 0,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 8, minWidth: 0 }}>
        <span
          style={{
            fontSize: 12,
            fontWeight: 500,
            color: C.ink,
            whiteSpace: "nowrap",
            overflow: "hidden",
            textOverflow: "ellipsis",
            flex: 1,
            minWidth: 0,
          }}
          title={filename}
        >
          {filename}
        </span>
        {pct != null && (
          <span style={{ fontSize: 11, fontFamily: "monospace", color: C.accent }}>
            {pct}%
          </span>
        )}
        {isFailed && onRetry && (
          <IconAction title="Retry" onClick={onRetry}>
            <RefreshCw size={12} />
          </IconAction>
        )}
        {onCancel && (
          <IconAction title="Cancel" onClick={onCancel}>
            <X size={13} />
          </IconAction>
        )}
      </div>

      {showBar && (
        <ProgressBar
          value={pct != null ? pct : undefined}
          indeterminate={pct == null}
          height={4}
        />
      )}

      {message && (
        <span
          style={{
            fontSize: 11,
            color: statusColor(status),
            whiteSpace: "nowrap",
            overflow: "hidden",
            textOverflow: "ellipsis",
          }}
          title={message}
        >
          {message}
        </span>
      )}
    </div>
  );
});

function IconAction({ children, title, onClick }) {
  return (
    <button
      type="button"
      title={title}
      onClick={onClick}
      style={{
        display: "grid",
        placeItems: "center",
        width: 20,
        height: 20,
        border: "none",
        background: "transparent",
        color: C.inkSoft,
        cursor: "pointer",
        borderRadius: 4,
        padding: 0,
        flexShrink: 0,
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.background = C.lineSoft;
        e.currentTarget.style.color = C.ink;
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.background = "transparent";
        e.currentTarget.style.color = C.inkSoft;
      }}
    >
      {children}
    </button>
  );
}
