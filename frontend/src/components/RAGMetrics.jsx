import { C } from "../theme";

// ─────────────────────────────────────────────────────────────────────────────
// MetricChip
// ─────────────────────────────────────────────────────────────────────────────

function MetricChip({ value, label }) {
  return (
    <div
      className="flex-1 text-center rounded"
      style={{
        padding: "8px 12px",
        background: C.bgSoft,
        border: `1px solid ${C.lineSoft}`,
      }}
    >
      <div className="font-mono font-semibold text-sm" style={{ color: C.ink, lineHeight: 1.2 }}>
        {value}
      </div>
      <div className="font-mono uppercase tracking-widest mt-1" style={{ fontSize: 9.5, color: C.inkMuted }}>
        {label}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Export — props unchanged: chunksFound, latency, avgScore
// ─────────────────────────────────────────────────────────────────────────────

export default function RAGMetrics({ chunksFound, latency, avgScore }) {
  return (
    <div className="flex gap-2 mt-5">
      <MetricChip value={chunksFound} label="Chunks found" />
      <MetricChip value={`${latency} ms`} label="Latency" />
      {avgScore != null && <MetricChip value={avgScore} label="Avg score" />}
    </div>
  );
}
