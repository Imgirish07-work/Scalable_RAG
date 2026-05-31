import { C } from "../theme";

export default function TokenUsageBadge({ usage }) {
  const pct       = usage.daily_limit > 0 ? Math.min(100, Math.round((usage.daily_used / usage.daily_limit) * 100)) : 0;
  const isDanger  = pct >= 90;
  const isWarning = pct >= 70;
  const barBg     = isDanger ? "bg-red-500"        : isWarning ? "bg-amber-400"   : "bg-indigo-500";
  const numCol    = isDanger ? "text-red-400"       : isWarning ? "text-amber-400" : "text-indigo-400";
  return (
    <div className="rounded-xl p-3 min-w-[220px]" style={{ background: C.bgCard, border: `1px solid ${isDanger ? "rgba(239,68,68,0.3)" : isWarning ? "rgba(245,158,11,0.3)" : C.line}` }}>
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-semibold uppercase tracking-wide" style={{ color: C.inkMuted }}>Daily Tokens</span>
        <span className={`text-xs font-bold tabular-nums ${numCol}`}>{pct}% used</span>
      </div>
      <div className="h-1.5 rounded-full overflow-hidden mb-2.5" style={{ background: C.bgSoft }}>
        <div className={`h-full rounded-full transition-all duration-700 ${barBg}`} style={{ width: `${pct}%` }} />
      </div>
      <div className="flex items-center justify-between">
        <span className="text-xs" style={{ color: C.inkMuted }}>
          <span className={`font-bold ${numCol}`}>{usage.daily_used.toLocaleString()}</span>
          <span style={{ color: C.inkMuted }}> / {usage.daily_limit.toLocaleString()}</span>
        </span>
        <span className={`text-xs font-semibold ${numCol}`}>{usage.remaining.toLocaleString()} left</span>
      </div>
    </div>
  );
}

export function TokenUsageSkeleton() {
  return (
    <div className="rounded-xl p-3 min-w-[220px] animate-pulse" style={{ background: C.bgCard, border: `1px solid ${C.line}` }}>
      <div className="flex justify-between mb-2">
        <div className="h-3 rounded w-24" style={{ background: C.bgSoft }} />
        <div className="h-3 rounded w-12" style={{ background: C.bgSoft }} />
      </div>
      <div className="h-1.5 rounded-full mb-2.5" style={{ background: C.bgSoft }} />
      <div className="flex justify-between">
        <div className="h-2.5 rounded w-20" style={{ background: C.bgSoft }} />
        <div className="h-2.5 rounded w-14" style={{ background: C.bgSoft }} />
      </div>
    </div>
  );
}
