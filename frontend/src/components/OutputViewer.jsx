import { C } from "../theme";
import { Icon } from "./atoms";
import clockIcon from "../Assets/svg/clock.svg";
import fontIcon from "../Assets/svg/font.svg";
import arrowUpIcon from "../Assets/svg/arrow-up.svg";
import arrowDownIcon from "../Assets/svg/arrow-down.svg";

export default function OutputViewer({ result }) {
  if (!result) return null;
  return (
    <div className="rounded-lg p-4 space-y-3" style={{ background: C.bgDeep, border: `1px solid ${C.line}` }}>
      <div className="flex flex-wrap gap-3 text-xs pb-2" style={{ color: C.inkMuted, borderBottom: `1px solid ${C.line}` }}>
        <span className="flex items-center gap-1"><Icon src={clockIcon} className="w-3 h-3 text-[var(--c-inkMuted)]" />{result.latency_seconds}s</span>
        <span className="flex items-center gap-1"><Icon src={fontIcon} className="w-3 h-3 text-[var(--c-inkMuted)]" />{result.tokens?.total_tokens} tokens</span>
        <span className="flex items-center gap-1"><Icon src={arrowUpIcon} className="w-3 h-3 text-[var(--c-inkMuted)]" />{result.tokens?.prompt_tokens} prompt</span>
        <span className="flex items-center gap-1"><Icon src={arrowDownIcon} className="w-3 h-3 text-[var(--c-inkMuted)]" />{result.tokens?.completion_tokens} completion</span>
      </div>
      <pre className="text-sm whitespace-pre-wrap leading-relaxed" style={{ color: C.ink }}>
        {result.response}
      </pre>
    </div>
  );
}