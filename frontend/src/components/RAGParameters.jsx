import { useState, useRef, useEffect } from "react";
import { Button, Icon } from "./atoms";
import { C } from "../theme";
import gearIcon from "../Assets/svg/gear.svg";
import lightbulbIcon from "../Assets/svg/lightbulb.svg";
import warningIcon from "../Assets/svg/warning.svg";

function getTempMeta(val) {
  const v = parseFloat(val);
  if (v <= 0.3) return {
    label: "Precise", color: "text-blue-400",
    tip: "Very deterministic — best for factual Q&A and structured output.",
    detail: "The model gives the most accurate, consistent answer every time.",
    example: "Best for: answering facts, writing code, summarizing documents.",
    avoid: "Avoid for: creative writing — responses may feel robotic.",
  };
  if (v <= 0.7) return {
    label: "Balanced", color: "text-emerald-400",
    tip: "Balanced creativity and accuracy — good for most tasks",
    detail: "A healthy mix of accuracy and natural language. Great starting point for most tasks.",
    example: "Best for: emails, explanations, customer support, general Q&A.",
    avoid: "Works well for almost everything.",
  };
  if (v <= 1.0) return {
    label: "Creative", color: "text-amber-400",
    tip: "More varied, imaginative responses — good for brainstorming",
    detail: "The model thinks more freely. Each response will feel unique.",
    example: "Best for: brainstorming, blog writing, storytelling, marketing copy.",
    avoid: "Avoid for: precise tasks like coding or factual research.",
  };
  return {
    label: "Wild", color: "text-red-400",
    tip: "Highly random — responses may be unpredictable",
    detail: "Very unexpected word choices. Responses can be surprising or off-topic.",
    example: "Best for: experimental or artistic use only.",
    avoid: "Avoid for: any real task — responses may be confusing.",
  };
}

function getTokenMeta(val) {
  const v = parseInt(val);
  if (v <= 256)  return { label: "Short reply",     color: "text-blue-400",    detail: "A brief reply — like a quick answer or summary.",                        example: "Best for: simple questions, one-liner answers, quick lookups." };
  if (v <= 1024) return { label: "Medium reply",    color: "text-emerald-400", detail: "A standard paragraph or two — enough for most everyday tasks.",           example: "Best for: emails, summaries, explanations, short essays." };
  if (v <= 3000) return { label: "Long reply",      color: "text-amber-400",   detail: "Several detailed paragraphs — good when you need thorough coverage.",     example: "Best for: detailed explanations, reports, long-form content." };
  return           { label: "Very long reply",   color: "text-red-400",    detail: "In-depth response — the model will elaborate as much as possible.",      example: "Best for: essays, research summaries, comprehensive documentation." };
}

function Tooltip({ text }) {
  return (
    <span className="ml-1 group relative cursor-help text-theme-ink-muted hover:text-theme-ink-soft">
      ⓘ
      <span className="pointer-events-none absolute left-0 top-5 z-20 w-56 rounded-lg p-2 text-xs leading-relaxed opacity-0 group-hover:opacity-100 transition-opacity shadow-xl bg-[var(--c-bgCard)] border border-[var(--c-line)] text-[var(--c-ink)]">
        {text}
      </span>
    </span>
  );
}

function InfoPanel({ meta, color }) {
  if (!meta?.detail) return null;
  return (
    <div className="mt-2 rounded-lg p-3 space-y-1.5 text-xs" style={{ background: C.bgDeep, border: `1px solid ${C.line}` }}>
      <p className="leading-relaxed" style={{ color: C.ink }}>{meta.detail}</p>
      <p className={`${color} font-medium flex items-center gap-1`}>
        <Icon src={lightbulbIcon} className="w-3 h-3 text-[var(--c-ink)]" />
        {meta.example}
      </p>
      {meta.avoid && (
        <p className="italic flex items-center gap-1 text-theme-ink-muted">
          <Icon src={warningIcon} className="w-3 h-3 text-[var(--c-inkMuted)]" />
          {meta.avoid}
        </p>
      )}
    </div>
  );
}

const SECTION_CLS = "space-y-2 rounded-lg p-3 border bg-[var(--c-bgCard)] border-[var(--c-line)]";

const TOKEN_PRESETS = [
  { label: "Short",  value: 256  },
  { label: "Medium", value: 1024 },
  { label: "Long",   value: 2048 },
  { label: "Max",    value: 4096 },
];

// button + popover for Temperature and Max Tokens
export default function RAGParameters({
  temperature,
  setTemperature,
  maxTok,
  setMaxTok,
  disabled = false,
  onCommit,
}) {
  const [open, setOpen]   = useState(false);
  const [shown, setShown] = useState(new Set(["temp", "maxTok"]));
  const [pos, setPos]     = useState({ top: 0, right: 0 });
  const btnRef            = useRef(null);

  const tempMeta  = getTempMeta(temperature);
  const tokenMeta = getTokenMeta(maxTok);

  const reveal = key => {
    if (!disabled) setShown(prev => new Set([...prev, key]));
  };

  const handleToggle = () => {
    if (!open && btnRef.current) {
      const rect = btnRef.current.getBoundingClientRect();
      setPos({
        top:       rect.bottom + 8,
        right:     window.innerWidth - rect.right,
        maxHeight: window.innerHeight - rect.bottom - 24,
      });
    }
    setOpen(prev => !prev);
  };

  // close on window resize to avoid stale position
  useEffect(() => {
    if (!open) return;
    const close = () => setOpen(false);
    window.addEventListener("resize", close);
    return () => window.removeEventListener("resize", close);
  }, [open]);

  return (
    <div>
      <Button
        ref={btnRef}
        onClick={handleToggle}
        variant={open ? "primary" : "outline"}
        size="sm"
      >
        <Icon src={gearIcon} className="w-3.5 h-3.5 text-[var(--c-ink)]" />
        Parameters
      </Button>

      {open && (
        <>
          <div onClick={() => setOpen(false)} className="fixed inset-0 z-40" />

          {/* fixed so overflow:hidden parents don't clip it */}
          <div
            style={{ top: pos.top, right: pos.right, maxHeight: pos.maxHeight, background: C.bgDeep, border: `1px solid ${C.line}` }}
            className="fixed z-50 w-96 overflow-y-auto overflow-x-hidden rounded-xl p-5 shadow-2xl
                        [&::-webkit-scrollbar]:w-1 [&::-webkit-scrollbar-track]:bg-transparent
                        [&::-webkit-scrollbar-thumb]:bg-[var(--c-bgSoft)] [&::-webkit-scrollbar-thumb]:rounded-full"
          >
            <h3 className="text-sm font-semibold mb-4 flex items-center gap-2" style={{ color: C.ink }}>
              <Icon src={gearIcon} className="w-3.5 h-3.5 text-[var(--c-ink)]" />
              Parameters
            </h3>

            <div className="space-y-3">

              <div className={SECTION_CLS}>
                <div className="flex items-center justify-between">
                  <label className="text-xs font-medium text-theme-ink">
                    Temperature
                    <Tooltip text="Controls randomness. Low = accurate & predictable. High = creative & unpredictable." />
                  </label>
                  <span className={`text-xs font-bold px-2 py-0.5 rounded-full bg-[var(--c-bgSoft)] ${tempMeta.color}`}>
                    {tempMeta.label} · {temperature}
                  </span>
                </div>
                <input
                  type="range" min="0" max="2" step="0.1"
                  value={temperature}
                  onChange={e => { setTemperature(parseFloat(e.target.value)); reveal("temp"); }}
                  onMouseUp={e => onCommit({ temperature: parseFloat(e.target.value) })}
                  onTouchEnd={e => onCommit({ temperature: parseFloat(e.target.value) })}
                  disabled={disabled}
                  className="w-full h-1.5 rounded cursor-pointer accent-indigo-500 disabled:cursor-not-allowed"
                />
                <div className="flex justify-between text-xs text-theme-ink-muted">
                  <span>0 — Precise</span>
                  <span>0.8 — Creative</span>
                  <span>1.1 — Wild</span>
                </div>
                {shown.has("temp")
                  ? <InfoPanel meta={tempMeta} color={tempMeta.color} />
                  : <p className="text-xs italic text-theme-ink-muted">{tempMeta.tip}</p>}
              </div>

              <div className={SECTION_CLS}>
                <div className="flex items-center justify-between">
                  <label className="text-xs font-medium text-theme-ink">
                    Max Tokens
                    <Tooltip text="Controls the maximum response length. 1 token ≈ 4 characters." />
                  </label>
                  <span className={`text-xs font-bold px-2 py-0.5 rounded-full bg-[var(--c-bgSoft)] ${tokenMeta.color}`}>
                    {tokenMeta.label} · {maxTok}
                  </span>
                </div>
                <input
                  type="range" min="64" max="4096" step="64"
                  value={maxTok}
                  onChange={e => { setMaxTok(parseInt(e.target.value)); reveal("maxTok"); }}
                  onMouseUp={e => onCommit({ max_tokens: parseInt(e.target.value) })}
                  onTouchEnd={e => onCommit({ max_tokens: parseInt(e.target.value) })}
                  disabled={disabled}
                  className="w-full h-1.5 rounded cursor-pointer accent-indigo-500 disabled:cursor-not-allowed"
                />
                <div className="flex justify-between text-xs text-theme-ink-muted">
                  <span>64 — Short</span>
                  <span>1024 — Medium</span>
                  <span>4096 — Max</span>
                </div>
                <div className="flex gap-2 mt-1">
                  {TOKEN_PRESETS.map(({ label, value }) => (
                    <Button
                      key={value}
                      onClick={() => { setMaxTok(value); reveal("maxTok"); onCommit({ max_tokens: value }); }}
                      disabled={disabled}
                      variant={maxTok === value ? "primary" : "outline"}
                      size="sm"
                      className="flex-1 flex-col"
                    >
                      <span>{label}</span>
                      <span className="text-[10px] opacity-60">{value}</span>
                    </Button>
                  ))}
                </div>
                {shown.has("maxTok") && <InfoPanel meta={tokenMeta} color={tokenMeta.color} />}
              </div>

            </div>
          </div>
        </>
      )}
    </div>
  );
}
