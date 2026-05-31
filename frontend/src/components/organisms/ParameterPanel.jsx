import { useState, useRef } from "react";
import { Alert, Tooltip, Slider } from "../atoms";
import { ParamInfoPanel } from "../molecules";
import { C } from "../../theme";

/**
 * Shared metadata helpers for parameter controls.
 * Re-exported so consumers (pages, other organisms) can use them directly.
 */
export function getTempMeta(val) {
  const v = parseFloat(val);

  if (v <= 0.3) return { label: "Precise", color: "text-blue-400", tip: "Very deterministic — best for factual Q&A, code generation, structured output", detail: "The AI acts like a calculator — it gives the most accurate, consistent answer every time.", example: "Best for: answering facts, writing code, summarizing documents, filling forms.", avoid: "Avoid for: creative writing — responses may feel robotic." };

  if (v <= 0.7) return { label: "Balanced", color: "text-emerald-400", tip: "Balanced creativity and accuracy — good for most tasks", detail: "A healthy mix of accuracy and natural language. Great starting point for most tasks.", example: "Best for: emails, explanations, customer support replies, general Q&A.", avoid: "Works well for almost everything." };

  if (v <= 1.0) return { label: "Creative", color: "text-amber-400", tip: "More varied, imaginative responses — good for brainstorming", detail: "The AI thinks more freely and imaginatively. Each response will feel unique.", example: "Best for: brainstorming, blog writing, storytelling, marketing copy.", avoid: "Avoid for: precise tasks like coding or factual research." };

  return { label: "Wild", color: "text-red-400", tip: "Highly random — responses may be unpredictable", detail: "Very unexpected word choices. Responses can be surprising or off-topic.", example: "Best for: experimental or artistic use only.", avoid: "Avoid for: any real task — responses may be confusing." };
}

export function getTopPMeta(val) {
  const v = parseFloat(val);

  if (v <= 0.5) return { label: "Focused", color: "text-blue-400", tip: "Only the most probable words are sampled", detail: "The AI only picks from its most confident word choices — safe and clear.", example: "Best for: technical writing, instructions, structured output.", avoid: "Avoid for: creative tasks — writing may feel dry." };

  if (v <= 0.85) return { label: "Moderate", color: "text-emerald-400", tip: "Balanced vocabulary diversity — recommended for most tasks", detail: "Clear and coherent but with natural variety. The sweet spot for most use cases.", example: "Best for: most tasks — emails, explanations, summaries, Q&A.", avoid: "Recommended setting for nearly all users." };

  return { label: "Diverse", color: "text-amber-400", tip: "Wide word selection — richer vocabulary", detail: "Responses feel rich and expressive. Great for creative writing.", example: "Best for: poetry, storytelling, creative brainstorming.", avoid: "Avoid for: factual or structured tasks." };
}

export function getTokenMeta(val) {
  const v = parseInt(val);
  
  if (v <= 256)  return { label: "Short reply", color: "text-blue-400", detail: "A brief reply — like a text message or quick answer.", example: "Best for: simple questions, one-liner answers." };

  if (v <= 1024) return { label: "Medium reply", color: "text-emerald-400", detail: "A standard paragraph or two — enough for most tasks.", example: "Best for: emails, summaries, explanations." };

  if (v <= 3000) return { label: "Long reply", color: "text-amber-400", detail: "Several detailed paragraphs.", example: "Best for: detailed explanations, reports." };
  return { label: "Very long reply", color: "text-red-400", detail: "In-depth, thorough response.", example: "Best for: essays, research summaries, documentation." };
}

const TOKEN_PRESETS = [
  { label: "Short",  value: 256  },
  { label: "Medium", value: 1024 },
  { label: "Long",   value: 2048 },
  { label: "Max",    value: 4096 },
];

/**
 * ParameterPanel — unified temperature / max-tokens / top-p controls organism.
 *
 * Props:
 *   temp / setTemp        — temperature value + setter
 *   maxTok / setMaxTok    — max tokens value + setter
 *   topP / setTopP        — top-p value + setter (optional — omit to hide)
 *   disabled              — lock controls during generation
 *   onCommit(patch)       — optional, called on mouseUp for backend sync
 *   style / className     — extra styles
 */
export default function ParameterPanel({
  temp, setTemp,
  maxTok, setMaxTok,
  topP, setTopP,
  disabled  = false,
  onCommit,
  style,
  className = "",
}) {
  const [shown, setShown] = useState(new Set(["temp", "maxTok", "topP"]));
  const tempRef   = useRef(null);
  const maxTokRef = useRef(null);
  const topPRef   = useRef(null);
  const refs = { temp: tempRef, maxTok: maxTokRef, topP: topPRef };

  const reveal = (key) => {
    if (!disabled) {
      setShown(prev => new Set([...prev, key]));
      setTimeout(() => refs[key]?.current?.scrollIntoView({ behavior: "smooth", block: "nearest" }), 50);
    }
  };

  const commit = (patch) => onCommit?.(patch);

  const tempMeta  = getTempMeta(temp);
  const tokenMeta = getTokenMeta(maxTok);
  const topPMeta  = topP != null ? getTopPMeta(topP) : null;

  const sectionCls = `space-y-2 rounded-lg p-3 border bg-[var(--c-bg)] border-[var(--c-line)] ${disabled ? "opacity-50 pointer-events-none select-none" : ""}`;

  return (
    <div className={`space-y-3 ${className}`} style={style}>
      {disabled && <Alert type="warning" message="Parameters locked while generating…" />}

      {/* Temperature */}
      <div ref={tempRef} className={sectionCls}>
        <div className="flex items-center justify-between">
          <label className="text-xs font-medium text-theme-ink">
            Temperature
            <Tooltip text="Controls randomness. Low = accurate & predictable. High = creative & unpredictable." />
          </label>
          <span className={`text-xs font-bold px-2 py-0.5 rounded-full bg-[var(--c-bgCard)] ${tempMeta.color}`}>
            {tempMeta.label} · {temp}
          </span>
        </div>
        <Slider
          min={0} max={2} step={0.1}
          value={temp}
          onChange={v => { setTemp(v); reveal("temp"); }}
          onCommit={v => commit({ temperature: parseFloat(v) })}
          disabled={disabled}
          formatValue={() => ""}
        />
        <div className="flex justify-between text-xs text-theme-ink-muted">
          <span>0 — Precise</span><span>0.8 — Creative</span><span>1.1 — Wild</span>
        </div>
        {shown.has("temp")
          ? <ParamInfoPanel meta={tempMeta} color={tempMeta.color} />
          : <p className="text-xs italic text-theme-ink-muted">{tempMeta.tip}</p>}
      </div>

      {/* Max Tokens */}
      <div ref={maxTokRef} className={sectionCls}>
        <div className="flex items-center justify-between">
          <label className="text-xs font-medium text-theme-ink">
            Max Tokens
            <Tooltip text="Controls the maximum length of the response. 1 token ≈ 4 characters." />
          </label>
          <span className={`text-xs font-bold px-2 py-0.5 rounded-full bg-[var(--c-bgCard)] ${tokenMeta.color}`}>
            {tokenMeta.label} · {maxTok}
          </span>
        </div>
        <Slider
          min={64} max={4096} step={64}
          value={maxTok}
          onChange={v => { setMaxTok(v); reveal("maxTok"); }}
          onCommit={v => commit({ max_tokens: parseInt(v) })}
          disabled={disabled}
          formatValue={() => ""}
        />
        <div className="flex justify-between text-xs text-theme-ink-muted">
          <span>64 — Short</span><span>1024 — Medium</span><span>4096 — Max</span>
        </div>
        <div className="flex gap-2 mt-1">
          {TOKEN_PRESETS.map(({ label, value }) => (
            <button key={value}
              onClick={() => { setMaxTok(value); reveal("maxTok"); commit({ max_tokens: value }); }}
              disabled={disabled}
              className={`flex-1 text-xs py-1 rounded border transition-all flex flex-col items-center leading-tight ${
                parseInt(maxTok) === value
                  ? "bg-cyan-500/20 border-cyan-500/40 text-cyan-300"
                  : "bg-[var(--c-bgCard)] border-[var(--c-line)] text-theme-ink-muted hover:border-cyan-500/40 hover:text-theme-ink-soft"
              }`}>
              <span>{label}</span>
              <span className="text-[10px] opacity-60">{value}</span>
            </button>
          ))}
        </div>
        {shown.has("maxTok") && <ParamInfoPanel meta={tokenMeta} color={tokenMeta.color} />}
      </div>

      {/* Top-p — only rendered if topP prop is provided */}
      {topP != null && setTopP && (
        <div ref={topPRef} className={sectionCls}>
          <div className="flex items-center justify-between">
            <label className="text-xs font-medium text-theme-ink">
              Top-p (Nucleus Sampling)
              <Tooltip text="Controls which words are considered. Lower = safer. Higher = more diverse." />
            </label>
            <span className={`text-xs font-bold px-2 py-0.5 rounded-full bg-[var(--c-bgCard)] ${topPMeta.color}`}>
              {topPMeta.label} · {topP}
            </span>
          </div>
          <Slider
            min={0.1} max={1} step={0.05}
            value={topP}
            onChange={v => { setTopP(v); reveal("topP"); }}
            onCommit={v => commit({ top_p: parseFloat(v) })}
            disabled={disabled}
            formatValue={() => ""}
          />
          <div className="flex justify-between text-xs text-theme-ink-muted">
            <span>0.1 — Focused</span><span>0.5 — Moderate</span><span>0.9 — Diverse</span>
          </div>
          {shown.has("topP")
            ? <ParamInfoPanel meta={topPMeta} color={topPMeta.color} />
            : <p className="text-xs italic text-theme-ink-muted">{topPMeta.tip}</p>}
        </div>
      )}
    </div>
  );
}
