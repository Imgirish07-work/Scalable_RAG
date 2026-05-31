import { memo, useEffect, useState } from "react";
import { Icon } from "../atoms";
import fileIcon from "../../Assets/svg/file.svg";
import { C } from "../../theme";

/**
 * ChunkCard — single retrieved chunk card with filename, score, and animated bar.
 *
 * Props:
 *   source    — { filename, source_url, score }
 *   index     — position in list (for staggered animation)
 *   onOpen    — called with source_url on filename click
 *   scoreType — "Cosine" | "RRF" etc.
 *   maxScore  — highest score in the set (for normalisation)
 *   style     — extra inline styles
 *   className — extra Tailwind classes
 */
export default memo(function ChunkCard({ source, index = 0, onOpen, scoreType, maxScore, style, className = "" }) {
  const [barWidth, setBarWidth] = useState(0);
  const targetWidth = maxScore > 0 ? (source.score / maxScore) * 100 : 0;

  useEffect(() => {
    const t = setTimeout(() => setBarWidth(targetWidth), 80 + index * 120);
    return () => clearTimeout(t);
  }, [targetWidth, index]);

  const displayScore = scoreType === "RRF" && maxScore > 0
    ? (source.score / maxScore).toFixed(2)
    : source.score?.toFixed(2);

  return (
    <div
      className={`rounded-lg p-3 mb-2 last:mb-0 ${className}`}
      style={{ background: C.bgSoft, border: `1px solid ${C.line}`, ...style }}
    >
      <div className="flex items-center justify-between mb-2">
        <button
          onClick={() => onOpen(source.source_url)}
          className="text-xs font-mono truncate flex-1 text-left hover:underline underline-offset-2"
          style={{ color: C.accent }}
        >
          <Icon src={fileIcon} className="w-3 h-3 inline mr-1 text-[var(--c-ink)]" />
          {source.filename}
        </button>
        <span className="text-xs font-bold ml-3 flex-shrink-0" style={{ color: C.ok }}>
          Score: {displayScore}
        </span>
      </div>
      <div className="w-full rounded-full h-1 overflow-hidden" style={{ background: C.lineSoft }}>
        <div
          className="h-full rounded-full transition-all duration-[600ms] ease-out"
          style={{ width: `${barWidth}%`, background: "linear-gradient(to right, #3b82f6, #22c55e)" }}
        />
      </div>
    </div>
  );
})
