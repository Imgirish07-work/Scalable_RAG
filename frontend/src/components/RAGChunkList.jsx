import { useEffect, useState } from "react";
import { Button, Icon } from "./atoms";
import fileIcon from "../Assets/svg/file.svg";
import { C } from "../theme";

function ChunkCard({ source, index, onOpen, scoreType, maxScore }) {
  const [barWidth, setBarWidth] = useState(0);

  // normalize relative to max score so the bar is always visible
  const targetWidth = maxScore > 0 ? (source.score / maxScore) * 100 : 0;

  useEffect(() => {
    const t = setTimeout(() => setBarWidth(targetWidth), 80 + index * 120);
    return () => clearTimeout(t);
  }, [targetWidth, index]);

  const displayScore = scoreType === "RRF" && maxScore > 0
    ? (source.score / maxScore).toFixed(2)
    : source.score?.toFixed(2);
  const scoreLabel = `Score: ${displayScore}`;

  return (
    <div className="rounded-lg p-3 mb-2 last:mb-0" style={{ background: C.bgCard, border: `1px solid ${C.line}` }}>
      <div className="flex items-center justify-between mb-2">
        <Button
          onClick={() => onOpen(source.source_url)}
          variant="ghost"
          size="sm"
          className="truncate flex-1 text-left"
          style={{ fontFamily: "monospace" }}
        >
          <Icon src={fileIcon} className="w-3 h-3 inline mr-1 text-[var(--c-ink)]" />
          {source.filename}
        </Button>
        <span className="text-xs text-green-400 font-bold ml-3 flex-shrink-0">
          {scoreLabel}
        </span>
      </div>
      <div className="w-full rounded-full h-1 overflow-hidden" style={{ background: C.bgSoft }}>
        <div
          className="h-full rounded-full transition-all duration-[600ms] ease-out"
          style={{
            width: `${barWidth}%`,
            background: "linear-gradient(to right, #3b82f6, #22c55e)",
          }}
        />
      </div>
    </div>
  );
}

export default function RAGChunkList({ sources, onOpenSource, scoreType = "Cosine" }) {
  const maxScore = Math.max(...sources.map((s) => s.score ?? 0));
  return (
    <div>
      {sources.map((source, i) => (
        <ChunkCard key={i} index={i} source={source} onOpen={onOpenSource} scoreType={scoreType} maxScore={maxScore} />
      ))}
    </div>
  );
}
