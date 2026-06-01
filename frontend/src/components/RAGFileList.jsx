import { Button, Icon } from "./atoms";
import closeIcon from "../Assets/svg/close.svg";
import { C } from "../theme";

function getExt(filename) {
  return filename.split(".").pop().toLowerCase();
}

function FileCard({ file, onRemove }) {
  const ext = getExt(file.filename).toUpperCase().slice(0, 4);

  return (
    <div
      className="flex flex-col flex-shrink-0 rounded-lg"
      style={{
        width: 80,
        height: 90,
        border: `1px solid ${C.lineSoft}`,
        background: C.bgCard,
        padding: "8px 8px 6px",
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
        <Button
          onClick={() => onRemove(file.filename)}
          title="Remove from session"
          variant="ghost"
          size="sm"
          style={{ padding: "1px 2px", height: "auto", minWidth: 0 }}
        >
          <Icon src={closeIcon} className="w-2.5 h-2.5 text-[var(--c-ink)]" />
        </Button>
      </div>
    </div>
  );
}

export default function RAGFileList({ files, onRemove }) {
  const showFade = files.length > 3;

  return (
    <div className="relative">
      <div
        className="flex gap-2 overflow-x-auto"
        style={{ scrollbarWidth: "thin", scrollbarColor: `${C.lineSoft} transparent`, paddingBottom: 4 }}
      >
        {files.map((file, i) => (
          <FileCard key={`${file.filename}-${i}`} file={file} onRemove={onRemove} />
        ))}
      </div>

      {/* right fade hints that more cards exist */}
      {showFade && (
        <div
          className="absolute top-0 right-0 bottom-1 pointer-events-none"
          style={{
            width: 36,
            background: `linear-gradient(to right, transparent, ${C.bgCard}ee)`,
          }}
        />
      )}
    </div>
  );
}
