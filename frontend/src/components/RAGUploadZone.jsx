import { useRef, useState } from "react";
import { Button, ProgressBar } from "./atoms";
import { C } from "../theme";

// drop zone with idle / staged / uploading states — parent owns queue/progress state
export default function RAGUploadZone({
  onAddFiles,
  onStart,
  accept      = ".pdf,.txt,.md,.docx,.pptx",
  disabled    = false,
  stagedCount = 0,
  activeJob   = null,
}) {
  const inputRef = useRef();
  const [dragging, setDragging] = useState(false);

  const pick = (fileList) => {
    if (disabled) return;
    const files = Array.from(fileList || []);
    if (files.length) onAddFiles(files);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    pick(e.dataTransfer.files);
  };

  const isUploading = !!activeJob;
  const isStaged    = !isUploading && stagedCount > 0;
  const active      = dragging && !disabled;

  const borderStyle = isUploading || isStaged ? "solid" : "dashed";
  const borderTone  = isUploading || active   ? C.accent : C.accentBorder;
  const bgTone      = isUploading || active   ? C.accentBg : C.accentGlow;

  return (
    <div
      role="button"
      tabIndex={disabled ? -1 : 0}
      onClick={() => !disabled && inputRef.current.click()}
      onKeyDown={(e) => { if (!disabled && e.key === "Enter") inputRef.current.click(); }}
      onDragOver={(e) => { e.preventDefault(); if (!disabled) setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      className="flex items-center rounded-lg select-none"
      style={{
        padding: isUploading ? "6px 10px" : "0 10px",
        minHeight: isUploading ? 52 : 40,
        border: `1.5px ${borderStyle} ${borderTone}`,
        background: bgTone,
        boxShadow: isUploading
          ? "0 0 8px rgba(13,148,136,0.2)"
          : active
            ? "0 0 12px rgba(13,148,136,0.4)"
            : "0 0 6px rgba(13,148,136,0.1), inset 0 0 8px rgba(13,148,136,0.05)",
        cursor: disabled ? "not-allowed" : "pointer",
        opacity: disabled ? 0.5 : 1,
        transition: "border-color 0.15s, background 0.15s, box-shadow 0.15s, min-height 0.15s",
      }}
    >
      {isUploading ? (
        <UploadingView job={activeJob} />
      ) : isStaged ? (
        <StagedView active={active} onStart={onStart} />
      ) : (
        <IdleView active={active} />
      )}

      <input
        ref={inputRef}
        type="file"
        accept={accept}
        multiple
        onChange={(e) => { pick(e.target.files); e.target.value = ""; }}
        className="hidden"
      />
    </div>
  );
}

function IdleView({ active }) {
  return (
    <div className="flex items-center justify-center gap-2 w-full">
      <UploadIcon active={active} />
      <p
        className="text-xs"
        style={{ color: active ? C.accent : C.inkSoft, transition: "color 0.15s", letterSpacing: "0.02em" }}
      >
        Drag & Drop Or{" "}
        <span className="font-semibold" style={{ color: C.accent }}>Browse</span>
      </p>
    </div>
  );
}

function StagedView({ active, onStart }) {
  return (
    <div className="flex items-center gap-2 w-full">
      <UploadIcon active={active} />
      <p
        className="text-xs flex-1 min-w-0"
        style={{ color: active ? C.accent : C.inkSoft, letterSpacing: "0.02em" }}
      >
        Drag & Drop Or{" "}
        <span className="font-semibold" style={{ color: C.accent }}>Browse</span>
      </p>
      <Button
        onClick={(e) => { e.stopPropagation(); onStart?.(); }}
        size="sm"
      >
        Upload
      </Button>
    </div>
  );
}

function UploadingView({ job }) {
  return (
    <div className="flex flex-col gap-1.5 w-full min-w-0">
      <div className="flex items-center gap-2 min-w-0">
        <span className="text-[11px] truncate flex-1 min-w-0" style={{ color: C.ink }} title={job.filename}>
          {job.filename}
        </span>
        <span className="text-[10px] flex-shrink-0 font-mono" style={{ color: C.accent }}>
          {job.progress}%
        </span>
      </div>
      <ProgressBar value={job.progress} height={4} gradient={C.accent} />
      <span className="text-[10px] truncate" style={{ color: C.inkSoft }}>
        {job.message || "Working..."}
      </span>
    </div>
  );
}

function UploadIcon({ active }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke={active ? C.accent : C.inkMuted}
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      style={{ width: 14, height: 14, flexShrink: 0, transition: "stroke 0.15s" }}
    >
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="17 8 12 3 7 8" />
      <line x1="12" y1="3" x2="12" y2="15" />
    </svg>
  );
}
