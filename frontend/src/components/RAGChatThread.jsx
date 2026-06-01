import { useState } from "react";
import ReactMarkdown from "react-markdown";
import { C } from "../theme";
import StreamingCaret from "./atoms/StreamingCaret";

function SourceChip({ filename, onClick }) {
  const [hovered, setHovered] = useState(false);
  return (
    <button
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      className="inline-flex items-center gap-1.5 text-xs transition-colors"
      style={{
        padding: "4px 10px",
        background: hovered ? C.accentBg : C.bgSoft,
        border: `1px solid ${hovered ? C.accent : C.lineSoft}`,
        borderRadius: 4,
        color: hovered ? C.accent : C.inkSoft,
        cursor: onClick ? "pointer" : "default",
        transition: "all 0.15s",
      }}
    >
      <span style={{ width: 5, height: 5, borderRadius: "50%", background: C.accent, flexShrink: 0, display: "inline-block" }} />
      {filename}
    </button>
  );
}

export function MsgActionBtn({ title, onClick, children }) {
  const [h, setH] = useState(false);
  return (
    <button
      title={title}
      onClick={onClick}
      onMouseEnter={() => setH(true)}
      onMouseLeave={() => setH(false)}
      className="flex items-center justify-center rounded transition-colors"
      style={{
        width: 26, height: 26,
        background: h ? C.bgSoft : "none",
        border: "none",
        cursor: "pointer",
        color: h ? C.inkSoft : C.inkMuted,
      }}
    >
      {children}
    </button>
  );
}

export function IconCopy({ active }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke={active ? C.ok : "currentColor"}
      strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ width: 13, height: 13 }}>
      <rect x="9" y="9" width="13" height="13" rx="2" />
      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
    </svg>
  );
}

export function IconEdit() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ width: 13, height: 13 }}>
      <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
      <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
    </svg>
  );
}

export function IconRetry() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ width: 13, height: 13 }}>
      <polyline points="1 4 1 10 7 10" />
      <path d="M3.51 15a9 9 0 1 0 .49-4.5" />
    </svg>
  );
}

export function UserBubble({ msg, copiedId, onCopy, onEdit, disabled }) {
  const [hovered,   setHovered]   = useState(false);
  const [editMode,  setEditMode]  = useState(false);
  const [editValue, setEditValue] = useState(msg.content);

  const submitEdit = () => {
    const trimmed = editValue.trim();
    if (!trimmed) return;
    setEditMode(false);
    onEdit(msg, trimmed);
  };

  const cancelEdit = () => {
    setEditValue(msg.content);
    setEditMode(false);
  };

  return (
    <div
      className="flex gap-3 justify-end"
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <div className="flex-1 min-w-0 max-w-[75%]">
      {editMode ? (
        <div>
          <textarea
            value={editValue}
            onChange={e => setEditValue(e.target.value)}
            onKeyDown={e => {
              if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submitEdit(); }
              if (e.key === "Escape") cancelEdit();
            }}
            autoFocus
            rows={Math.max(2, editValue.split("\n").length)}
            className="w-full text-sm outline-none resize-none rounded-2xl rounded-tr-sm px-4 py-2.5"
            style={{
              background: C.bgSoft,
              border: `1px solid ${C.accent}`,
              color: C.ink,
              boxShadow: `0 0 0 2px ${C.accent}25`,
            }}
          />
          <div className="flex items-center justify-end gap-2 mt-1.5">
            <button
              onClick={cancelEdit}
              className="text-xs"
              style={{ color: C.inkMuted, background: "none", border: "none", cursor: "pointer" }}
            >
              Cancel
            </button>
            <button
              onClick={submitEdit}
              disabled={!editValue.trim()}
              className="text-xs font-medium rounded px-3 py-1"
              style={{
                background: editValue.trim() ? C.accent : C.bgSoft,
                color: editValue.trim() ? "#fff" : C.inkMuted,
                border: "none",
                cursor: editValue.trim() ? "pointer" : "not-allowed",
              }}
            >
              Send
            </button>
          </div>
        </div>
      ) : (
        <div className="rounded-2xl rounded-tr-sm px-4 py-2.5" style={{ background: C.accentBg, border: `1px solid ${C.accentBorder}` }}>
          <p className="text-sm text-theme-ink-soft whitespace-pre-wrap leading-relaxed">{msg.content}</p>
        </div>
      )}

      {!editMode && (
        <div
          className="flex items-center gap-0.5 justify-end mt-1 transition-opacity duration-150"
          style={{ opacity: disabled ? 0.3 : 1, pointerEvents: disabled ? "none" : "auto" }}
        >
          <MsgActionBtn title="Copy" onClick={() => onCopy(msg.id, msg.content)}>
            <IconCopy active={copiedId === msg.id} />
          </MsgActionBtn>
          <MsgActionBtn title="Edit" onClick={() => { setEditValue(msg.content); setEditMode(true); }}>
            <IconEdit />
          </MsgActionBtn>
        </div>
      )}
      </div>
      <div className="flex-shrink-0 w-7 h-7 rounded-full flex items-center justify-center text-[10px] font-bold mt-0.5" style={{ background: C.accentBg, border: `1px solid ${C.accentBorder}`, color: C.accent }}>U</div>
    </div>
  );
}

export function AssistantBubble({ msg, copiedId, onCopy, onRetry, disabled }) {
  const [hovered, setHovered] = useState(false);

  const uniqueSources = msg.sources?.length
    ? [...new Map(msg.sources.map(s => [s.filename, s])).values()]
    : [];

  return (
    <div className="flex gap-2.5">
      <div
        className="flex flex-col gap-2"
        style={{ maxWidth: "85%" }}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
      >

      <div
        className="prose prose-invert prose-sm max-w-none text-sm leading-relaxed"
        style={{ color: C.ink }}
      >
        <ReactMarkdown
          components={{
            a: ({ href, children }) => (
              <a href={href} target="_blank" rel="noopener noreferrer">{children}</a>
            ),
          }}
        >{msg.content}</ReactMarkdown>
      </div>

      {uniqueSources.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {uniqueSources.map((s, i) => (
            <SourceChip key={i} filename={s.filename} />
          ))}
        </div>
      )}

      <div
        className="flex items-center gap-0.5 transition-opacity duration-150"
        style={{ opacity: hovered && !disabled ? 1 : 0, pointerEvents: hovered && !disabled ? "auto" : "none" }}
      >
        <MsgActionBtn title="Copy" onClick={() => onCopy(msg.id, msg.content)}>
          <IconCopy active={copiedId === msg.id} />
        </MsgActionBtn>
        <MsgActionBtn title="Retry" onClick={() => onRetry(msg)}>
          <IconRetry />
        </MsgActionBtn>
      </div>

    </div>
    </div>
  );
}

export function StreamingBubble({ text }) {
  return (
    <div className="flex gap-2.5">
      <div style={{ maxWidth: "85%" }}>
      <p className="text-sm leading-relaxed whitespace-pre-wrap" style={{ color: C.ink }}>
        {text}
        <StreamingCaret />
      </p>
    </div>
    </div>
  );
}
