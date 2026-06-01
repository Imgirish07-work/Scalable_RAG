import { memo } from "react";
import SendButton from "../atoms/SendButton";
import StopButton from "../atoms/StopButton";
import FileAttachButton from "../atoms/FileAttachButton";
import AttachmentPill from "./AttachmentPill";
import { C } from "../../theme";

// text input + Send/Stop button with optional multi-file attachments
export default memo(function QueryBox({
  value,
  onChange,
  onSubmit,
  onStop,
  isLoading = false,
  disabled = false,
  placeholder = "Send a message",
  inputRef,
  className = "",
  leftAdornment = null,
  rightAdornment = null,
  canSubmit,
  attachments,
  onAttachmentsChange,
  attachError,
  onAttachError,
  attachAccept,
  attachMaxBytes,
}) {
  const fileAttachEnabled = typeof onAttachmentsChange === "function";
  const files = Array.isArray(attachments) ? attachments : [];

  // files alone are not sufficient — user must also type a prompt
  const hasContent = value.trim().length > 0;
  const baseCanSend = canSubmit !== undefined ? canSubmit : hasContent;
  const canSend     = baseCanSend && !isLoading && !disabled;

  const handleAdd = (newFiles) => {
    if (!fileAttachEnabled) return;
    const arr = Array.isArray(newFiles) ? newFiles : [newFiles];
    onAttachmentsChange([...files, ...arr]);
  };
  const handleRemove = (idx) => {
    if (!fileAttachEnabled) return;
    onAttachmentsChange(files.filter((_, i) => i !== idx));
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (isLoading && onStop) {
        onStop();
      } else if (canSend) {
        onSubmit();
      }
    }
  };

  return (
    <div
      className={`flex flex-col rounded-2xl mx-4 mt-4 mb-3 ${className}`}
      style={{
        background: C.bgInput,
        border: `1.5px solid ${C.lineCard}`,
      }}
    >
      {fileAttachEnabled && files.length > 0 && (
        <div className="flex flex-wrap gap-1 pt-2 px-4">
          {files.map((f, i) => (
            <AttachmentPill
              key={`${f.name}-${f.size}-${i}`}
              file={f}
              onClear={() => handleRemove(i)}
              error={i === files.length - 1 ? attachError : undefined}
            />
          ))}
        </div>
      )}

      <textarea
        ref={inputRef}
        rows={1}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        disabled={disabled}
        aria-label={placeholder}
        className="bg-transparent outline-none resize-none overflow-y-auto
                   text-sm text-theme-ink placeholder-theme-ink-muted px-4 pt-3 pb-1
                   disabled:cursor-not-allowed disabled:opacity-50
                   focus:outline-none"
        style={{ lineHeight: 1.5, maxHeight: 140 }}
      />

      <div className="flex items-center gap-1.5 px-1.5 pb-1.5">
        {fileAttachEnabled && (
          <div className="flex-shrink-0">
            <FileAttachButton
              onSelect={handleAdd}
              onError={onAttachError}
              active={files.length > 0}
              accept={attachAccept}
              maxBytes={attachMaxBytes}
              disabled={disabled || isLoading}
            />
          </div>
        )}
        {leftAdornment && <div className="flex-shrink-0">{leftAdornment}</div>}
        <div className="flex-1" />
        {rightAdornment && <div className="flex-shrink-0">{rightAdornment}</div>}
        <div className="flex-shrink-0">
          {isLoading && onStop ? (
            <StopButton onClick={onStop} />
          ) : (
            <SendButton onClick={onSubmit} disabled={!canSend} />
          )}
        </div>
      </div>
    </div>
  );
})
