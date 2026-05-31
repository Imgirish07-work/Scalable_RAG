import { memo, useRef } from "react";
import { C } from "../../theme";
import { ATTACH_ACCEPT, ATTACH_MAX_BYTES, ATTACH_MAX_MB } from "../../utils/attachIcons";

/**
 * FileAttachButton — paperclip icon button with a hidden file input.
 *
 * Props:
 *   onSelect(file)   — required; called with a File object after validation
 *   onError(msg)     — optional; called with an error string if validation fails
 *   accept           — optional; defaults to the shared ATTACH_ACCEPT list
 *   maxBytes         — optional; defaults to the shared ATTACH_MAX_BYTES
 *   active           — optional; visually highlights the button when a file is staged
 *   disabled         — optional
 *   size             — optional pixel size for the button (default 34)
 *   title            — optional tooltip (default mentions accepted types and size)
 */
export default memo(function FileAttachButton({
  onSelect,
  onError,
  accept   = ATTACH_ACCEPT,
  maxBytes = ATTACH_MAX_BYTES,
  active   = false,
  disabled = false,
  size     = 34,
  title,
}) {
  const inputRef = useRef(null);

  const handleClick = () => {
    if (disabled) return;
    inputRef.current?.click();
  };

  const handleChange = (e) => {
    const picked = Array.from(e.target.files || []);
    e.target.value = "";
    if (!picked.length) return;
    const valid = [];
    for (const file of picked) {
      if (file.size > maxBytes) {
        onError?.(`"${file.name}" is too large. Max ${ATTACH_MAX_MB} MB.`);
      } else {
        valid.push(file);
      }
    }
    if (valid.length) onSelect?.(valid);
  };

  return (
    <>
      <button
        type="button"
        title={title || `Attach a file (max ${ATTACH_MAX_MB} MB)`}
        aria-label="Attach a file"
        onClick={handleClick}
        disabled={disabled}
        aria-disabled={disabled}
        className="flex-shrink-0 inline-flex items-center justify-center rounded-lg transition-colors disabled:cursor-not-allowed disabled:opacity-40"
        style={{
          width: size,
          height: size,
          background: active ? C.accentBg : "transparent",
          color:      active ? C.accent  : C.inkMuted,
          border:     "none",
          cursor:     disabled ? "not-allowed" : "pointer",
          padding:    0,
        }}
      >
        <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.586-6.586a4 4 0 00-5.656-5.656l-6.586 6.586a6 6 0 108.486 8.485L20.5 13"
          />
        </svg>
      </button>
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        multiple
        onChange={handleChange}
        className="hidden"
      />
    </>
  );
});
