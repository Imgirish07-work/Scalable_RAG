import { memo } from "react";
import Icon from "../atoms/Icon";
import IconButton from "../atoms/IconButton";
import { getAttachIcon } from "../../utils/attachIcons";
import { C } from "../../theme";

/**
 * AttachmentPill — shows a staged file with name, size, and a remove button.
 *
 * Props:
 *   file     — required; a File object (or { name, size } shape)
 *   onClear  — required; called when the user removes the attachment
 *   error    — optional; error string shown below the pill
 */
export default memo(function AttachmentPill({ file, onClear, error }) {
  if (!file) return null;
  const sizeKB = (file.size / 1024).toFixed(0);
  return (
    <div>
      <div
        className="inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1 text-xs max-w-xs"
        style={{
          background: C.accentBg,
          border: `1px solid ${C.accentBorder}`,
          color: C.accent,
        }}
      >
        <Icon
          src={getAttachIcon(file.name)}
          className="w-3.5 h-3.5 flex-shrink-0 text-theme-accent"
        />
        <span className="truncate font-medium">{file.name}</span>
        <span className="ml-1 flex-shrink-0" style={{ color: C.inkMuted }}>
          ({sizeKB} KB)
        </span>
        <IconButton
          onClick={onClear}
          title="Remove attachment"
          size={16}
          style={{ color: C.danger, marginLeft: 4 }}
        >
          ✕
        </IconButton>
      </div>
      {error && (
        <p className="mt-1 text-[11px]" style={{ color: C.danger }}>
          {error}
        </p>
      )}
    </div>
  );
});
