import { memo } from "react";
import Icon from "../atoms/Icon";
import IconButton from "../atoms/IconButton";
import { getAttachIcon } from "../../utils/attachIcons";
import { C } from "../../theme";

export default memo(function AttachmentPill({ file, onClear, error, onClick }) {
  if (!file) return null;
  const sizeKB = (file.size / 1024).toFixed(0);
  const isClickable = typeof onClick === "function";

  const handleClear = (e) => {
    e?.stopPropagation?.();
    onClear?.();
  };

  return (
    <div>
      <div
        onClick={onClick}
        className="inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1 text-xs max-w-xs"
        style={{
          background: C.accentBg,
          border: `1px solid ${C.accentBorder}`,
          color: C.accent,
          cursor: isClickable ? "pointer" : "default",
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
          onClick={handleClear}
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
