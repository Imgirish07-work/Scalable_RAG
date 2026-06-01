import { memo } from "react";
import { C } from "../../theme";
import { ProgressBar } from "../atoms";

export default memo(function UploadProgress({ progress, style, className = "" }) {
  const hasDeterminate =
    (progress.stage === "embedding" && progress.total_batches) ||
    (progress.stage === "reading"   && progress.total_pages);

  const percentage = progress.total_batches
    ? Math.round((progress.batch / progress.total_batches) * 100)
    : progress.total_pages
      ? Math.round((progress.page / progress.total_pages) * 100)
      : null;

  return (
    <div className={`flex flex-col items-center justify-center w-full gap-1.5 px-4 ${className}`} style={style}>
      <div className="flex items-center gap-2 w-full" style={{ maxWidth: 320 }}>
        <ProgressBar
          value={hasDeterminate ? percentage : undefined}
          indeterminate={!hasDeterminate}
          height={6}
          className="flex-1"
        />
        {hasDeterminate && (
          <span className="text-[10px] font-mono flex-shrink-0" style={{ color: C.accent }}>
            {percentage}%
          </span>
        )}
      </div>
      <span className="text-[10px]" style={{ color: C.inkSoft }}>
        {progress.message}
      </span>
    </div>
  );
})
