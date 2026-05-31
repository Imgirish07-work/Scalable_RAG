import { useRef } from "react";
import { Icon } from "./atoms";
import documentIcon from "../Assets/svg/document.svg";

export default function FileUploader({ onUpload, accept = ".pdf,.txt,.docx,.md" }) {
  const ref = useRef();
  return (
    <div
      onClick={() => ref.current.click()}
      className="border-2 border-dashed border-theme-line rounded-lg p-6 text-center
                 cursor-pointer hover:border-blue-500 hover:bg-blue-500/5 transition-colors">
      <Icon src={documentIcon} className="w-7 h-7 mx-auto mb-2 opacity-60 text-[var(--c-ink)]" />
      <p className="text-sm text-theme-ink-soft">Click to upload document</p>
      <p className="text-xs text-theme-ink-muted mt-1">{accept}</p>
      <input
        ref={ref} type="file" accept={accept}
        onChange={e => { const f = e.target.files[0]; if (f) onUpload(f); }}
        className="hidden"
      />
    </div>
  );
}