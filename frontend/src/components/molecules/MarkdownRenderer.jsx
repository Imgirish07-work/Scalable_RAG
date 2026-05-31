import { memo } from "react";
import ReactMarkdown from "react-markdown";

/**
 * MarkdownRenderer — prose-styled markdown display molecule.
 *
 * Props:
 *   content   — markdown string
 *   style     — extra inline styles on the wrapper
 *   className — extra Tailwind classes on the wrapper
 */
export default memo(function MarkdownRenderer({ content, style, className = "" }) {
  if (!content) return null;
  return (
    <div
      role="article"
      className={`prose prose-invert prose-sm max-w-none
        prose-p:text-[var(--c-ink)] prose-p:leading-relaxed prose-p:my-1
        prose-strong:text-[var(--c-ink)] prose-headings:text-[var(--c-ink)]
        prose-code:text-indigo-300 prose-code:bg-[var(--c-bgSoft)] prose-code:px-1 prose-code:rounded prose-code:text-xs
        prose-pre:bg-[var(--c-bgDeep)] prose-pre:border prose-pre:border-[var(--c-line)] prose-pre:rounded-lg
        prose-ul:text-[var(--c-ink)] prose-ol:text-[var(--c-ink)] prose-li:marker:text-indigo-400
        ${className}`}
      style={style}
    >
      <ReactMarkdown
        components={{
          a: ({ href, children }) => (
            <a href={href} target="_blank" rel="noopener noreferrer">{children}</a>
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
})
