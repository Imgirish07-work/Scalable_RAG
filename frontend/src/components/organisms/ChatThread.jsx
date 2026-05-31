import { useRef, useEffect } from "react";
import { C } from "../../theme";

/**
 * ChatThread — scrollable chat message list organism.
 *
 * Renders an array of messages using caller-supplied render functions for each
 * role, keeping the thread auto-scrolled to the bottom on new messages.
 *
 * Props:
 *   messages          — [{ id, role: "user"|"assistant", content, ... }]
 *   renderUser(msg, i, isLast)      — returns JSX for a user message
 *   renderAssistant(msg, i)         — returns JSX for an assistant message
 *   streamingText     — if truthy, renders a streaming bubble at the bottom
 *   renderStreaming(text)            — returns JSX for the streaming indicator
 *   emptyState        — React node shown when messages is empty
 *   style             — extra inline styles
 *   className         — extra Tailwind classes
 */
export default function ChatThread({
  messages     = [],
  renderUser,
  renderAssistant,
  streamingText,
  renderStreaming,
  emptyState,
  style,
  className = "",
}) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length, streamingText]);

  if (!messages.length && !streamingText && emptyState) {
    return <div className={className} style={style}>{emptyState}</div>;
  }

  return (
    <div
      className={`flex flex-col gap-5 overflow-y-auto ${className}`}
      style={{ flex: 1, paddingBottom: 16, ...style }}
    >
      {messages.map((msg, i) => {
        const isLast = i === messages.length - 1;
        return (
          <div key={msg.id ?? i}>
            {msg.role === "user"
              ? renderUser(msg, i, isLast)
              : renderAssistant(msg, i)}
          </div>
        );
      })}

      {streamingText && renderStreaming && (
        <div>{renderStreaming(streamingText)}</div>
      )}

      <div ref={bottomRef} />
    </div>
  );
}
