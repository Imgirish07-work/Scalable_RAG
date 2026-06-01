import { useRef, useEffect } from "react";
import { C } from "../../theme";

// scrollable chat message list, auto-scrolls to bottom on new messages
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
