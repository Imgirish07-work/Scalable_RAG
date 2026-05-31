import { C } from "../../theme";

/**
 * StreamingCaret — blinking accent-coloured caret shown at the end of
 * streaming assistant content. Single source of truth so the cursor looks
 * identical across Model Playground, RAG Playground, and Prompt Solution.
 */
export default function StreamingCaret() {
  return (
    <span
      aria-hidden
      className="inline-block w-1.5 h-4 rounded-sm ml-0.5 animate-pulse align-middle"
      style={{ background: C.accent }}
    />
  );
}
