import { C } from "../../theme";

export default function StreamingCaret() {
  return (
    <span
      aria-hidden
      className="inline-block w-1.5 h-4 rounded-sm ml-0.5 animate-pulse align-middle"
      style={{ background: C.accent }}
    />
  );
}
