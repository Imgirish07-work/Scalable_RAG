import { C } from "../../theme";

// sidebar + header + main content shell; pages own their own padding
export default function PlaygroundLayout({
  sidebar,
  header,
  children,
  style,
  className  = "",
}) {
  return (
    <div
      className={`flex h-screen overflow-hidden font-sans antialiased ${className}`}
      style={style}
    >
      {sidebar}

      <div className="flex-1 flex flex-col overflow-hidden">
        {header}
        <main
          className="flex-1 overflow-hidden"
          style={{ background: C.bg }}
        >
          {children}
        </main>
      </div>
    </div>
  );
}
