import { C } from "../../theme";

/**
 * PlaygroundLayout — sidebar + header + main content shell.
 *
 * This is the authenticated app shell used by App.jsx.
 * It accepts the sidebar, header, and main content as slots.
 * Main area has no padding — pages own their own spacing.
 *
 * Props:
 *   sidebar      — React node for the left nav rail (e.g. <Sidebar />)
 *   header       — React node for the top bar (e.g. <Header />)
 *   children     — main content area (pages rendered via <Routes />)
 *   style        — extra inline styles on root
 *   className    — extra Tailwind classes on root
 */
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
      {/* Nav rail */}
      {sidebar}

      {/* Main column */}
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
