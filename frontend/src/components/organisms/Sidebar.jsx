import { useState } from "react";
import { NavLink } from "react-router-dom";
import { C } from "../../theme";

/**
 * Sidebar — vertical icon-based navigation rail organism.
 *
 * Props:
 *   navItems   — [{ to, label, Icon, end? }]
 *   onLogout   — logout handler
 *   LogoutIcon — SVG icon component for the logout button
 *   style      — extra inline styles
 *   className  — extra Tailwind classes
 */
export default function Sidebar({ navItems, onLogout, LogoutIcon, style, className = "" }) {
  const [logoutHover, setLogoutHover] = useState(false);
  return (
    <aside
      className={`w-[68px] border-r flex flex-col items-center shrink-0 py-3 ${className}`}
      style={{ background: C.bgPanel, borderColor: C.line, ...style }}
    >
      {/* Logo */}
      <div className="mb-4 pb-4 w-full flex flex-col items-center gap-2" style={{ borderBottom: `1px solid ${C.line}` }}>
        <div className="w-10 h-10 rounded-2xl flex items-center justify-center"
          style={{
            background: `linear-gradient(135deg, var(--c-accent), var(--c-accentHover))`,
            boxShadow: `0 0 18px var(--c-accentGlow)`,
            outline: `1px solid var(--c-accentBorder)`,
          }}>
          <svg className="w-5 h-5 text-white drop-shadow" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2.2} stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 13.5l10.5-11.25L12 10.5h8.25L9.75 21.75 12 13.5H3.75z" />
          </svg>
        </div>
        <div className="flex flex-col items-center gap-0.5">
          <span className="text-[9px] font-extrabold tracking-[0.18em] uppercase leading-none"
            style={{ color: C.accent }}>
            AI
          </span>
          <span className="text-[7.5px] font-semibold tracking-[0.22em] uppercase leading-none"
            style={{ color: C.inkMuted }}>
            Lab
          </span>
        </div>
      </div>

      {/* Nav items */}
      <nav className="flex-1 flex flex-col items-center gap-1 w-full px-2">
        {navItems.map(({ to, label, Icon, end }) => (
          <NavLink key={to} to={to} end={end} className="w-full">
            {({ isActive }) => (
              <div className="flex flex-col items-center gap-1 py-2 px-1 rounded-xl transition-all cursor-pointer"
                style={{ background: isActive ? C.accentBg : undefined }}>
                <span className="w-8 h-8 flex items-center justify-center rounded-lg transition-all"
                  style={{
                    background: isActive ? C.accent : C.bgSoft,
                    boxShadow: isActive ? `0 2px 8px var(--c-accentGlow)` : undefined,
                  }}>
                  <Icon className="w-4 h-4" color={isActive ? "#fff" : "var(--c-inkMuted)"} />
                </span>
                <span className="text-[9px] font-semibold leading-none"
                  style={{ color: isActive ? C.accent : C.inkMuted }}>
                  {label}
                </span>
              </div>
            )}
          </NavLink>
        ))}
      </nav>

      {/* Logout */}
      <div className="mt-2 pt-2 w-full flex justify-center px-2" style={{ borderTop: `1px solid ${C.line}` }}>
        <button
          onClick={onLogout}
          onMouseEnter={() => setLogoutHover(true)}
          onMouseLeave={() => setLogoutHover(false)}
          title="Logout"
          className="flex flex-col items-center gap-1 py-2 px-1 w-full rounded-xl transition-all"
          style={{
            background: logoutHover ? "color-mix(in srgb, var(--c-danger) 12%, transparent)" : "transparent",
            color: logoutHover ? C.danger : C.inkMuted,
          }}
        >
          <span className="w-8 h-8 flex items-center justify-center rounded-lg transition-colors"
            style={{ background: logoutHover ? "color-mix(in srgb, var(--c-danger) 18%, transparent)" : C.bgSoft }}>
            {LogoutIcon && <LogoutIcon className="w-4 h-4" color={logoutHover ? C.danger : "var(--c-inkMuted)"} />}
          </span>
          <span className="text-[9px] font-semibold leading-none">Logout</span>
        </button>
      </div>
    </aside>
  );
}
