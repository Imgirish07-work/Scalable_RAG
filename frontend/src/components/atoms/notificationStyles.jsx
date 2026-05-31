const SEVERITY = {
  success: "Success",
  warning: "Warning",
  error:   "Error",
  info:    "Info",
};

export const NOTIFICATION_STYLES = Object.fromEntries(
  Object.entries(SEVERITY).map(([k, s]) => [k, {
    bg:     `var(--c-bg${s})`,
    border: `1px solid var(--c-border${s})`,
    color:  `var(--c-text${s})`,
  }])
);

export function NotificationIcon({ type, color }) {
  switch (type) {
    case "success":
      return (
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
          <circle cx="8" cy="8" r="7" stroke={color} strokeWidth="1.5" />
          <path d="M5 8.5l2 2 4-4" stroke={color} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      );
    case "warning":
      return (
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
          <path d="M8 2L14.5 13.5H1.5L8 2Z" stroke={color} strokeWidth="1.5" strokeLinejoin="round" />
          <path d="M8 6.5V9.5" stroke={color} strokeWidth="1.5" strokeLinecap="round" />
          <circle cx="8" cy="11.5" r="0.75" fill={color} />
        </svg>
      );
    case "error":
      return (
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
          <circle cx="8" cy="8" r="7" stroke={color} strokeWidth="1.5" />
          <path d="M5.5 5.5l5 5M10.5 5.5l-5 5" stroke={color} strokeWidth="1.5" strokeLinecap="round" />
        </svg>
      );
    default:
      return (
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
          <circle cx="8" cy="8" r="7" stroke={color} strokeWidth="1.5" />
          <path d="M8 7v4.5" stroke={color} strokeWidth="1.5" strokeLinecap="round" />
          <circle cx="8" cy="5" r="0.75" fill={color} />
        </svg>
      );
  }
}

export const CloseIcon = () => (
  <svg width="13" height="13" viewBox="0 0 14 14" fill="none">
    <path d="M3 3l8 8M11 3l-8 8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
  </svg>
);
