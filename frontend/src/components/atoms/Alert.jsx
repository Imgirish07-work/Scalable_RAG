import { memo } from "react";
import { NOTIFICATION_STYLES, NotificationIcon, CloseIcon } from "./notificationStyles";

// inline, persistent notification — visual twin of Toast
function Alert({
  type = "info",
  title,
  message,
  dismissible = false,
  onDismiss,
  action,
  className = "",
  style,
  children,
}) {
  const s = NOTIFICATION_STYLES[type] ?? NOTIFICATION_STYLES.info;

  return (
    <div
      role="alert"
      className={className}
      style={{
        display: "flex",
        alignItems: "flex-start",
        gap: "12px",
        padding: "12px 14px",
        background: s.bg,
        border: s.border,
        borderRadius: "12px",
        color: s.color,
        width: "100%",
        boxSizing: "border-box",
        ...style,
      }}
    >
      <span style={{ flexShrink: 0, marginTop: "2px", display: "flex" }}>
        <NotificationIcon type={type} color={s.color} />
      </span>

      <div style={{ flex: 1, minWidth: 0 }}>
        {title && (
          <p style={{ margin: 0, fontSize: "13px", fontWeight: 600, color: "inherit" }}>
            {title}
          </p>
        )}
        {(message || children) && (
          <div
            style={{
              margin: title ? "3px 0 0" : 0,
              fontSize: "12px",
              color: "inherit",
              opacity: 0.85,
              lineHeight: 1.5,
            }}
          >
            {message ?? children}
          </div>
        )}
        {action && <div style={{ marginTop: "8px" }}>{action}</div>}
      </div>

      {dismissible && (
        <button
          type="button"
          onClick={onDismiss}
          aria-label="Dismiss"
          style={{
            flexShrink: 0,
            background: "none",
            border: "none",
            padding: 0,
            cursor: "pointer",
            color: "inherit",
            opacity: 0.55,
            display: "flex",
            alignItems: "center",
            marginTop: "2px",
          }}
          onMouseEnter={e => { e.currentTarget.style.opacity = 1; }}
          onMouseLeave={e => { e.currentTarget.style.opacity = 0.55; }}
        >
          <CloseIcon />
        </button>
      )}
    </div>
  );
}

export default memo(Alert);
