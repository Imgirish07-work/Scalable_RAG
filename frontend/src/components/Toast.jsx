import { useState, useEffect, createContext, useContext, useCallback } from "react";
import {
  NOTIFICATION_STYLES,
  NotificationIcon,
  CloseIcon,
} from "./atoms/notificationStyles";

function ToastItem({ id, type = "info", title, message, duration = 10000, onDismiss }) {
  const [visible, setVisible] = useState(false);
  const [leaving, setLeaving] = useState(false);
  const s = NOTIFICATION_STYLES[type] ?? NOTIFICATION_STYLES.info;

  const dismiss = useCallback(() => {
    setLeaving(true);
    setTimeout(() => onDismiss(id), 300);
  }, [id, onDismiss]);

  useEffect(() => {
    requestAnimationFrame(() => setVisible(true));
    if (duration !== Infinity) {
      const t = setTimeout(dismiss, duration);
      return () => clearTimeout(t);
    }
  }, [dismiss, duration]);

  return (
    <div
      role="alert"
      aria-live="assertive"
      style={{
        display: "flex",
        alignItems: "flex-start",
        gap: "12px",
        padding: "12px 14px",
        background: s.bg,
        border: s.border,
        borderRadius: "12px",
        color: s.color,
        width: "380px",
        boxSizing: "border-box",
        boxShadow: "0 8px 24px rgba(0,0,0,0.45)",
        transform: visible && !leaving ? "translateX(0)" : "translateX(calc(100% + 100px))",
        opacity: visible && !leaving ? 1 : 0,
        transition: "transform 0.3s cubic-bezier(0.16, 1, 0.3, 1), opacity 0.3s ease",
        willChange: "transform, opacity",
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
        {message && (
          <p style={{
            margin: title ? "3px 0 0" : 0,
            fontSize: "12px",
            color: "inherit",
            opacity: 0.85,
            lineHeight: 1.5,
          }}>
            {message}
          </p>
        )}
      </div>

      <button
        onClick={dismiss}
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
    </div>
  );
}

function ToastContainer({ toasts, onDismiss }) {
  return (
    <div
      aria-label="Notifications"
      style={{
        position: "fixed",
        top: "60px",
        right: "10px",
        display: "flex",
        flexDirection: "column",
        gap: "10px",
        zIndex: 9999,
        pointerEvents: "none",
      }}
    >
      {toasts.map(t => (
        <div key={t.id} style={{ pointerEvents: "auto" }}>
          <ToastItem {...t} onDismiss={onDismiss} />
        </div>
      ))}
    </div>
  );
}

const ToastContext = createContext(null);

export function ToastProvider({ children }) {
  const [toasts, setToasts] = useState([]);

  const dismiss = useCallback((id) => {
    setToasts(prev => prev.filter(t => t.id !== id));
  }, []);

  const toast = useCallback((options) => {
    const id = crypto.randomUUID();
    setToasts(prev => [...prev, { id, ...options }]);
    return id;
  }, []);

  toast.success = (title, message, opts) => toast({ type: "success", title, message, ...opts });
  toast.error   = (title, message, opts) => toast({ type: "error",   title, message, ...opts });
  toast.warning = (title, message, opts) => toast({ type: "warning", title, message, ...opts });
  toast.info    = (title, message, opts) => toast({ type: "info",    title, message, ...opts });

  return (
    <ToastContext.Provider value={toast}>
      {children}
      <ToastContainer toasts={toasts} onDismiss={dismiss} />
    </ToastContext.Provider>
  );
}

export function useToast() {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error("useToast must be used inside <ToastProvider>");
  return ctx;
}
