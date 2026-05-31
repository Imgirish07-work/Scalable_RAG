// Error-handling helpers. Use these instead of `.catch(() => {})` so intent is explicit.

const RAW_PY_ERROR_RE = /^(invalid literal|could not convert|unsupported operand|division by zero|object has no attribute|name '\w+' is not defined)/i;

// Replace raw Python exception leaks with a friendly fallback.
function sanitise(msg, fallback) {
  return RAW_PY_ERROR_RE.test(msg) ? fallback : msg;
}

// Pulls the best available message from an axios/fetch/wrapped error.
export function extractErrorMessage(err, fallback = "Something went wrong.") {
  if (!err) return fallback;

  // Raw axios shape, or our interceptor's preserved responseData
  const detail = err?.response?.data?.detail ?? err?.responseData?.detail;
  if (typeof detail === "string" && detail.trim()) {
    return sanitise(detail, fallback);
  }
  if (Array.isArray(detail) && detail.length > 0 && detail[0]?.msg) {
    // FastAPI 422 shape: [{ msg, loc, ... }]
    return sanitise(detail[0].msg, fallback);
  }

  // Interceptor-flattened errors or plain Errors
  if (typeof err.message === "string" && err.message.trim()) {
    return sanitise(err.message, fallback);
  }
  return fallback;
}

// Explicit no-op for fire-and-forget calls.
export function ignoreError() { /* intentional no-op */ }

// Logs + shows a toast.error with the extracted detail.
export function reportError(toast, title, fallback) {
  return (err) => {
    console.warn(`[${title}]`, err);
    toast.error(title, extractErrorMessage(err, fallback));
  };
}

// Logs + shows a toast.warning. For non-blocking degraded UX (stale counters, etc.).
export function reportWarning(toast, title, fallback) {
  return (err) => {
    console.warn(`[${title}]`, err);
    toast.warning(title, extractErrorMessage(err, fallback));
  };
}
