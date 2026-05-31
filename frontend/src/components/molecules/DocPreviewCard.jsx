import { memo, useEffect, useRef, useState } from "react";
import { RefreshCw, X } from "lucide-react";
import DocPreviewBadge from "../atoms/DocPreviewBadge";
import { loadPdfJs } from "../../utils/pdfjs";
import { C } from "../../theme";

const IMAGE_EXTS = new Set(["png", "jpg", "jpeg", "gif", "webp", "svg", "bmp"]);

function extOf(name = "") {
  return (name.split(".").pop() || "").toLowerCase();
}

function inferKind(filename, mimeType, file) {
  const ext = extOf(filename || file?.name || "");
  const mime = mimeType || file?.type || "";
  if (ext === "pdf" || mime === "application/pdf") return "pdf";
  if (IMAGE_EXTS.has(ext) || mime.startsWith("image/")) return "image";
  return "other";
}

function inferLabel(filename, kind) {
  if (kind === "image") return "IMG";
  const ext = extOf(filename);
  return (ext || "FILE").toUpperCase().slice(0, 4);
}

function statusOverlayStyle(status) {
  if (status === "failed") {
    return {
      background: "rgba(184, 61, 44, 0.12)",
      borderColor: "var(--c-textError)",
    };
  }
  if (status === "processing") {
    return {
      background: "rgba(208, 138, 106, 0.08)",
      borderColor: C.accentBorder,
    };
  }
  return null;
}

export default memo(function DocPreviewCard({
  file,
  fetchUrl,
  url,
  filename = "",
  label,
  width = 190,
  height = 234,
  status,
  progress,
  message,
  onClick,
  onRemove,
  onRetry,
}) {
  const displayName = filename || file?.name || "Untitled";
  const kind = inferKind(displayName, undefined, file);
  const badgeLabel = label || inferLabel(displayName, kind);

  const rootRef = useRef(null);
  const canvasRef = useRef(null);
  const [resolvedUrl, setResolvedUrl] = useState(url || null);
  const [thumbState, setThumbState] = useState("idle"); // idle | loading | ready | error
  const [visible, setVisible] = useState(!fetchUrl); // immediate render unless we need to lazy-fetch
  const [hovered, setHovered] = useState(false);
  const isClickable = typeof onClick === "function";
  const overlay = statusOverlayStyle(status);

  // Lazy: only fetch the presigned URL when card enters viewport. Avoids
  // hammering MinIO with 50 GETs on first render of a large indexed library.
  useEffect(() => {
    if (!fetchUrl || resolvedUrl) return undefined;
    if (!rootRef.current) return undefined;
    const io = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          setVisible(true);
          io.disconnect();
        }
      },
      { rootMargin: "120px" },
    );
    io.observe(rootRef.current);
    return () => io.disconnect();
  }, [fetchUrl, resolvedUrl]);

  // Resolve URL once card is visible (or immediately for File / direct URL).
  useEffect(() => {
    if (!visible) return undefined;
    if (file) {
      // For local File: only need an object URL if we render the image branch.
      if (kind === "image") {
        const blobUrl = URL.createObjectURL(file);
        setResolvedUrl(blobUrl);
        return () => URL.revokeObjectURL(blobUrl);
      }
      return undefined;
    }
    if (url) {
      setResolvedUrl(url);
      return undefined;
    }
    if (fetchUrl && !resolvedUrl) {
      let cancelled = false;
      fetchUrl()
        .then((u) => { if (!cancelled) setResolvedUrl(u); })
        .catch(() => { if (!cancelled) setThumbState("error"); });
      return () => { cancelled = true; };
    }
    return undefined;
  }, [visible, file, url, fetchUrl, kind, resolvedUrl]);

  // Render PDF first page into the canvas once we have data + URL.
  useEffect(() => {
    if (kind !== "pdf") return undefined;
    if (!file && !resolvedUrl) return undefined;
    let cancelled = false;
    setThumbState("loading");
    (async () => {
      try {
        const lib = await loadPdfJs();
        const data = file ? new Uint8Array(await file.arrayBuffer()) : null;
        const task = data
          ? lib.getDocument({ data })
          : lib.getDocument({ url: resolvedUrl });
        const pdf = await task.promise;
        const page = await pdf.getPage(1);
        if (cancelled || !canvasRef.current) return;
        const dpr = window.devicePixelRatio || 1;
        const base = page.getViewport({ scale: 1 });
        const scale = (width * dpr) / base.width;
        const vp = page.getViewport({ scale });
        const canvas = canvasRef.current;
        canvas.width = vp.width;
        canvas.height = vp.height;
        canvas.style.width = "100%";
        await page.render({
          canvasContext: canvas.getContext("2d"),
          viewport: vp,
        }).promise;
        if (!cancelled) setThumbState("ready");
      } catch {
        if (!cancelled) setThumbState("error");
      }
    })();
    return () => { cancelled = true; };
  }, [kind, file, resolvedUrl, width]);

  // Cards are theme-aware now (match active theme bg/border/text) so they
  // sit naturally on the surrounding pane in both light and dark modes.
  // PDF thumbnails keep their own white paper inside, so the badge needs
  // "dark" tone when over a PDF, otherwise "themed".
  const cardBadgeTone = kind === "pdf" || kind === "image" ? "dark" : "themed";
  return (
    <div
      ref={rootRef}
      onClick={isClickable ? onClick : undefined}
      title={message || displayName}
      style={{
        position: "relative",
        width,
        height,
        flexShrink: 0,
        borderRadius: 12,
        overflow: "hidden",
        background: C.bgCard,
        border: overlay
          ? `1.5px solid ${overlay.borderColor}`
          : `1px solid ${C.lineSoft}`,
        boxShadow: "0 1px 2px rgba(0,0,0,0.10), 0 4px 12px rgba(0,0,0,0.10)",
        cursor: isClickable ? "pointer" : "default",
        transition: "transform 0.15s, box-shadow 0.15s",
      }}
      onMouseEnter={(e) => {
        setHovered(true);
        if (isClickable) {
          e.currentTarget.style.transform = "translateY(-2px)";
          e.currentTarget.style.boxShadow = "0 2px 4px rgba(0,0,0,0.22), 0 14px 36px rgba(0,0,0,0.30)";
        }
      }}
      onMouseLeave={(e) => {
        setHovered(false);
        if (isClickable) {
          e.currentTarget.style.transform = "none";
          e.currentTarget.style.boxShadow = "0 1px 2px rgba(0,0,0,0.18), 0 10px 30px rgba(0,0,0,0.25)";
        }
      }}
    >
      {/* Thumbnail layer — top-aligned so the page head shows, like Claude.
          PDF/image keep their own white paper feel; the generic placeholder
          adopts the card's themed surface so it doesn't flash white in dark mode. */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          overflow: "hidden",
          background: kind === "other" ? C.bgCard : "#ffffff",
        }}
      >
        {kind === "image" && resolvedUrl && (
          <img
            src={resolvedUrl}
            alt={displayName}
            style={{ width: "100%", height: "100%", objectFit: "cover", objectPosition: "top" }}
          />
        )}
        {kind === "pdf" && (
          <canvas ref={canvasRef} style={{ display: "block", width: "100%", background: "#ffffff" }} />
        )}
        {kind === "other" && <GenericPagePlaceholder filename={displayName} />}

        {(thumbState === "loading" || (!visible && fetchUrl)) && (
          <div
            style={{
              position: "absolute",
              inset: 0,
              display: "grid",
              placeItems: "center",
              color: "#9b958a",
              fontSize: 11,
              fontFamily: "system-ui, sans-serif",
            }}
          >
            {!visible ? "" : "rendering…"}
          </div>
        )}

        {thumbState === "error" && kind === "pdf" && <GenericPagePlaceholder />}

        {/* Status tint over the white page when uploading/processing/failed */}
        {overlay && (
          <div
            style={{
              position: "absolute",
              inset: 0,
              background: overlay.background,
              pointerEvents: "none",
            }}
          />
        )}
      </div>

      {/* Top progress bar when uploading */}
      {status === "uploading" && (
        <div
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            right: 0,
            height: 3,
            background: "rgba(0,0,0,0.08)",
          }}
        >
          <div
            style={{
              height: "100%",
              width: `${progress || 0}%`,
              background: C.accent,
              transition: "width 0.2s",
            }}
          />
        </div>
      )}

      {status === "processing" && (
        <div
          className="animate-pulse"
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            right: 0,
            height: 3,
            background: C.accent,
          }}
        />
      )}

      {/* Corner buttons — hover-only, fade in. Retry is always-visible for
          failed cards so users notice them; the × is hover-only. */}
      {(onRemove || (onRetry && status === "failed")) && (
        <div
          style={{
            position: "absolute",
            top: 4,
            right: 4,
            display: "flex",
            gap: 3,
            zIndex: 2,
            opacity: hovered || status === "failed" ? 1 : 0,
            transition: "opacity 0.15s",
            pointerEvents: hovered || status === "failed" ? "auto" : "none",
          }}
        >
          {onRetry && status === "failed" && (
            <CornerButton
              title="Retry"
              onClick={(e) => { e.stopPropagation(); onRetry(); }}
            >
              <RefreshCw size={10} />
            </CornerButton>
          )}
          {onRemove && (
            <CornerButton
              title="Remove"
              onClick={(e) => { e.stopPropagation(); onRemove(); }}
            >
              <X size={11} />
            </CornerButton>
          )}
        </div>
      )}

      <DocPreviewBadge label={badgeLabel} tone={cardBadgeTone} />

      {status === "uploading" && (
        <div
          style={{
            position: "absolute",
            left: 12,
            right: 12,
            bottom: 48,
            padding: "3px 8px",
            background: "var(--c-bgSoft)",
            color: "var(--c-inkSoft)",
            fontSize: 10,
            fontWeight: 500,
            borderRadius: 6,
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
            pointerEvents: "none",
            fontFamily: "system-ui, sans-serif",
          }}
        >
          {progress != null ? `${progress}%` : "Uploading"}
          {message ? ` · ${message}` : ""}
        </div>
      )}
    </div>
  );
});

function CornerButton({ children, title, onClick }) {
  return (
    <button
      title={title}
      onClick={onClick}
      style={{
        width: 18,
        height: 18,
        display: "grid",
        placeItems: "center",
        borderRadius: "50%",
        background: "rgba(58,57,54,0.85)",
        backdropFilter: "blur(6px)",
        WebkitBackdropFilter: "blur(6px)",
        border: "none",
        color: "#ffffff",
        cursor: "pointer",
        padding: 0,
        boxShadow: "0 1px 3px rgba(0,0,0,0.25)",
      }}
      onMouseEnter={(e) => { e.currentTarget.style.background = "rgba(58,57,54,1)"; }}
      onMouseLeave={(e) => { e.currentTarget.style.background = "rgba(58,57,54,0.85)"; }}
    >
      {children}
    </button>
  );
}

function GenericPagePlaceholder({ filename }) {
  // Clean fallback for files without a renderable thumbnail (DOCX, XLSX,
  // TXT, PPT, etc.)
  return (
    <div
      style={{
        position: "absolute",
        inset: 0,
        padding: "8px 10px",
        background: "var(--c-bgCard)",
      }}
    >
      {filename && (
        <p
          style={{
            margin: 0,
            color: "var(--c-ink)",
            fontSize: 10,
            fontWeight: 500,
            lineHeight: 1.3,
            textAlign: "left",
            wordBreak: "break-word",
            overflowWrap: "anywhere",
            display: "-webkit-box",
            WebkitLineClamp: 4,
            WebkitBoxOrient: "vertical",
            overflow: "hidden",
            fontFamily: "system-ui, -apple-system, sans-serif",
          }}
          title={filename}
        >
          {filename}
        </p>
      )}
    </div>
  );
}
