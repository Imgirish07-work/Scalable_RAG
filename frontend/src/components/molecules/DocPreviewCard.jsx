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

export default memo(function DocPreviewCard({
  file,
  fetchUrl,
  url,
  filename = "",
  label,
  width = 190,
  height = 234,
  status,
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
  const [thumbState, setThumbState] = useState("idle");
  const [visible, setVisible] = useState(!fetchUrl);
  const [hovered, setHovered] = useState(false);
  const isClickable = typeof onClick === "function";
  const isFailed = status === "failed";

  // Lazy-fetch presigned URL when card enters viewport; avoids 50 GETs on first render.
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

  useEffect(() => {
    if (!visible) return undefined;
    if (file) {
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

  const cardBadgeTone = kind === "pdf" || kind === "image" ? "dark" : "themed";
  return (
    <div
      ref={rootRef}
      onClick={isClickable ? onClick : undefined}
      title={displayName}
      style={{
        position: "relative",
        width,
        height,
        flexShrink: 0,
        borderRadius: 12,
        overflow: "hidden",
        background: C.bgCard,
        border: isFailed
          ? `1.5px solid var(--c-textError)`
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

        {isFailed && (
          <div
            style={{
              position: "absolute",
              inset: 0,
              background: "rgba(184, 61, 44, 0.12)",
              pointerEvents: "none",
            }}
          />
        )}
      </div>

      {(onRemove || (onRetry && isFailed)) && (
        <div
          style={{
            position: "absolute",
            top: 4,
            right: 4,
            display: "flex",
            gap: 3,
            zIndex: 2,
            opacity: hovered || isFailed ? 1 : 0,
            transition: "opacity 0.15s",
            pointerEvents: hovered || isFailed ? "auto" : "none",
          }}
        >
          {onRetry && isFailed && (
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
