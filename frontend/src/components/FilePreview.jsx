import { useEffect, useRef, useState } from 'react'
import { loadPdfJs } from '../utils/pdfjs.js'
import { C } from '../theme.js'

const TEXT_EXTS = new Set(['txt', 'md', 'csv', 'json', 'log', 'xml', 'yml', 'yaml'])
const IMAGE_EXTS = new Set(['png', 'jpg', 'jpeg', 'gif', 'webp', 'svg', 'bmp'])

function detectKind(filename = '', mime = '') {
  const ext = (filename.split('.').pop() || '').toLowerCase()
  if (ext === 'pdf' || mime === 'application/pdf') return 'pdf'
  if (IMAGE_EXTS.has(ext) || mime.startsWith('image/')) return 'image'
  if (TEXT_EXTS.has(ext) || mime.startsWith('text/')) return 'text'
  if (ext === 'docx') return 'docx'
  return 'unknown'
}


export default function FilePreview({ file, url, filename = '', mimeType = '' }) {
  const displayName = filename || file?.name || ''
  const kind = detectKind(displayName, mimeType || file?.type || '')

  const [text, setText] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const pdfRef = useRef(null)

  // local File becomes an object URL; cleanup revokes the blob URL on unmount
  const [resolvedUrl, setResolvedUrl] = useState(null)
  useEffect(() => {
    if (file) {
      const blobUrl = URL.createObjectURL(file)
      setResolvedUrl(blobUrl)
      return () => URL.revokeObjectURL(blobUrl)
    }
    setResolvedUrl(url || null)
    return undefined
  }, [file, url])

  useEffect(() => {
    if (kind !== 'text') return undefined
    let cancelled = false
    setLoading(true)
    setError('')
    setText('')
    const load = async () => {
      try {
        if (file) {
          const buf = await file.arrayBuffer()
          if (!cancelled) setText(new TextDecoder().decode(new Uint8Array(buf)))
        } else if (url) {
          const resp = await fetch(url)
          const body = await resp.text()
          if (!cancelled) setText(body)
        }
      } catch (e) {
        if (!cancelled) setError(String(e?.message || e))
      } finally {
        if (!cancelled) setLoading(false)
      }
    }
    load()
    return () => { cancelled = true }
  }, [kind, file, url])

  useEffect(() => {
    if (kind !== 'pdf' || !resolvedUrl) return undefined
    let cancelled = false
    setLoading(true)
    setError('')
    const render = async () => {
      try {
        const lib = await loadPdfJs()
        let data
        if (file) {
          data = new Uint8Array(await file.arrayBuffer())
        }
        const task = data
          ? lib.getDocument({ data })
          : lib.getDocument({ url: resolvedUrl })
        const pdf = await task.promise
        if (cancelled || !pdfRef.current) return
        pdfRef.current.innerHTML = ''
        const dpr = window.devicePixelRatio || 1
        const scale = 1.4 * dpr
        for (let n = 1; n <= pdf.numPages; n++) {
          if (cancelled) return
          const page = await pdf.getPage(n)
          const vp = page.getViewport({ scale })
          const canvas = document.createElement('canvas')
          canvas.width = vp.width
          canvas.height = vp.height
          Object.assign(canvas.style, {
            width: `${vp.width / dpr}px`,
            maxWidth: '100%',
            display: 'block',
            margin: '0 auto 18px',
            background: '#fff',
            borderRadius: '6px',
            boxShadow: '0 1px 2px rgba(0,0,0,0.15), 0 8px 24px rgba(0,0,0,0.12)',
          })
          await page.render({
            canvasContext: canvas.getContext('2d'),
            viewport: vp,
          }).promise
          if (!cancelled) pdfRef.current.appendChild(canvas)
        }
        if (!cancelled) setLoading(false)
      } catch (e) {
        if (!cancelled) {
          setError(String(e?.message || e))
          setLoading(false)
        }
      }
    }
    render()
    return () => { cancelled = true }
  }, [kind, resolvedUrl, file])

  if (loading) {
    return (
      <div
        className="flex items-center justify-center h-64"
        style={{ color: C.inkMuted, fontSize: 13 }}
      >
        Opening…
      </div>
    )
  }

  if (error) {
    return (
      <div
        className="flex flex-col items-center justify-center h-64 text-center px-6"
        style={{ color: C.inkMuted, fontSize: 13 }}
      >
        <p>Couldn't render this file.</p>
        <p className="mt-1 text-[11px]" style={{ color: 'var(--c-textError)' }}>
          {error}
        </p>
      </div>
    )
  }

  if (kind === 'image' && resolvedUrl) {
    return (
      <div className="flex items-center justify-center p-6">
        <img
          src={resolvedUrl}
          alt={displayName}
          style={{ maxWidth: '100%', maxHeight: '80vh', borderRadius: 6 }}
        />
      </div>
    )
  }

  if (kind === 'text') {
    return (
      <pre
        className="p-6 text-xs whitespace-pre-wrap"
        style={{
          color: C.ink,
          fontFamily: 'ui-monospace, "SF Mono", Menlo, monospace',
          lineHeight: 1.6,
          margin: 0,
        }}
      >
        {text}
      </pre>
    )
  }

  if (kind === 'pdf') {
    return <div ref={pdfRef} className="p-6" />
  }

  if (kind === 'docx') {
    return (
      <div
        className="flex flex-col items-center justify-center h-64 text-center px-6"
        style={{ color: C.inkMuted, fontSize: 13 }}
      >
        <p>DOCX preview not enabled.</p>
        <p className="mt-1 text-[11px]">
          Install <code>mammoth</code> to render Word documents in-app.
        </p>
      </div>
    )
  }

  return (
    <div
      className="flex items-center justify-center h-64 text-center px-6"
      style={{ color: C.inkMuted, fontSize: 13 }}
    >
      <p>No preview available for this file type.</p>
    </div>
  )
}
