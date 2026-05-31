import { useEffect } from 'react'
import { X } from 'lucide-react'
import { C, Z } from '../theme.js'


export default function Modal({ open, onClose, title, children, maxWidth = 960 }) {
  useEffect(() => {
    if (!open) return undefined
    const onKey = (e) => {
      if (e.key === 'Escape') onClose?.()
    }
    document.addEventListener('keydown', onKey)
    const prev = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      document.removeEventListener('keydown', onKey)
      document.body.style.overflow = prev
    }
  }, [open, onClose])

  if (!open) return null

  return (
    <div
      role="dialog"
      aria-modal="true"
      onClick={onClose}
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: Z.modal,
        background: 'rgba(0, 0, 0, 0.55)',
        backdropFilter: 'blur(2px)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: 24,
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          width: '100%',
          maxWidth,
          maxHeight: 'calc(100vh - 48px)',
          background: C.bgPanel,
          border: `1px solid ${C.line}`,
          borderRadius: 12,
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
        }}
      >
        <div
          className="flex items-center justify-between gap-3 px-4 py-3 flex-shrink-0"
          style={{ borderBottom: `1px solid ${C.lineSoft}` }}
        >
          <span
            className="text-sm font-semibold truncate"
            style={{ color: C.ink }}
            title={typeof title === 'string' ? title : undefined}
          >
            {title}
          </span>
          <button
            onClick={onClose}
            title="Close"
            className="flex items-center justify-center rounded p-1"
            style={{
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              color: C.inkMuted,
              flexShrink: 0,
            }}
            onMouseEnter={(e) => { e.currentTarget.style.color = C.ink }}
            onMouseLeave={(e) => { e.currentTarget.style.color = C.inkMuted }}
          >
            <X size={16} />
          </button>
        </div>

        <div className="flex-1 overflow-auto" style={{ background: C.bg }}>
          {children}
        </div>
      </div>
    </div>
  )
}
