import { memo, useRef, useState } from 'react'
import { C } from '../../theme'

export default memo(function CollectionPill({ value, suggestions = [], onChange }) {
  const inputRef = useRef(null)
  const [editing, setEditing] = useState(false)
  const [hover, setHover] = useState(false)

  const startEdit = () => {
    setEditing(true)
    requestAnimationFrame(() => inputRef.current?.focus())
  }

  if (editing) {
    return (
      <span style={pillFrame(C, true)}>
        <span style={labelStyle(C)}>Collection name</span>
        <input
          ref={inputRef}
          list="collection-pill-options"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onBlur={() => setEditing(false)}
          onKeyDown={(e) => { if (e.key === 'Enter' || e.key === 'Escape') setEditing(false) }}
          style={{
            background: 'transparent',
            border: 'none',
            outline: 'none',
            color: C.accent,
            fontSize: 11,
            fontWeight: 600,
            width: Math.max(56, (value?.length || 4) * 7),
          }}
        />
        <datalist id="collection-pill-options">
          {suggestions.map((c) => <option key={c.name} value={c.name} />)}
        </datalist>
      </span>
    )
  }

  return (
    <button
      type="button"
      onClick={startEdit}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{
        ...pillFrame(C, hover),
        border: 'none',
        cursor: 'pointer',
      }}
    >
      <span style={labelStyle(C)}>Saving to</span>
      <span style={{ color: C.accent, fontSize: 11, fontWeight: 600 }}>{value || 'default'}</span>
    </button>
  )
})

function pillFrame(C, accent) {
  return {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 5,
    padding: '4px 9px',
    borderRadius: 999,
    background: accent ? C.accentBg : C.bgSoft,
    border: `1px solid ${accent ? C.accentBorder : C.lineSoft}`,
    transition: 'background 0.12s, border-color 0.12s',
  }
}

function labelStyle(C) {
  return {
    fontSize: 10,
    color: C.inkMuted,
    letterSpacing: '0.04em',
  }
}
