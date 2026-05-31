import { memo, useState } from 'react'
import { Star, Trash2 } from 'lucide-react'
import { C } from '../../theme'

export default memo(function ConversationListItem({
  title,
  active = false,
  starred = false,
  onSelect,
  onToggleStar,
  onDelete,
}) {
  const [hover, setHover] = useState(false)
  const background = active ? C.accentBg : hover ? C.bgSoft : 'transparent'
  const color = active ? C.accent : C.inkSoft

  return (
    <div
      onClick={onSelect}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 6,
        padding: '6px 8px',
        borderRadius: 6,
        background,
        cursor: 'pointer',
        transition: 'background 0.12s',
      }}
    >
      <span
        style={{
          flex: 1,
          minWidth: 0,
          fontSize: 12.5,
          color,
          whiteSpace: 'nowrap',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
        }}
        title={title}
      >
        {title}
      </span>
      <RowIcon
        title={starred ? 'Unstar' : 'Star'}
        visible={hover || starred}
        active={starred}
        onClick={(e) => { e.stopPropagation(); onToggleStar?.() }}
      >
        <Star size={12} fill={starred ? 'currentColor' : 'none'} />
      </RowIcon>
      <RowIcon
        title="Delete"
        visible={hover}
        onClick={(e) => { e.stopPropagation(); onDelete?.() }}
        hoverColor={C.danger}
      >
        <Trash2 size={12} />
      </RowIcon>
    </div>
  )
})

function RowIcon({ children, title, visible, active, onClick, hoverColor }) {
  const [hover, setHover] = useState(false)
  const color = active ? C.accent : hover ? (hoverColor || C.ink) : C.inkMuted
  return (
    <button
      type="button"
      title={title}
      onClick={onClick}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{
        flexShrink: 0,
        display: 'grid',
        placeItems: 'center',
        width: 20,
        height: 20,
        background: 'none',
        border: 'none',
        padding: 0,
        cursor: 'pointer',
        color,
        opacity: visible ? 1 : 0,
        pointerEvents: visible ? 'auto' : 'none',
        transition: 'opacity 0.12s, color 0.12s',
      }}
    >
      {children}
    </button>
  )
}
