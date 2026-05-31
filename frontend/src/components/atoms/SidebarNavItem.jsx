import { memo, useState } from 'react'
import { C } from '../../theme'

export default memo(function SidebarNavItem({
  icon: Icon,
  label,
  active = false,
  onClick,
  trailing,
}) {
  const [hover, setHover] = useState(false)
  const background = active
    ? C.accentBg
    : hover
      ? C.bgSoft
      : 'transparent'
  const color = active ? C.accent : hover ? C.ink : C.inkSoft

  return (
    <button
      type="button"
      onClick={onClick}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{
        width: '100%',
        display: 'flex',
        alignItems: 'center',
        gap: 10,
        padding: '7px 10px',
        borderRadius: 8,
        background,
        color,
        border: 'none',
        cursor: 'pointer',
        fontSize: 13,
        fontWeight: active ? 600 : 500,
        textAlign: 'left',
        transition: 'background 0.12s, color 0.12s',
      }}
    >
      {Icon && <Icon size={15} style={{ flexShrink: 0 }} />}
      <span style={{ flex: 1, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
        {label}
      </span>
      {trailing}
    </button>
  )
})
