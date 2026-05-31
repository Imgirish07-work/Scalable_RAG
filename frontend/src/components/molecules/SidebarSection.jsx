import { memo } from 'react'
import { C } from '../../theme'

export default memo(function SidebarSection({ title, action, children }) {
  return (
    <section style={{ display: 'flex', flexDirection: 'column', gap: 4, minHeight: 0 }}>
      {title && (
        <header
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '8px 10px 4px',
          }}
        >
          <span
            style={{
              fontSize: 10,
              fontWeight: 600,
              letterSpacing: '0.08em',
              color: C.inkMuted,
              textTransform: 'uppercase',
            }}
          >
            {title}
          </span>
          {action}
        </header>
      )}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 1 }}>{children}</div>
    </section>
  )
})
