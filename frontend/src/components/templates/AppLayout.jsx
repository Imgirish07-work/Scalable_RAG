import { useState } from 'react'
import { Outlet } from 'react-router-dom'
import { PanelLeftOpen } from 'lucide-react'
import Sidebar from '../organisms/Sidebar'
import { C } from '../../theme'

export default function AppLayout() {
  const [collapsed, setCollapsed] = useState(false)

  return (
    <div style={{ display: 'flex', height: '100vh', background: C.bg, color: C.ink }}>
      {!collapsed && <Sidebar onCollapse={() => setCollapsed(true)} />}
      <main
        style={{
          flex: 1,
          minWidth: 0,
          display: 'flex',
          flexDirection: 'column',
          position: 'relative',
        }}
      >
        {collapsed && (
          <button
            type="button"
            onClick={() => setCollapsed(false)}
            title="Show sidebar"
            style={{
              position: 'absolute',
              top: 12,
              left: 12,
              zIndex: 5,
              display: 'grid',
              placeItems: 'center',
              width: 28,
              height: 28,
              background: C.bgSoft,
              border: `1px solid ${C.lineSoft}`,
              borderRadius: 6,
              color: C.inkSoft,
              cursor: 'pointer',
              padding: 0,
            }}
          >
            <PanelLeftOpen size={15} />
          </button>
        )}
        <Outlet />
      </main>
    </div>
  )
}
