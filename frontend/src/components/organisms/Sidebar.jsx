import { useMemo, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { Library, Moon, PanelLeftClose, Plus, Sun } from 'lucide-react'
import SidebarNavItem from '../atoms/SidebarNavItem'
import ConversationListItem from '../atoms/ConversationListItem'
import SidebarSection from '../molecules/SidebarSection'
import { useConversationStore } from '../../stores/conversationStore'
import { useTheme } from '../ThemeProvider'
import { C } from '../../theme'

const PAGE_SIZE = 10

export default function Sidebar({ onCollapse }) {
  const navigate = useNavigate()
  const location = useLocation()
  const { id: themeId, toggle: toggleTheme } = useTheme()

  const conversations = useConversationStore((s) => s.conversations)
  const activeId = useConversationStore((s) => s.activeId)
  const createConversation = useConversationStore((s) => s.createConversation)
  const setActive = useConversationStore((s) => s.setActive)
  const toggleStarred = useConversationStore((s) => s.toggleStarred)
  const deleteConversation = useConversationStore((s) => s.deleteConversation)

  const [recentLimit, setRecentLimit] = useState(PAGE_SIZE)

  const sorted = useMemo(
    () => [...conversations].sort((a, b) => b.updatedAt - a.updatedAt),
    [conversations],
  )
  const recent = useMemo(() => sorted.slice(0, recentLimit), [sorted, recentLimit])
  const starred = useMemo(() => sorted.filter((c) => c.starred), [sorted])

  const onChatRoute = location.pathname === '/'
  const onLibraryRoute = location.pathname === '/library'
  const isDark = themeId === 'claude-dark'

  const handleNewChat = () => {
    createConversation()
    if (!onChatRoute) navigate('/')
  }

  const handleSelectConversation = (id) => {
    setActive(id)
    if (!onChatRoute) navigate('/')
  }

  return (
    <aside
      style={{
        width: 260,
        flexShrink: 0,
        display: 'flex',
        flexDirection: 'column',
        background: C.bgPanel,
        borderRight: `1px solid ${C.lineSoft}`,
      }}
    >
      <Brand onCollapse={onCollapse} />

      <div style={{ display: 'flex', flexDirection: 'column', gap: 2, padding: '0 8px 6px' }}>
        <SidebarNavItem icon={Plus} label="New chat" onClick={handleNewChat} />
        <SidebarNavItem
          icon={Library}
          label="Library"
          active={onLibraryRoute}
          onClick={() => navigate('/library')}
        />
      </div>

      <div style={{ flex: 1, minHeight: 0, overflowY: 'auto', padding: '4px 8px 8px' }}>
        {starred.length > 0 && (
          <SidebarSection title="Starred">
            {starred.map((c) => (
              <ConversationListItem
                key={c.id}
                title={c.title}
                active={c.id === activeId && onChatRoute}
                starred={c.starred}
                onSelect={() => handleSelectConversation(c.id)}
                onToggleStar={() => toggleStarred(c.id)}
                onDelete={() => deleteConversation(c.id)}
              />
            ))}
          </SidebarSection>
        )}

        <div style={{ marginTop: starred.length > 0 ? 12 : 0 }}>
          <SidebarSection title="Recent">
            {recent.length === 0 ? (
              <EmptyHint text="No conversations yet" />
            ) : (
              recent.map((c) => (
                <ConversationListItem
                  key={c.id}
                  title={c.title}
                  active={c.id === activeId && onChatRoute}
                  starred={c.starred}
                  onSelect={() => handleSelectConversation(c.id)}
                  onToggleStar={() => toggleStarred(c.id)}
                  onDelete={() => deleteConversation(c.id)}
                />
              ))
            )}
            {sorted.length > recentLimit && (
              <button
                type="button"
                onClick={() => setRecentLimit((n) => n + PAGE_SIZE)}
                style={{
                  marginTop: 4,
                  padding: '4px 10px',
                  background: 'none',
                  border: 'none',
                  color: C.inkMuted,
                  fontSize: 11,
                  cursor: 'pointer',
                  textAlign: 'left',
                }}
              >
                Show more…
              </button>
            )}
          </SidebarSection>
        </div>
      </div>

      <UserFooter isDark={isDark} onToggleTheme={toggleTheme} />
    </aside>
  )
}

function Brand({ onCollapse }) {
  return (
    <header
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '14px 14px 10px',
      }}
    >
      <span style={{ fontSize: 14, fontWeight: 600, color: C.ink, letterSpacing: '-0.01em' }}>
        Scalable RAG
      </span>
      {onCollapse && (
        <button
          type="button"
          onClick={onCollapse}
          title="Hide sidebar"
          style={{
            display: 'grid',
            placeItems: 'center',
            width: 24,
            height: 24,
            background: 'none',
            border: 'none',
            color: C.inkMuted,
            cursor: 'pointer',
            padding: 0,
          }}
        >
          <PanelLeftClose size={14} />
        </button>
      )}
    </header>
  )
}

function EmptyHint({ text }) {
  return (
    <p style={{ margin: 0, padding: '6px 10px', fontSize: 11, color: C.inkMuted }}>
      {text}
    </p>
  )
}

function UserFooter({ isDark, onToggleTheme }) {
  const initial = 'D'
  const name = 'dev-user'
  return (
    <footer
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        padding: '10px 12px',
        borderTop: `1px solid ${C.lineSoft}`,
      }}
    >
      <div
        style={{
          width: 26,
          height: 26,
          borderRadius: '50%',
          background: C.accentBg,
          color: C.accent,
          display: 'grid',
          placeItems: 'center',
          fontSize: 11,
          fontWeight: 600,
          flexShrink: 0,
        }}
      >
        {initial}
      </div>
      <span
        style={{
          flex: 1,
          fontSize: 12,
          color: C.inkSoft,
          minWidth: 0,
          whiteSpace: 'nowrap',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
        }}
      >
        {name}
      </span>
      <button
        type="button"
        onClick={onToggleTheme}
        title={isDark ? 'Switch to light' : 'Switch to dark'}
        style={{
          display: 'grid',
          placeItems: 'center',
          width: 24,
          height: 24,
          background: 'none',
          border: 'none',
          color: C.inkMuted,
          cursor: 'pointer',
          padding: 0,
        }}
      >
        {isDark ? <Sun size={14} /> : <Moon size={14} />}
      </button>
    </footer>
  )
}
