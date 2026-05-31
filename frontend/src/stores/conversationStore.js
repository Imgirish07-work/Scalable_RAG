import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'

const newId = () => `c_${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`
const now = () => Date.now()

const deriveTitle = (text = '') => {
  const trimmed = text.trim().replace(/\s+/g, ' ')
  if (!trimmed) return 'New chat'
  return trimmed.length > 48 ? `${trimmed.slice(0, 45)}…` : trimmed
}

export const useConversationStore = create(
  persist(
    (set, get) => ({
      conversations: [],
      messages: {},
      activeId: null,

      createConversation(collection = 'default') {
        const id = newId()
        const convo = {
          id,
          title: 'New chat',
          starred: false,
          collection,
          createdAt: now(),
          updatedAt: now(),
        }
        set((s) => ({
          conversations: [convo, ...s.conversations],
          messages: { ...s.messages, [id]: [] },
          activeId: id,
        }))
        return id
      },

      setActive(id) {
        set({ activeId: id })
      },

      renameConversation(id, title) {
        set((s) => ({
          conversations: s.conversations.map((c) =>
            c.id === id ? { ...c, title, updatedAt: now() } : c,
          ),
        }))
      },

      toggleStarred(id) {
        set((s) => ({
          conversations: s.conversations.map((c) =>
            c.id === id ? { ...c, starred: !c.starred } : c,
          ),
        }))
      },

      setCollection(id, collection) {
        set((s) => ({
          conversations: s.conversations.map((c) =>
            c.id === id ? { ...c, collection } : c,
          ),
        }))
      },

      deleteConversation(id) {
        set((s) => {
          const conversations = s.conversations.filter((c) => c.id !== id)
          const { [id]: _, ...messages } = s.messages
          const activeId = s.activeId === id ? conversations[0]?.id ?? null : s.activeId
          return { conversations, messages, activeId }
        })
      },

      appendMessage(conversationId, message) {
        set((s) => {
          const list = s.messages[conversationId] || []
          const next = [...list, message]
          const convo = s.conversations.find((c) => c.id === conversationId)
          const shouldTitle = convo && convo.title === 'New chat' && message.role === 'user'
          return {
            messages: { ...s.messages, [conversationId]: next },
            conversations: s.conversations.map((c) =>
              c.id === conversationId
                ? {
                    ...c,
                    updatedAt: now(),
                    title: shouldTitle ? deriveTitle(message.content) : c.title,
                  }
                : c,
            ),
          }
        })
      },

      replaceMessages(conversationId, messages) {
        set((s) => ({
          messages: { ...s.messages, [conversationId]: messages },
          conversations: s.conversations.map((c) =>
            c.id === conversationId ? { ...c, updatedAt: now() } : c,
          ),
        }))
      },
    }),
    {
      name: 'scalable-rag-conversations',
      storage: createJSONStorage(() => localStorage),
      version: 1,
    },
  ),
)

export function selectActiveConversation(state) {
  return state.conversations.find((c) => c.id === state.activeId) || null
}

export function selectActiveMessages(state) {
  return state.activeId ? state.messages[state.activeId] || [] : []
}
