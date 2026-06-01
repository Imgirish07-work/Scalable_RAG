import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  selectActiveConversation,
  selectActiveMessages,
  useConversationStore,
} from '../stores/conversationStore'
import { AssistantBubble, UserBubble } from '../components/RAGChatThread'
import ChatComposer from '../components/organisms/ChatComposer'
import Modal from '../components/Modal'
import FilePreview from '../components/FilePreview'
import { useToast } from '../components/Toast'
import { postQuery } from '../api/query'
import { listCollections } from '../api/collections'
import { C } from '../theme'

export default function ChatPage() {
  const toast = useToast()

  const conversations = useConversationStore((s) => s.conversations)
  const activeId = useConversationStore((s) => s.activeId)
  const activeConversation = useConversationStore(selectActiveConversation)
  const messages = useConversationStore(selectActiveMessages)
  const createConversation = useConversationStore((s) => s.createConversation)
  const setActive = useConversationStore((s) => s.setActive)
  const appendMessage = useConversationStore((s) => s.appendMessage)
  const replaceMessages = useConversationStore((s) => s.replaceMessages)
  const setCollection = useConversationStore((s) => s.setCollection)

  const [text, setText] = useState('')
  const [queryLoading, setQueryLoading] = useState(false)
  const [copiedId, setCopiedId] = useState(null)
  const [collections, setCollections] = useState([])
  const [viewer, setViewer] = useState(null)
  const endRef = useRef(null)

  useEffect(() => {
    if (activeId) return
    if (conversations.length > 0) {
      const newest = [...conversations].sort((a, b) => b.updatedAt - a.updatedAt)[0]
      setActive(newest.id)
    } else {
      createConversation()
    }
  }, [activeId, conversations, setActive, createConversation])

  useEffect(() => {
    listCollections()
      .then((r) => setCollections(r.collections || []))
      .catch(() => {})
  }, [])

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const collection = activeConversation?.collection || 'default'

  const handleCollectionChange = useCallback(
    (next) => activeId && setCollection(activeId, next),
    [activeId, setCollection],
  )

  const runQuery = useCallback(
    async (queryText, replaceFromIndex = null) => {
      if (!activeId) return
      setQueryLoading(true)
      try {
        const resp = await postQuery({ query: queryText, collection })
        const sources = (resp.sources || []).map((s) => ({
          filename: s.file_name || s.doc_id || 'unknown',
        }))
        const msg = {
          id: Date.now() + 1,
          role: 'assistant',
          content: resp.answer || '(no answer)',
          sources,
        }
        const current = useConversationStore.getState().messages[activeId] || []
        const next =
          replaceFromIndex != null
            ? [...current.slice(0, replaceFromIndex), msg]
            : [...current, msg]
        replaceMessages(activeId, next)
      } catch (e) {
        const detail = e?.response?.data?.detail || e?.message || 'Query failed'
        const errMsg = { id: Date.now() + 1, role: 'assistant', content: detail, error: true }
        const current = useConversationStore.getState().messages[activeId] || []
        const next =
          replaceFromIndex != null
            ? [...current.slice(0, replaceFromIndex), errMsg]
            : [...current, errMsg]
        replaceMessages(activeId, next)
      } finally {
        setQueryLoading(false)
      }
    },
    [activeId, collection, replaceMessages],
  )

  const handleSubmit = useCallback(async () => {
    const value = text.trim()
    if (!value || queryLoading || !activeId) return
    setText('')
    appendMessage(activeId, { id: Date.now(), role: 'user', content: value })
    await runQuery(value)
  }, [text, queryLoading, activeId, appendMessage, runQuery])

  const handleRetry = useCallback(
    async (assistantMsg) => {
      const all = useConversationStore.getState().messages[activeId] || []
      const idx = all.findIndex((m) => m.id === assistantMsg.id)
      if (idx < 1) return
      const prevUser = all[idx - 1]
      if (prevUser?.role !== 'user') return
      await runQuery(prevUser.content, idx)
    },
    [activeId, runQuery],
  )

  const handleEdit = useCallback(
    async (userMsg, newContent) => {
      const all = useConversationStore.getState().messages[activeId] || []
      const idx = all.findIndex((m) => m.id === userMsg.id)
      if (idx < 0) return
      const updated = [...all]
      updated[idx] = { ...userMsg, content: newContent }
      if (updated[idx + 1]?.role === 'assistant') updated.splice(idx + 1, 1)
      replaceMessages(activeId, updated)
      await runQuery(newContent, idx + 1)
    },
    [activeId, replaceMessages, runQuery],
  )

  const handleCopy = useCallback(async (id, copyText) => {
    try {
      await navigator.clipboard.writeText(copyText)
      setCopiedId(id)
      setTimeout(() => setCopiedId(null), 2000)
    } catch {
      toast.error('Copy failed', 'Clipboard not available')
    }
  }, [toast])

  const handlePreviewJob = useCallback((job) => {
    if (!job?.file) return
    setViewer({ file: job.file, filename: job.filename, mimeType: job.mimeType })
  }, [])

  const isEmpty = useMemo(
    () => messages.length === 0 && !queryLoading,
    [messages, queryLoading],
  )

  return (
    <>
      <div
        className="hide-scrollbar"
        style={{
          flex: 1,
          minHeight: 0,
          overflowY: 'auto',
          display: 'flex',
          justifyContent: 'center',
          scrollbarWidth: 'none',
          msOverflowStyle: 'none',
        }}
      >
        <div
          style={{
            flex: 1,
            maxWidth: 820,
            width: '100%',
            padding: '24px 24px 16px',
          }}
        >
          {isEmpty ? (
            <EmptyState />
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
              {messages.map((m) =>
                m.role === 'user' ? (
                  <UserBubble
                    key={m.id}
                    msg={m}
                    copiedId={copiedId}
                    onCopy={handleCopy}
                    onEdit={handleEdit}
                    disabled={queryLoading}
                  />
                ) : (
                  <AssistantBubble
                    key={m.id}
                    msg={m}
                    copiedId={copiedId}
                    onCopy={handleCopy}
                    onRetry={handleRetry}
                    disabled={queryLoading}
                  />
                ),
              )}
              {queryLoading && (
                <p style={{ fontSize: 12, fontStyle: 'italic', color: C.inkMuted, margin: 0 }}>
                  Generating response…
                </p>
              )}
            </div>
          )}
          <div ref={endRef} />
        </div>
      </div>

      <ChatComposer
        conversationId={activeId}
        collection={collection}
        onCollectionChange={handleCollectionChange}
        collections={collections}
        value={text}
        onChange={setText}
        onSubmit={handleSubmit}
        isLoading={queryLoading}
        onPreviewJob={handlePreviewJob}
      />

      <Modal
        open={viewer != null}
        onClose={() => setViewer(null)}
        title={viewer?.filename || ''}
      >
        {viewer && (
          <FilePreview
            file={viewer.file}
            url={viewer.url}
            filename={viewer.filename}
            mimeType={viewer.mimeType}
          />
        )}
      </Modal>
    </>
  )
}

function EmptyState() {
  return (
    <div
      style={{
        height: '100%',
        minHeight: 360,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 8,
        textAlign: 'center',
      }}
    >
      <p style={{ fontSize: 24, fontWeight: 500, color: C.ink, margin: 0 }}>
        Ask anything about your documents
      </p>
      <p style={{ fontSize: 13, color: C.inkMuted, margin: 0 }}>
        Attach files with the + button or pick from your Library
      </p>
    </div>
  )
}
