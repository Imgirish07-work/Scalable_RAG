import { useEffect, useMemo, useRef, useState } from 'react'
import { Badge, Icon } from '../components/atoms'
import { PanelHeader } from '../components/molecules'
import FileCard from '../components/molecules/FileCard.jsx'
import QueryBox from '../components/molecules/QueryBox.jsx'
import RAGUploadZone from '../components/RAGUploadZone.jsx'
import {
  AssistantBubble,
  UserBubble,
} from '../components/RAGChatThread.jsx'
import TwoPaneLayout from '../components/templates/TwoPaneLayout.jsx'
import { useToast } from '../components/Toast.jsx'
import { useUploadStore } from '../stores/uploadStore.js'
import {
  deleteDocument,
  listDocuments,
  retryIngestion,
} from '../api/documents.js'
import { listCollections } from '../api/collections.js'
import { postQuery } from '../api/query.js'
import { apiBaseUrl } from '../api/client.js'
import { openDocumentEventStream } from '../api/sse.js'
import { C } from '../theme.js'
import closeIcon from '../Assets/svg/close.svg'
import lightbulbIcon from '../Assets/svg/lightbulb.svg'
import documentIcon from '../Assets/svg/document.svg'
import searchIcon from '../Assets/svg/search.svg'
import gearIcon from '../Assets/svg/gear.svg'

const SUPPORTED_EXTENSIONS = ['PDF', 'TXT', 'MD', 'DOCX', 'PPTX']

const GUIDE_STEPS = [
  { icon: documentIcon, title: 'Upload',     desc: 'Drop files in the right panel, then click Upload to start indexing.' },
  { icon: gearIcon,     title: 'Collection', desc: 'Group related docs by typing a collection name above the upload zone.' },
  { icon: searchIcon,   title: 'Ask',        desc: 'Type a question once at least one document is indexed and ready.' },
  { icon: lightbulbIcon,title: 'Refine',     desc: 'Iterate — the chat history stays during your session.' },
]

export default function MainPage() {
  const toast = useToast()

  const [collection,    setCollection]    = useState('default')
  const [collections,   setCollections]   = useState([])
  const [messages,      setMessages]      = useState([])
  const [query,         setQuery]         = useState('')
  const [queryLoading,  setQueryLoading]  = useState(false)
  const [copiedId,      setCopiedId]      = useState(null)
  const [indexedDocs,   setIndexedDocs]   = useState([])
  const [guideOpen,     setGuideOpen]     = useState(true)
  const answerEndRef = useRef(null)

  const jobs          = useUploadStore((s) => s.jobs)
  const addFiles      = useUploadStore((s) => s.addFiles)
  const startUploads  = useUploadStore((s) => s.start)
  const removeJob     = useUploadStore((s) => s.removeJob)

  const queuedCount = useMemo(
    () => jobs.filter((j) => j.status === 'queued').length,
    [jobs],
  )
  const uploadingJob = useMemo(
    () => jobs.find((j) => j.status === 'uploading') || null,
    [jobs],
  )
  // Count jobs in ANY terminal state — both ready and failed need to surface
  // in the indexed-docs list (so failed uploads are visible + retryable).
  const terminalCount = useMemo(
    () => jobs.filter((j) => j.status === 'ready' || j.status === 'failed').length,
    [jobs],
  )

  useEffect(() => {
    refreshDocuments()
    refreshCollections()
  }, [])

  useEffect(() => {
    if (terminalCount > 0) {
      refreshDocuments()
      refreshCollections()
    }
  }, [terminalCount])

  useEffect(() => {
    answerEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  async function refreshDocuments() {
    try {
      const resp = await listDocuments()
      setIndexedDocs(resp.documents || [])
    } catch {
      // List-fetch failures are non-fatal; the user can retry by re-uploading.
    }
  }

  async function refreshCollections() {
    try {
      const resp = await listCollections()
      setCollections(resp.collections || [])
    } catch {
      // Non-fatal — the input still works as a free-text field.
    }
  }

  function handleAddFiles(files) {
    addFiles(files, collection)
  }

  async function handleStart() {
    try {
      await startUploads()
    } catch (e) {
      toast.error('Upload failed', e?.message || 'Unknown error')
    }
  }

  async function runQuery(text, replaceFromIndex = null) {
    setQueryLoading(true)
    try {
      const resp = await postQuery({
        query: text,
        collection: collection || undefined,
      })
      const sources = (resp.sources || []).map((s) => ({
        filename: s.file_name || s.doc_id || 'unknown',
      }))
      const assistantMsg = {
        id: Date.now() + 1,
        role: 'assistant',
        content: resp.answer || '(no answer)',
        sources,
      }
      setMessages((m) =>
        replaceFromIndex != null
          ? [...m.slice(0, replaceFromIndex), assistantMsg]
          : [...m, assistantMsg],
      )
    } catch (e) {
      const msg = e?.response?.data?.detail || e?.message || 'Query failed'
      const errorMsg = { id: Date.now() + 1, role: 'assistant', content: msg, error: true }
      setMessages((m) =>
        replaceFromIndex != null
          ? [...m.slice(0, replaceFromIndex), errorMsg]
          : [...m, errorMsg],
      )
    } finally {
      setQueryLoading(false)
    }
  }

  async function handleQuery() {
    const text = query.trim()
    if (!text || queryLoading) return
    setQuery('')
    setMessages((m) => [...m, { id: Date.now(), role: 'user', content: text }])
    await runQuery(text)
  }

  // Drop the assistant msg and re-run the user message immediately before it.
  async function handleRetry(assistantMsg) {
    const idx = messages.findIndex((m) => m.id === assistantMsg.id)
    if (idx < 1) return
    const prevUser = messages[idx - 1]
    if (prevUser?.role !== 'user') return
    await runQuery(prevUser.content, idx)
  }

  // Replace the user msg content + drop its assistant response, then re-query.
  async function handleEdit(userMsg, newContent) {
    const idx = messages.findIndex((m) => m.id === userMsg.id)
    if (idx < 0) return
    const updated = [...messages]
    updated[idx] = { ...userMsg, content: newContent }
    // Drop the immediately-following assistant msg if present.
    if (updated[idx + 1]?.role === 'assistant') updated.splice(idx + 1, 1)
    setMessages(updated)
    await runQuery(newContent, idx + 1)
  }

  async function handleCopy(id, text) {
    try {
      await navigator.clipboard.writeText(text)
      setCopiedId(id)
      setTimeout(() => setCopiedId(null), 2000)
    } catch {
      // Clipboard write blocked (no HTTPS, no permission) — silently skip.
    }
  }

  async function handleDeleteIndexed(docId) {
    try {
      await deleteDocument(docId)
      setIndexedDocs((ds) => ds.filter((d) => d.doc_id !== docId))
    } catch (e) {
      toast.error('Delete failed', e?.response?.data?.detail || e?.message)
    }
  }

  // Re-run ingestion on a failed document using its preserved MinIO blob
  // (backend DLQ retry). Refresh immediately to flip the status to processing,
  // then subscribe to SSE for the next terminal phase and refresh again.
  async function handleRetryIndexed(docId) {
    try {
      await retryIngestion(docId)
      toast.success('Retry started', 'Re-running ingestion in the background')
      refreshDocuments()
      const close = openDocumentEventStream({
        baseUrl: apiBaseUrl,
        docId,
        onEvent: (event) => {
          if (event.phase === 'ready' || event.phase === 'failed') {
            refreshDocuments()
            close()
          }
        },
        onError: () => close(),
      })
    } catch (e) {
      toast.error('Retry failed', e?.response?.data?.detail || e?.message)
    }
  }

  const hasDocuments = indexedDocs.length > 0

  return (
    <TwoPaneLayout
      className="h-full"
      rightWidth={360}

      left={<>
        {guideOpen && (
          <div className="px-5 pt-4 pb-2 flex-shrink-0">
            <div
              className="rounded-xl p-4"
              style={{
                background: C.bgDeep,
                border: `1px solid ${C.accentBorder}`,
              }}
            >
              <div className="flex items-start justify-between mb-3">
                <div className="flex items-center gap-2">
                  <Icon src={lightbulbIcon} className="w-4 h-4" style={{ color: C.accent }} />
                  <span className="text-sm font-semibold" style={{ color: C.accent }}>
                    How to use this app
                  </span>
                </div>
                <button
                  onClick={() => setGuideOpen(false)}
                  className="flex items-center gap-1 text-xs"
                  style={{ background: 'none', border: 'none', cursor: 'pointer', color: C.inkMuted }}
                >
                  <Icon src={closeIcon} className="w-3 h-3" />
                  Dismiss
                </button>
              </div>
              <div className="grid grid-cols-4 gap-3">
                {GUIDE_STEPS.map((step, i) => (
                  <div
                    key={i}
                    className="flex flex-col items-center text-center gap-1.5 rounded-lg p-3"
                    style={{
                      background: C.bgPanel,
                      border: `1px solid ${C.accentBorder}`,
                    }}
                  >
                    <Icon src={step.icon} className="w-5 h-5" style={{ color: C.accent }} />
                    <p className="text-xs font-semibold" style={{ color: C.ink }}>
                      {step.title}
                    </p>
                    <p className="text-xs leading-relaxed" style={{ color: C.inkMuted }}>
                      {step.desc}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        <div className="flex-1 overflow-y-auto min-h-0 relative">
          {messages.length === 0 && !queryLoading ? (
            <EmptyState
              hasDocuments={hasDocuments}
              docCount={indexedDocs.length}
            />
          ) : (
            <div className="px-8 py-6 flex flex-col gap-6">
              {messages.map((msg) =>
                msg.role === 'user' ? (
                  <UserBubble
                    key={msg.id}
                    msg={msg}
                    copiedId={copiedId}
                    onCopy={handleCopy}
                    onEdit={handleEdit}
                    disabled={queryLoading}
                  />
                ) : (
                  <AssistantBubble
                    key={msg.id}
                    msg={msg}
                    copiedId={copiedId}
                    onCopy={handleCopy}
                    onRetry={handleRetry}
                    disabled={queryLoading}
                  />
                ),
              )}
              {queryLoading && (
                <div className="text-xs italic" style={{ color: C.inkMuted }}>
                  Generating response…
                </div>
              )}
            </div>
          )}
          <div ref={answerEndRef} />
        </div>

        <QueryBox
          value={query}
          onChange={setQuery}
          onSubmit={handleQuery}
          isLoading={queryLoading}
          disabled={!hasDocuments}
          placeholder={
            hasDocuments
              ? 'Ask anything about your documents'
              : 'Upload a document first to start querying'
          }
        />
      </>}

      right={<>
        <PanelHeader title="Documents" border />

        <div className="px-3 pt-3 flex-shrink-0">
          <input
            type="text"
            value={collection}
            onChange={(e) => setCollection(e.target.value)}
            placeholder="Collection"
            list="collections-list"
            className="w-full text-xs px-3 py-2 rounded-lg mb-3 outline-none"
            style={{
              background: C.bgInput,
              border: `1px solid ${C.lineCard}`,
              color: C.ink,
            }}
          />
          <datalist id="collections-list">
            {collections.map((c) => (
              <option key={c.name} value={c.name} />
            ))}
          </datalist>

          <RAGUploadZone
            onAddFiles={handleAddFiles}
            onStart={handleStart}
            stagedCount={queuedCount}
            activeJob={
              uploadingJob && {
                filename: uploadingJob.filename,
                progress: uploadingJob.progress,
                message: uploadingJob.message,
              }
            }
          />
        </div>

        {jobs.length > 0 && (
          <div className="px-3 pt-3 pb-2 flex-shrink-0" style={{ maxHeight: 110 }}>
            <div
              className="flex gap-2 overflow-x-auto"
              style={{ scrollbarWidth: 'thin', paddingBottom: 4 }}
            >
              {jobs.map((j) => (
                <FileCard
                  key={j.id}
                  file={{ filename: j.filename }}
                  onRemove={() => removeJob(j.id)}
                />
              ))}
            </div>
          </div>
        )}

        <div
          className="flex-1 overflow-y-auto min-h-0"
          style={{ borderTop: `1px solid ${C.lineSoft}` }}
        >
          <PanelHeader title="Indexed" />
          {indexedDocs.length === 0 ? (
            <div className="flex flex-col items-center gap-1.5 px-3 pb-4">
              <span className="text-xs" style={{ color: C.inkMuted }}>
                Supported formats
              </span>
              <div className="flex flex-wrap gap-1 justify-center">
                {SUPPORTED_EXTENSIONS.map((ext) => (
                  <Badge key={ext} variant="default" size="sm">{ext}</Badge>
                ))}
              </div>
            </div>
          ) : (
            <div className="flex flex-col gap-1.5 px-3 pb-3">
              {indexedDocs.map((d) => (
                <IndexedDocRow
                  key={d.doc_id}
                  doc={d}
                  onDelete={handleDeleteIndexed}
                  onRetry={handleRetryIndexed}
                />
              ))}
            </div>
          )}
        </div>
      </>}
    />
  )
}

function EmptyState({ hasDocuments, docCount }) {
  return (
    <div className="flex flex-col items-center justify-center h-full text-center gap-3 select-none px-8">
      <p className="text-lg font-semibold" style={{ color: C.ink }}>
        Ask anything about your documents
      </p>
      <p className="text-sm" style={{ color: C.inkMuted }}>
        {hasDocuments
          ? `${docCount} document${docCount !== 1 ? 's' : ''} stored · ask your query below`
          : 'Upload files on the right to get started'}
      </p>
    </div>
  )
}

function IndexedDocRow({ doc, onDelete, onRetry }) {
  const isFailed = doc.status === 'failed'
  return (
    <div
      className="flex items-center gap-2 px-2 py-1.5 rounded"
      style={{ background: C.bgCard, border: `1px solid ${C.lineSoft}` }}
    >
      <span
        className="text-xs flex-1 min-w-0 truncate"
        style={{ color: C.ink }}
        title={doc.file_name}
      >
        {doc.file_name}
      </span>
      <span
        className="text-[10px] px-1.5 py-0.5 rounded"
        style={{
          background:
            doc.status === 'ready' ? C.accentBg
              : isFailed ? 'rgba(220,38,38,0.1)'
              : C.bgSoft,
          color:
            doc.status === 'ready' ? C.accent
              : isFailed ? C.danger
              : C.inkMuted,
        }}
      >
        {doc.status}
      </span>
      {isFailed && (
        <button
          onClick={() => onRetry(doc.doc_id)}
          title={doc.error_message || 'Retry ingestion'}
          className="text-[10px] px-1.5 py-0.5 rounded"
          style={{
            background: 'none',
            border: `1px solid ${C.accentBorder}`,
            cursor: 'pointer',
            color: C.accent,
          }}
        >
          retry
        </button>
      )}
      <button
        onClick={() => onDelete(doc.doc_id)}
        title="Delete"
        style={{
          background: 'none',
          border: 'none',
          cursor: 'pointer',
          color: C.inkMuted,
          fontSize: 14,
          lineHeight: 1,
          padding: 2,
        }}
      >
        ×
      </button>
    </div>
  )
}
