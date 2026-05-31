import { useCallback, useEffect, useMemo, useState } from 'react'
import { RefreshCw } from 'lucide-react'
import { DocPreviewCard } from '../components/molecules'
import Modal from '../components/Modal'
import FilePreview from '../components/FilePreview'
import { useToast } from '../components/Toast'
import {
  deleteDocument,
  getDocumentDownloadUrl,
  listDocuments,
  retryIngestion,
} from '../api/documents'
import { listCollections } from '../api/collections'
import { apiBaseUrl } from '../api/client'
import { openDocumentEventStream } from '../api/sse'
import { C } from '../theme'

export default function LibraryPage() {
  const toast = useToast()

  const [docs, setDocs] = useState([])
  const [collections, setCollections] = useState([])
  const [filter, setFilter] = useState('')
  const [loading, setLoading] = useState(false)
  const [viewer, setViewer] = useState(null)

  const refresh = useCallback(async () => {
    setLoading(true)
    try {
      const [docResp, collResp] = await Promise.all([
        listDocuments({ collection: filter || undefined }),
        listCollections(),
      ])
      setDocs(docResp.documents || [])
      setCollections(collResp.collections || [])
    } catch (e) {
      toast.error('Load failed', e?.response?.data?.detail || e?.message)
    } finally {
      setLoading(false)
    }
  }, [filter, toast])

  useEffect(() => { refresh() }, [refresh])

  const visible = useMemo(
    () => (filter ? docs.filter((d) => d.collection === filter) : docs),
    [docs, filter],
  )

  const handleOpen = useCallback(async (doc) => {
    try {
      const { presigned_url } = await getDocumentDownloadUrl(doc.doc_id)
      setViewer({
        url: presigned_url,
        filename: doc.file_name,
        mimeType: doc.mime_type,
      })
    } catch (e) {
      toast.error('Open failed', e?.response?.data?.detail || e?.message)
    }
  }, [toast])

  const handleDelete = useCallback(async (docId) => {
    try {
      await deleteDocument(docId)
      setDocs((d) => d.filter((x) => x.doc_id !== docId))
    } catch (e) {
      toast.error('Delete failed', e?.response?.data?.detail || e?.message)
    }
  }, [toast])

  const handleRetry = useCallback(async (docId) => {
    try {
      await retryIngestion(docId)
      toast.success('Retry started', 'Re-running ingestion')
      refresh()
      const close = openDocumentEventStream({
        baseUrl: apiBaseUrl,
        docId,
        onEvent: (event) => {
          if (event.phase === 'ready' || event.phase === 'failed') {
            refresh()
            close()
          }
        },
        onError: () => close(),
      })
    } catch (e) {
      toast.error('Retry failed', e?.response?.data?.detail || e?.message)
    }
  }, [refresh, toast])

  return (
    <div
      style={{
        flex: 1,
        minHeight: 0,
        overflowY: 'auto',
        padding: '28px 32px 40px',
      }}
    >
      <Header
        count={visible.length}
        loading={loading}
        onRefresh={refresh}
      />

      <CollectionFilter
        value={filter}
        collections={collections}
        onChange={setFilter}
      />

      {visible.length === 0 ? (
        <EmptyState filtered={!!filter} />
      ) : (
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))',
            gap: 14,
            marginTop: 18,
          }}
        >
          {visible.map((d) => (
            <DocPreviewCard
              key={d.doc_id}
              fetchUrl={() =>
                getDocumentDownloadUrl(d.doc_id).then((r) => r.presigned_url)
              }
              filename={d.file_name}
              width={160}
              height={200}
              status={d.status === 'failed' ? 'failed' : undefined}
              message={d.error_message}
              onClick={d.status === 'ready' ? () => handleOpen(d) : undefined}
              onRemove={() => handleDelete(d.doc_id)}
              onRetry={d.status === 'failed' ? () => handleRetry(d.doc_id) : undefined}
            />
          ))}
        </div>
      )}

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
    </div>
  )
}

function Header({ count, loading, onRefresh }) {
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'baseline',
        justifyContent: 'space-between',
        marginBottom: 8,
      }}
    >
      <h1 style={{ margin: 0, fontSize: 22, fontWeight: 600, color: C.ink }}>
        Library
      </h1>
      <button
        type="button"
        onClick={onRefresh}
        title="Refresh"
        disabled={loading}
        style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: 6,
          padding: '5px 11px',
          background: C.bgSoft,
          border: `1px solid ${C.lineSoft}`,
          borderRadius: 8,
          color: C.inkSoft,
          fontSize: 12,
          cursor: loading ? 'not-allowed' : 'pointer',
          opacity: loading ? 0.6 : 1,
        }}
      >
        <RefreshCw size={12} />
        {count} {count === 1 ? 'document' : 'documents'}
      </button>
    </div>
  )
}

function CollectionFilter({ value, collections, onChange }) {
  if (collections.length === 0) return null
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 12 }}>
      <Pill active={!value} onClick={() => onChange('')}>All</Pill>
      {collections.map((c) => (
        <Pill key={c.name} active={value === c.name} onClick={() => onChange(c.name)}>
          {c.name}
        </Pill>
      ))}
    </div>
  )
}

function Pill({ children, active, onClick }) {
  return (
    <button
      type="button"
      onClick={onClick}
      style={{
        padding: '4px 10px',
        borderRadius: 999,
        background: active ? C.accentBg : C.bgSoft,
        border: `1px solid ${active ? C.accentBorder : C.lineSoft}`,
        color: active ? C.accent : C.inkSoft,
        fontSize: 11,
        fontWeight: active ? 600 : 500,
        cursor: 'pointer',
      }}
    >
      {children}
    </button>
  )
}

function EmptyState({ filtered }) {
  return (
    <div
      style={{
        marginTop: 60,
        textAlign: 'center',
        color: C.inkMuted,
        fontSize: 13,
      }}
    >
      {filtered ? 'No documents in this collection.' : 'No documents yet — upload via the chat composer.'}
    </div>
  )
}
