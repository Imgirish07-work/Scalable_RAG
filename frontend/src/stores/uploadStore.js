import { create } from 'zustand'
import {
  createUploadSession,
  finalizeUpload,
  putBytesToPresignedUrl,
} from '../api/documents.js'
import { apiBaseUrl } from '../api/client.js'
import { openDocumentEventStream } from '../api/sse.js'

// Job lifecycle: queued → uploading → processing → ready|failed
const newJob = (file, collection, conversationId) => ({
  id: crypto.randomUUID(),
  file,
  filename: file.name,
  size: file.size,
  mimeType: file.type || 'application/octet-stream',
  collection,
  conversationId: conversationId || null,
  docId: null,
  status: 'queued',
  progress: 0,
  message: '',
})

export const useUploadStore = create((set, get) => ({
  jobs: [],
  _running: false,

  addFiles(files, collection, conversationId = null) {
    const additions = Array.from(files).map((f) => newJob(f, collection, conversationId))
    set((s) => ({ jobs: [...s.jobs, ...additions] }))
  },

  // Drain the queue sequentially. Indexing is expensive — parallel uploads
  // would hammer the backend embedder. Guarded so rapid double-clicks are no-ops.
  async start() {
    if (get()._running) return
    set({ _running: true })
    try {
      while (true) {
        const next = get().jobs.find((j) => j.status === 'queued')
        if (!next) break
        await get()._processJob(next.id)
      }
    } finally {
      set({ _running: false })
    }
  },

  removeJob(jobId) {
    set((s) => ({ jobs: s.jobs.filter((j) => j.id !== jobId) }))
  },

  retryJob(jobId) {
    get()._update(jobId, { status: 'queued', progress: 0, message: '' })
  },

  clearCompleted() {
    set((s) => ({
      jobs: s.jobs.filter((j) => !['ready', 'failed'].includes(j.status)),
    }))
  },

  _update(jobId, patch) {
    set((s) => ({
      jobs: s.jobs.map((j) => (j.id === jobId ? { ...j, ...patch } : j)),
    }))
  },

  async _processJob(jobId) {
    const job = get().jobs.find((j) => j.id === jobId)
    if (!job) return

    try {
      get()._update(jobId, { status: 'uploading', message: 'Creating session…' })
      const session = await createUploadSession({
        fileName: job.filename,
        mimeType: job.mimeType,
        sizeBytes: job.size,
        collection: job.collection,
      })

      get()._update(jobId, { docId: session.doc_id, message: 'Uploading…' })
      await putBytesToPresignedUrl({
        presignedUrl: session.presigned_url,
        file: job.file,
        contentType: job.mimeType,
        onProgress: (pct) => get()._update(jobId, { progress: pct }),
      })

      get()._update(jobId, { progress: 100, message: 'Finalizing…' })
      await finalizeUpload(session.doc_id)

      get()._update(jobId, { status: 'processing', message: 'Indexing…' })
      await waitForTerminalEvent(session.doc_id, (patch) =>
        get()._update(jobId, patch),
      )
    } catch (err) {
      const msg = err?.response?.data?.detail || err?.message || 'Upload failed'
      get()._update(jobId, { status: 'failed', message: msg })
    }
  },
}))

// Resolves once the SSE stream reports a terminal phase (ready or failed).
function waitForTerminalEvent(docId, applyPatch) {
  return new Promise((resolve) => {
    const close = openDocumentEventStream({
      baseUrl: apiBaseUrl,
      docId,
      onEvent: (event) => {
        if (event.phase === 'ready') {
          applyPatch({
            status: 'ready',
            message:
              event.chunks_count != null
                ? `${event.chunks_count} chunks indexed`
                : 'Ready',
          })
          close()
          resolve()
        } else if (event.phase === 'failed') {
          applyPatch({
            status: 'failed',
            message: event.message || 'Ingestion failed',
          })
          close()
          resolve()
        } else if (event.phase) {
          applyPatch({ message: event.phase })
        }
      },
      onError: () => {
        close()
        resolve()
      },
    })
  })
}
