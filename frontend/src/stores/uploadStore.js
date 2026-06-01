import { create } from 'zustand'
import {
  createUploadSession,
  deleteDocument,
  finalizeUpload,
  putBytesToPresignedUrl,
  retryIngestion,
} from '../api/documents.js'
import { apiBaseUrl } from '../api/client.js'
import { openDocumentEventStream } from '../api/sse.js'

const TERMINAL_STATUSES = new Set(['ready', 'failed', 'duplicate'])

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
  phase: 'queued',
  progress: 0,
  message: '',
  chunksProcessed: 0,
  chunksTotal: 0,
  etaMs: null,
  duplicateOf: null,
  errorReason: null,
  durationMs: null,
})

export const useUploadStore = create((set, get) => ({
  jobs: [],
  _running: false,

  addFiles(files, collection, conversationId = null) {
    const additions = Array.from(files).map((f) =>
      newJob(f, collection, conversationId),
    )
    set((s) => ({ jobs: [...s.jobs, ...additions] }))
  },

  // Indexing is expensive — drain queue sequentially to avoid hammering the embedder.
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

  async removeJob(jobId) {
    const job = get().jobs.find((j) => j.id === jobId)
    set((s) => ({ jobs: s.jobs.filter((j) => j.id !== jobId) }))
    if (job?.docId && !TERMINAL_STATUSES.has(job.status)) {
      try { await deleteDocument(job.docId) } catch { /* sweeper will reap */ }
    }
  },

  async retryJob(jobId) {
    const job = get().jobs.find((j) => j.id === jobId)
    if (!job?.docId) {
      get()._update(jobId, {
        status: 'queued',
        phase: 'queued',
        progress: 0,
        message: '',
        errorReason: null,
      })
      get().start()
      return
    }
    get()._update(jobId, {
      status: 'processing',
      phase: 'queued',
      progress: 100,
      message: 'Retrying',
      chunksProcessed: 0,
      chunksTotal: 0,
      etaMs: null,
      errorReason: null,
    })
    try {
      await retryIngestion(job.docId)
      await waitForTerminalEvent(job.docId, (patch) =>
        get()._update(jobId, patch),
      )
    } catch (err) {
      const msg = err?.response?.data?.detail || err?.message || 'Retry failed'
      get()._update(jobId, {
        status: 'failed',
        phase: 'failed',
        message: msg,
        errorReason: 'RetryError',
      })
    }
  },

  clearCompleted() {
    set((s) => ({
      jobs: s.jobs.filter((j) => !TERMINAL_STATUSES.has(j.status)),
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
      get()._update(jobId, {
        status: 'uploading',
        phase: 'uploading',
        message: 'Preparing',
      })
      const session = await createUploadSession({
        fileName: job.filename,
        mimeType: job.mimeType,
        sizeBytes: job.size,
        collection: job.collection,
      })

      get()._update(jobId, { docId: session.doc_id, message: 'Uploading' })
      await putBytesToPresignedUrl({
        presignedUrl: session.presigned_url,
        file: job.file,
        contentType: job.mimeType,
        onProgress: (pct) => get()._update(jobId, { progress: pct }),
      })

      get()._update(jobId, {
        progress: 100,
        status: 'finalizing',
        phase: 'finalizing',
        message: 'Finalizing',
      })
      await finalizeUpload(session.doc_id)

      get()._update(jobId, {
        status: 'processing',
        phase: 'queued',
        message: 'Queued',
      })
      await waitForTerminalEvent(session.doc_id, (patch) =>
        get()._update(jobId, patch),
      )
    } catch (err) {
      const msg = err?.response?.data?.detail || err?.message || 'Upload failed'
      get()._update(jobId, {
        status: 'failed',
        phase: 'failed',
        message: msg,
        errorReason: 'UploadError',
      })
    }
  },
}))

// SSE consumer — translates backend phases to job state, computes ETA during
// embedding, and resolves on any terminal phase.
function waitForTerminalEvent(docId, applyPatch) {
  return new Promise((resolve) => {
    const startedAt = Date.now()
    let firstEmbedAt = null
    let firstEmbedDone = 0

    const close = openDocumentEventStream({
      baseUrl: apiBaseUrl,
      docId,
      onEvent: (event) => {
        const phase = event.phase
        if (!phase) return

        if (phase === 'embedding') {
          const processed = event.chunks_processed ?? 0
          const total = event.chunks_total ?? 0
          // skip first batch from rolling average — onnx warmup skews it
          if (firstEmbedAt === null) {
            firstEmbedAt = Date.now()
            firstEmbedDone = processed
          }
          let etaMs = null
          if (processed > firstEmbedDone && total > processed) {
            const elapsed = Date.now() - firstEmbedAt
            const msPerChunk = elapsed / (processed - firstEmbedDone)
            etaMs = Math.max(0, (total - processed) * msPerChunk)
          }
          applyPatch({
            status: 'processing',
            phase: 'embedding',
            chunksProcessed: processed,
            chunksTotal: total,
            etaMs,
            message: buildEmbeddingMessage(processed, total, etaMs),
          })
          return
        }

        if (phase === 'ready') {
          applyPatch({
            status: 'ready',
            phase: 'ready',
            etaMs: null,
            durationMs: Date.now() - startedAt,
            message:
              event.chunks_count != null
                ? `${event.chunks_count} chunks indexed`
                : 'Ready',
          })
          close()
          resolve()
          return
        }

        if (phase === 'failed') {
          applyPatch({
            status: 'failed',
            phase: 'failed',
            etaMs: null,
            errorReason: event.reason || 'IngestionError',
            message: event.message || 'Ingestion failed',
          })
          close()
          resolve()
          return
        }

        if (phase === 'duplicate') {
          applyPatch({
            status: 'duplicate',
            phase: 'duplicate',
            etaMs: null,
            duplicateOf: event.duplicate_of || null,
            message: 'Already in your library',
          })
          close()
          resolve()
          return
        }

        applyPatch({ phase, message: phaseLabel(phase) })
      },
      onError: () => {
        close()
        resolve()
      },
    })
  })
}

function phaseLabel(phase) {
  switch (phase) {
    case 'processing': return 'Starting'
    case 'downloading': return 'Reading'
    case 'hashed': return 'Verifying'
    case 'chunking': return 'Chunking'
    default: return phase
  }
}

function buildEmbeddingMessage(processed, total, etaMs) {
  const base = total > 0 ? `Embedding ${processed}/${total}` : 'Embedding'
  const eta = formatEta(etaMs)
  return eta ? `${base} · ${eta}` : base
}

export function formatEta(etaMs) {
  if (etaMs == null || etaMs < 3000) return ''
  if (etaMs < 30000) return `~${Math.ceil(etaMs / 1000)}s left`
  if (etaMs < 120000) return `~${Math.round(etaMs / 10000) * 10}s left`
  return `~${Math.ceil(etaMs / 60000)} min left`
}

export function embeddingPercent(processed, total) {
  if (!total || total <= 0) return null
  return Math.min(100, Math.max(0, Math.round((processed / total) * 100)))
}
