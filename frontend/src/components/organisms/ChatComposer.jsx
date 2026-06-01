import { useEffect, useMemo, useRef } from 'react'
import DocPreviewCard from '../molecules/DocPreviewCard'
import CollectionPill from '../molecules/CollectionPill'
import FileAttachButton from '../atoms/FileAttachButton'
import SendButton from '../atoms/SendButton'
import { useToast } from '../Toast'
import { useUploadStore } from '../../stores/uploadStore'
import { MAX_UPLOAD_SIZE_BYTES, MAX_UPLOAD_SIZE_MB } from '../../config'
import { C } from '../../theme'

const STATUS_LABEL = {
  queued: 'Queued',
  uploading: 'Uploading',
  finalizing: 'Finalizing',
  processing: 'Processing',
}

export default function ChatComposer({
  conversationId,
  collection,
  onCollectionChange,
  collections = [],
  value,
  onChange,
  onSubmit,
  isLoading,
  onPreviewJob,
  placeholder = 'Ask anything about your documents',
}) {
  const toast = useToast()
  const textareaRef = useRef(null)

  const allJobs = useUploadStore((s) => s.jobs)
  const addFiles = useUploadStore((s) => s.addFiles)
  const start = useUploadStore((s) => s.start)
  const removeJob = useUploadStore((s) => s.removeJob)
  const retryJob = useUploadStore((s) => s.retryJob)

  const jobs = useMemo(
    () => allJobs.filter((j) => j.conversationId === conversationId),
    [allJobs, conversationId],
  )
  const anyBlocking = jobs.some(
    (j) => !['ready', 'failed', 'duplicate'].includes(j.status),
  )
  const canSubmit = value.trim().length > 0 && !anyBlocking && !isLoading

  const toastedTerminals = useRef(new Set())
  useEffect(() => {
    jobs.forEach((j) => {
      if (!['ready', 'failed', 'duplicate'].includes(j.status)) return
      if (toastedTerminals.current.has(j.id)) return
      toastedTerminals.current.add(j.id)
      if (j.status === 'ready') {
        toast.success('Document uploaded', j.filename)
      } else if (j.status === 'duplicate') {
        toast.info('Already in your library', j.filename)
      } else {
        toast.error(
          'Upload failed',
          `${j.filename} — ${j.message || 'Unknown error'}`,
        )
      }
    })
  }, [jobs, toast])

  const handleAttach = (newFiles) => {
    const arr = Array.isArray(newFiles) ? newFiles : [newFiles]
    if (arr.length === 0) return
    addFiles(arr, collection || 'default', conversationId)
    start()
  }

  const autoGrow = (el) => {
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`
  }

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      if (canSubmit) onSubmit()
    }
  }

  return (
    <div
      style={{
        padding: '0 16px 14px',
        maxWidth: 820,
        width: '100%',
        margin: '0 auto',
      }}
    >
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          borderRadius: 16,
          background: C.bgInput,
          border: `1.5px solid ${C.lineCard}`,
          overflow: 'hidden',
        }}
      >
        {jobs.length > 0 && (
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              gap: 10,
              padding: '12px 14px 10px',
              borderBottom: `1px solid ${C.lineSoft}`,
            }}
          >
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
              {jobs.map((j) => (
                <DocPreviewCard
                  key={j.id}
                  file={j.file}
                  filename={j.filename}
                  width={84}
                  height={104}
                  status={j.status}
                  phase={j.phase}
                  progress={j.progress}
                  chunksProcessed={j.chunksProcessed}
                  chunksTotal={j.chunksTotal}
                  message={j.message}
                  onClick={
                    j.status === 'ready' && onPreviewJob
                      ? () => onPreviewJob(j)
                      : undefined
                  }
                  onRemove={() => removeJob(j.id)}
                  onRetry={j.status === 'failed' ? () => retryJob(j.id) : undefined}
                />
              ))}
            </div>

            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                gap: 12,
                minHeight: 22,
              }}
            >
              <ProgressStrip jobs={jobs} />
              <CollectionPill
                value={collection || 'default'}
                suggestions={collections}
                onChange={onCollectionChange}
              />
            </div>
          </div>
        )}

        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => {
            onChange(e.target.value)
            autoGrow(e.target)
          }}
          onKeyDown={handleKey}
          placeholder={placeholder}
          rows={1}
          style={{
            background: 'transparent',
            border: 'none',
            outline: 'none',
            color: C.ink,
            fontSize: 14,
            resize: 'none',
            padding: '14px 14px 6px',
            fontFamily: 'inherit',
            lineHeight: 1.5,
            maxHeight: 160,
            overflowY: 'auto',
          }}
        />

        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 4,
            padding: '4px 8px 8px',
          }}
        >
          <FileAttachButton
            onSelect={handleAttach}
            onError={(msg) => toast.error('File rejected', msg)}
            accept=".pdf,.txt,.md,.docx,.pptx,.csv,.json"
            maxBytes={MAX_UPLOAD_SIZE_BYTES}
            active={jobs.length > 0}
            title={`Attach a file (max ${MAX_UPLOAD_SIZE_MB} MB)`}
          />
          <div style={{ flex: 1 }} />
          <SendButton onClick={onSubmit} disabled={!canSubmit} />
        </div>
      </div>
    </div>
  )
}

function ProgressStrip({ jobs }) {
  const inFlight = jobs.filter(
    (j) => !['ready', 'failed', 'duplicate'].includes(j.status),
  )

  if (inFlight.length === 0) {
    const ready = jobs.filter((j) => j.status === 'ready').length
    if (ready === 0) return <span />
    return <StripLine dot={C.ok} text={`${ready} ready to query`} />
  }

  const active = inFlight.find((j) => j.status === 'uploading') || inFlight[0]
  const detail = activeDetail(active)
  const more = inFlight.length > 1 ? ` · +${inFlight.length - 1} more` : ''

  return (
    <StripLine
      dot={C.accent}
      pulsing
      text={`${active.filename} · ${detail}${more}`}
    />
  )
}

function activeDetail(job) {
  if (job.status === 'uploading' && job.progress != null) {
    return `Uploading ${job.progress}%`
  }
  if (job.message) return job.message
  return STATUS_LABEL[job.status] || job.status
}

function StripLine({ dot, pulsing, text }) {
  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 6,
        fontSize: 11,
        color: C.inkSoft,
        minWidth: 0,
        flex: 1,
      }}
    >
      <span
        className={pulsing ? 'animate-pulse' : ''}
        style={{
          width: 6,
          height: 6,
          borderRadius: '50%',
          background: dot,
          flexShrink: 0,
        }}
      />
      <span
        style={{
          whiteSpace: 'nowrap',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
        }}
        title={text}
      >
        {text}
      </span>
    </span>
  )
}
