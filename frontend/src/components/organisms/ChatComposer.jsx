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
  'creating-session': 'Preparing',
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

  const jobs = useMemo(
    () => allJobs.filter((j) => j.conversationId === conversationId),
    [allJobs, conversationId],
  )
  const anyBlocking = jobs.some(
    (j) => j.status !== 'ready' && j.status !== 'failed',
  )
  const canSubmit = value.trim().length > 0 && !anyBlocking && !isLoading

  const toastedFailures = useRef(new Set())
  useEffect(() => {
    jobs.forEach((j) => {
      if (j.status === 'failed' && !toastedFailures.current.has(j.id)) {
        toastedFailures.current.add(j.id)
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
                  onClick={
                    j.status === 'ready' && onPreviewJob
                      ? () => onPreviewJob(j)
                      : undefined
                  }
                  onRemove={() => removeJob(j.id)}
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
    (j) => j.status !== 'ready' && j.status !== 'failed',
  )

  if (inFlight.length === 0) {
    const ready = jobs.filter((j) => j.status === 'ready').length
    if (ready === 0) return <span />
    return <StripLine dot={C.ok} text={`${ready} ready to query`} />
  }

  const active = inFlight.find((j) => j.status === 'uploading') || inFlight[0]
  const label = STATUS_LABEL[active.status] || active.status
  const pct =
    active.status === 'uploading' && active.progress != null
      ? ` ${active.progress}%`
      : ''
  const more = inFlight.length > 1 ? ` · +${inFlight.length - 1} more` : ''

  return (
    <StripLine
      dot={C.accent}
      pulsing
      text={`${label} ${active.filename}${pct}${more}`}
    />
  )
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
