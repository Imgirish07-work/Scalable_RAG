import { useEffect, useMemo, useRef } from 'react'
import { DocPreviewCard } from '../molecules'
import QueryBox from '../molecules/QueryBox'
import CollectionPill from '../molecules/CollectionPill'
import { useToast } from '../Toast'
import { useUploadStore } from '../../stores/uploadStore'
import { MAX_UPLOAD_SIZE_BYTES, MAX_UPLOAD_SIZE_MB } from '../../config'
import { C } from '../../theme'

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
    (j) => j.status !== 'ready' && j.status !== 'failed',
  )
  const canSubmit = value.trim().length > 0 && !anyBlocking

  const toastedFailures = useRef(new Set())
  useEffect(() => {
    jobs.forEach((j) => {
      if (j.status === 'failed' && !toastedFailures.current.has(j.id)) {
        toastedFailures.current.add(j.id)
        toast.error('Upload failed', `${j.filename} — ${j.message || 'Unknown error'}`)
      }
    })
  }, [jobs, toast])

  const handleAttachmentsChange = (files) => {
    if (!files || files.length === 0) return
    const arr = Array.from(files)
    const oversized = arr.filter((f) => f.size > MAX_UPLOAD_SIZE_BYTES)
    const allowed = arr.filter((f) => f.size <= MAX_UPLOAD_SIZE_BYTES)
    oversized.forEach((f) =>
      toast.error('File too large', `${f.name} exceeds ${MAX_UPLOAD_SIZE_MB} MB limit`),
    )
    if (allowed.length === 0) return
    addFiles(allowed, collection || 'default', conversationId)
    start()
  }

  const handleRetry = (jobId) => {
    toastedFailures.current.delete(jobId)
    retryJob(jobId)
    start()
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
      {jobs.length > 0 && (
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: 10,
            padding: '12px 12px 10px',
            background: C.bgPanel,
            border: `1px solid ${C.lineSoft}`,
            borderBottom: 'none',
            borderTopLeftRadius: 14,
            borderTopRightRadius: 14,
          }}
        >
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
            {jobs.map((j) => (
              <DocPreviewCard
                key={j.id}
                file={j.file}
                filename={j.filename}
                width={96}
                height={120}
                status={j.status === 'ready' ? undefined : j.status}
                progress={j.progress}
                message={j.status === 'uploading' ? j.message : undefined}
                onClick={j.status === 'ready' && onPreviewJob ? () => onPreviewJob(j) : undefined}
                onRemove={() => removeJob(j.id)}
                onRetry={j.status === 'failed' ? () => handleRetry(j.id) : undefined}
              />
            ))}
          </div>
          <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
            <CollectionPill
              value={collection || 'default'}
              suggestions={collections}
              onChange={onCollectionChange}
            />
          </div>
        </div>
      )}
      <QueryBox
        className={jobs.length > 0 ? '!mt-0 !rounded-t-none' : ''}
        value={value}
        onChange={onChange}
        onSubmit={onSubmit}
        isLoading={isLoading}
        disabled={false}
        placeholder={placeholder}
        canSubmit={canSubmit}
        attachments={[]}
        onAttachmentsChange={handleAttachmentsChange}
        attachAccept=".pdf,.txt,.md,.docx,.pptx,.csv,.json"
        attachMaxBytes={MAX_UPLOAD_SIZE_BYTES}
      />
    </div>
  )
}
