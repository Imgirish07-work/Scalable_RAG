// Browser EventSource wrapper around GET /v1/documents/{id}/events.
// The backend emits named events: `event: phase\ndata: {json}\n\n`.

export function openDocumentEventStream({ baseUrl, docId, onEvent, onError }) {
  const url = `${baseUrl}/v1/documents/${docId}/events`
  const es = new EventSource(url)

  es.addEventListener('phase', (e) => {
    try {
      onEvent?.(JSON.parse(e.data))
    } catch (err) {
      onError?.(err)
    }
  })

  // EventSource auto-reconnects on transient errors; only escalate when the
  // connection is permanently closed (the stream ended after a terminal phase).
  es.onerror = (err) => {
    if (es.readyState === EventSource.CLOSED) {
      onError?.(err)
    }
  }

  return () => es.close()
}
