import axios from 'axios'
import { api } from './client.js'

export async function createUploadSession({ fileName, mimeType, sizeBytes, collection }) {
  const { data } = await api.post('/v1/ingest', {
    file_name: fileName,
    mime_type: mimeType,
    size_bytes: sizeBytes,
    collection,
  })
  return data
}

// Presigned URL embeds the original Content-Type; sending a different one
// breaks the SigV4 check, so we forward exactly what the session was created with.
export async function putBytesToPresignedUrl({ presignedUrl, file, contentType, onProgress }) {
  await axios.put(presignedUrl, file, {
    headers: { 'Content-Type': contentType },
    onUploadProgress: (evt) => {
      if (onProgress && evt.total) {
        onProgress(Math.round((evt.loaded * 100) / evt.total))
      }
    },
  })
}

export async function finalizeUpload(docId) {
  const { data } = await api.post(`/v1/documents/${docId}/finalize`)
  return data
}

export async function retryIngestion(docId) {
  const { data } = await api.post(`/v1/documents/${docId}/retry`)
  return data
}

export async function listDocuments({ collection, status, limit = 20, offset = 0 } = {}) {
  const params = { limit, offset }
  if (collection) params.collection = collection
  if (status) params.status = status
  const { data } = await api.get('/v1/documents', { params })
  return data
}

export async function getDocument(docId) {
  const { data } = await api.get(`/v1/documents/${docId}`)
  return data
}

export async function deleteDocument(docId) {
  await api.delete(`/v1/documents/${docId}`)
}

export async function getDocumentDownloadUrl(docId) {
  const { data } = await api.get(`/v1/documents/${docId}/download`)
  return data
}
