import { api } from './client.js'

export async function postQuery({
  query,
  collection,
  topK,
  temperature,
  includeSources = true,
}) {
  const body = { query, include_sources: includeSources }
  if (collection) body.collection = collection
  if (topK != null) body.top_k = topK
  if (temperature != null) body.temperature = temperature

  const { data } = await api.post('/v1/query', body)
  return data
}
