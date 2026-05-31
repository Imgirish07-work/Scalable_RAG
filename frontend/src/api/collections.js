import { api } from './client.js'

export async function listCollections() {
  const { data } = await api.get('/v1/collections')
  return data
}
