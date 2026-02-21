const BASE_URL = 'http://localhost:8000'

export interface Location {
  name: string
  region: string
  planningPermissionChance: number
}

export interface QueryResponse {
  locations: Location[]
  summary: string
}

export async function submitQuery(message: string): Promise<QueryResponse> {
  const res = await fetch(`${BASE_URL}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message }),
  })

  if (!res.ok) {
    throw new Error(`API error: ${res.status}`)
  }

  return res.json()
}
