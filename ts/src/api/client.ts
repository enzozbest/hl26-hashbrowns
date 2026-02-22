const API_BASE = 'http://localhost:8000'

export interface CouncilInfo {
  council_name: string
  council_id:   number
  lad_name:     string
  region:       string | null
  polygon:      { type: string; coordinates: unknown }
}

/** One council's analysis result from POST /api/analyse. */
export interface CouncilResult {
  council_id:          number
  score: number        // 0–100 percentage
  [key: string]:       unknown       // additional metrics from the backend
}

export interface AnalyseResponse {
    analysis_id: string
    scores: CouncilResult[]
}

export async function fetchCouncils(): Promise<CouncilInfo[]> {
  const res = await fetch(`${API_BASE}/api/councils`)
  if (!res.ok) throw new Error(`Failed to fetch councils: ${res.status}`)
  return res.json()
}

export async function submitAnalyse(
  council_ids: number[],
  prompt: string,
): Promise<AnalyseResponse> {
  const res = await fetch(`${API_BASE}/api/analyse`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ council_ids, prompt }),
  })
  if (!res.ok) throw new Error(`Analysis failed: ${res.status}`)
  return res.json()
}


