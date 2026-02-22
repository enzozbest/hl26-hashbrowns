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
  council_name?:       string
  indicators?:         Indicator[]
  [key: string]:       unknown       // additional metrics from the backend
}

export interface Indicator {
  name:         string
  value:        number
  contribution: number
  direction:    string
}

export interface AnalyseResponse {
    analysis_id: string
    scores: CouncilResult[]
}

/** Real IBex statistics for a single council. */
export interface CouncilStats {
  council_id:                   number
  approval_rate:                number
  refusal_rate:                 number
  activity_level:               string
  average_decision_time:        Record<string, number | null>
  number_of_applications:       Record<string, number | null>
  number_of_new_homes_approved: number | null
}

export interface CouncilStatsResponse {
  council_stats: Record<string, CouncilStats>
  errors:        string[]
}

export async function fetchCouncils(): Promise<CouncilInfo[]> {
  const res = await fetch(`${API_BASE}/api/councils`)
  if (!res.ok) throw new Error(`Failed to fetch councils: ${res.status}`)
  return res.json()
}

export async function submitAnalyse(
    council_ids: number[],
    prompt: string
): Promise<AnalyseResponse> {
  const res = await fetch(`${API_BASE}/api/analyse`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ council_ids, prompt }),
  })
  if (!res.ok) throw new Error(`Analysis failed: ${res.status}`)
  return res.json()
}

/** Fetch real IBex planning statistics for a list of council IDs. */
export async function fetchCouncilStats(
    council_ids: number[],
): Promise<CouncilStatsResponse> {
  const res = await fetch(`${API_BASE}/api/council-stats`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ council_ids }),
  })
  if (!res.ok) throw new Error(`Council stats fetch failed: ${res.status}`)
  return res.json()
}


