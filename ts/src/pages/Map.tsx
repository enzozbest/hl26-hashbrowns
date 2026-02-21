import { useLocation, Navigate } from 'react-router-dom'
import type { QueryResponse } from '../api/client'

interface MapState {
  query: string
  result: QueryResponse
}

export default function Map() {
  const { state } = useLocation() as { state: MapState | null }

  if (!state) {
    return <Navigate to="/chat" replace />
  }

  return (
    <div>
      <h2 className="text-2xl font-bold mb-4">Results</h2>
      <p className="text-gray-500 dark:text-gray-400 mb-4">Query: {state.query}</p>
      {/* TODO: implement map and results display */}
    </div>
  )
}
