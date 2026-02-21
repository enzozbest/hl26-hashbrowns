import { useLocation, Navigate } from 'react-router-dom'
import { MapContainer, GeoJSON, useMap } from 'react-leaflet'
import { useEffect, useState, useCallback, useRef } from 'react'
import L from 'leaflet'
import type { QueryResponse } from '../api/client'
import type { Feature, FeatureCollection, Geometry } from 'geojson'
import type { Layer, PathOptions } from 'leaflet'

const GEOJSON_URL =
  'https://raw.githubusercontent.com/martinjc/UK-GeoJSON/master/json/administrative/eng/lad.json'

// -- Colour config --
const BG_COLOR = '#0a1628'
const BORDER_COLOR = '#1e3a5f'
const NO_DATA_FILL = '#112240'
const HIGHLIGHT_LOW = '#1b4965'
const HIGHLIGHT_HIGH = '#5ec4e6'
const SELECTED_FILL = '#f59e0b'
const SELECTED_BORDER = '#fbbf24'

const DEFAULT_CENTER: L.LatLngTuple = [52.5, -1.5]
const DEFAULT_ZOOM = 6
const MIN_ZOOM = 5
const MAX_ZOOM = 12
const MAX_BOUNDS: L.LatLngBoundsExpression = [
  [49.5, -7],   // south-west
  [56, 2.5],    // north-east
]

interface MapState {
  query: string
  result: QueryResponse
}

interface SelectedBorough {
  name: string
  pct: number | undefined
}

function getColor(pct: number): string {
  const low = { r: 0x1b, g: 0x49, b: 0x65 }
  const high = { r: 0x5e, g: 0xc4, b: 0xe6 }
  const t = pct / 100
  const r = Math.round(low.r + (high.r - low.r) * t)
  const g = Math.round(low.g + (high.g - low.g) * t)
  const b = Math.round(low.b + (high.b - low.b) * t)
  return `rgb(${r},${g},${b})`
}

function normalise(name: string): string {
  return name.toLowerCase().replace(/[^a-z]/g, '')
}

// -- Borough detail panel --
function BoroughDetail({ name, pct }: { name: string; pct: number | undefined }) {
  return (
    <div
      className="absolute top-4 right-4 z-[1000] rounded-xl p-5 shadow-lg backdrop-blur-sm"
      style={{ background: 'rgba(10, 22, 40, 0.92)', border: `1px solid ${SELECTED_BORDER}`, minWidth: 220 }}
    >
      <h3 className="text-lg font-bold text-white mb-3">{name}</h3>
      {pct !== undefined ? (
        <div>
          <div className="text-sm text-gray-400 mb-1">Planning permission likelihood</div>
          <div className="text-3xl font-bold" style={{ color: HIGHLIGHT_HIGH }}>{pct}%</div>
          <div className="mt-3 h-2 rounded-full overflow-hidden" style={{ background: HIGHLIGHT_LOW }}>
            <div
              className="h-full rounded-full transition-all duration-500"
              style={{ width: `${pct}%`, background: HIGHLIGHT_HIGH }}
            />
          </div>
          <div className="mt-3 text-xs text-gray-500">
            {pct >= 70 ? 'High chance of approval' : pct >= 40 ? 'Moderate chance of approval' : 'Low chance of approval'}
          </div>
        </div>
      ) : (
        <div className="text-sm text-gray-400">No data available for this borough.</div>
      )}
      <div className="mt-3 text-xs text-gray-600">Click outside to return</div>
    </div>
  )
}

// -- Map interaction controller --
function MapController({
  selected,
  selectedLayer,
  onReset,
}: {
  selected: SelectedBorough | null
  selectedLayer: L.Layer | null
  onReset: () => void
}) {
  const map = useMap()

  useEffect(() => {
    if (selected && selectedLayer) {
      const bounds = (selectedLayer as L.GeoJSON).getBounds()
      map.flyToBounds(bounds, { padding: [80, 80], duration: 0.6 })
    } else if (!selected) {
      map.flyTo(DEFAULT_CENTER, DEFAULT_ZOOM, { duration: 0.6 })
    }
  }, [selected, selectedLayer, map])

  // Click on the map background resets selection
  useEffect(() => {
    function handleClick(e: L.LeafletMouseEvent) {
      // Only reset if the click is directly on the map (not on a polygon)
      if ((e.originalEvent.target as HTMLElement)?.classList?.contains('leaflet-container')) {
        onReset()
      }
    }
    map.on('click', handleClick)
    return () => { map.off('click', handleClick) }
  }, [map, onReset])

  return null
}

export default function MapPage() {
  const { state } = useLocation() as { state: MapState | null }
  const [geoData, setGeoData] = useState<FeatureCollection | null>(null)
  const [loading, setLoading] = useState(true)
  const [selected, setSelected] = useState<SelectedBorough | null>(null)
  const [selectedLayer, setSelectedLayer] = useState<L.Layer | null>(null)
  const selectedLayerRef = useRef<L.Path | null>(null)
  const geoJsonRef = useRef<L.GeoJSON | null>(null)

  useEffect(() => {
    fetch(GEOJSON_URL)
      .then((res) => res.json())
      .then((data: FeatureCollection) => {
        setGeoData(data)
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }, [])

  // Apply orange highlight AFTER re-render so it isn't overridden by react-leaflet
  useEffect(() => {
    if (selectedLayerRef.current) {
      selectedLayerRef.current.setStyle({
        fillColor: SELECTED_FILL,
        weight: 2,
        color: SELECTED_BORDER,
        opacity: 1,
        fillOpacity: 0.9,
      })
      selectedLayerRef.current.bringToFront()
    }
  }, [selected])

  const handleReset = useCallback(() => {
    if (selectedLayerRef.current && geoJsonRef.current) {
      geoJsonRef.current.resetStyle(selectedLayerRef.current)
    }
    selectedLayerRef.current = null
    setSelected(null)
    setSelectedLayer(null)
  }, [])

  if (!state) {
    return <Navigate to="/" replace />
  }

  const { boroughs } = state.result

  const boroughLookup = new Map<string, number>()
  for (const [name, pct] of Object.entries(boroughs)) {
    boroughLookup.set(normalise(name), pct)
  }

  function getPercentage(featureName: string): number | undefined {
    return boroughLookup.get(normalise(featureName))
  }

  function featureStyle(feature: Feature<Geometry> | undefined): PathOptions {
    const name = feature?.properties?.LAD13NM ?? feature?.properties?.name ?? ''
    const pct = getPercentage(name)
    return {
      fillColor: pct !== undefined ? getColor(pct) : NO_DATA_FILL,
      weight: 0.5,
      opacity: 0.6,
      color: BORDER_COLOR,
      fillOpacity: pct !== undefined ? 0.85 : 0.4,
    }
  }

  function onEachFeature(feature: Feature<Geometry>, layer: Layer) {
    const name = feature.properties?.LAD13NM ?? feature.properties?.name ?? 'Unknown'
    const pct = getPercentage(name)
    const tip =
      pct !== undefined
        ? `<strong>${name}</strong><br/>${pct}% planning permission`
        : `<strong>${name}</strong><br/><em>No data</em>`
    layer.bindTooltip(tip, { sticky: true })

    layer.on('click', (e) => {
      L.DomEvent.stopPropagation(e as L.LeafletEvent & { originalEvent: Event })
      selectedLayerRef.current = layer as L.Path
      setSelected({ name, pct })
      setSelectedLayer(layer)
    })
  }

  return (
    <div className="relative h-screen w-screen" style={{ background: BG_COLOR }}>
      {loading ? (
        <div className="h-full flex items-center justify-center text-gray-500">
          Loading map data...
        </div>
      ) : !geoData ? (
        <div className="h-full flex items-center justify-center text-red-500">
          Failed to load map data.
        </div>
      ) : (
        <>
          <MapContainer
            center={DEFAULT_CENTER}
            zoom={DEFAULT_ZOOM}
            minZoom={MIN_ZOOM}
            maxZoom={MAX_ZOOM}
            maxBounds={MAX_BOUNDS}
            maxBoundsViscosity={0.9}
            className="h-full w-full"
            style={{ background: BG_COLOR }}
            zoomControl={false}
            attributionControl={false}
          >
            <MapController
              selected={selected}
              selectedLayer={selectedLayer}
              onReset={handleReset}
            />
            <GeoJSON
              ref={geoJsonRef}
              data={geoData}
              style={featureStyle}
              onEachFeature={onEachFeature}
            />
          </MapContainer>

          {/* Legend */}
          <div className="absolute bottom-4 left-4 z-[1000] flex items-center gap-2 text-xs text-gray-400">
            <span>0%</span>
            <div
              className="h-2 w-24 rounded"
              style={{ background: `linear-gradient(to right, ${HIGHLIGHT_LOW}, ${HIGHLIGHT_HIGH})` }}
            />
            <span>100%</span>
          </div>

          {selected && (
            <BoroughDetail name={selected.name} pct={selected.pct} />
          )}
        </>
      )}
    </div>
  )
}
