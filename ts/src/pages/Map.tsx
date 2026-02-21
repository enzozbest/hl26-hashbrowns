import { useLocation, Navigate } from 'react-router-dom'
import { MapContainer, GeoJSON, useMap } from 'react-leaflet'
import { useEffect, useState, useCallback, useRef } from 'react'
import L from 'leaflet'
import type { QueryResponse } from '../api/client'
import type { Feature, FeatureCollection, Geometry } from 'geojson'
import type { Layer, PathOptions } from 'leaflet'
import { REGIONS } from '../data/regions'

// ── Constants ────────────────────────────────────────────────────────────────
const GEOJSON_URL =
  'https://raw.githubusercontent.com/martinjc/UK-GeoJSON/master/json/administrative/eng/lad.json'

const LEADERBOARD_W = 292 // px

const DEFAULT_CENTER: L.LatLngTuple = [52.5, -1.5]
const DEFAULT_ZOOM = 6
const MIN_ZOOM = 5
const MAX_ZOOM = 12
const MAX_BOUNDS: L.LatLngBoundsExpression = [[49.5, -7], [56, 2.5]]

const REGION_BOUNDS: Record<string, L.LatLngBoundsExpression> = {
  'London':             [[51.28, -0.51], [51.69,  0.33]],
  'South East':         [[50.60, -1.80], [51.80,  1.50]],
  'South West':         [[49.85, -5.75], [51.75, -1.65]],
  'East of England':    [[51.45, -0.50], [53.00,  1.80]],
  'East Midlands':      [[52.00, -2.20], [53.50,  0.80]],
  'West Midlands':      [[52.00, -2.90], [53.00, -1.40]],
  'Yorkshire & Humber': [[53.30, -2.55], [54.60, -0.20]],
  'North West':         [[53.30, -3.40], [54.65, -1.80]],
  'North East':         [[54.40, -2.45], [55.80, -1.00]],
}

// ── Earthy palette ───────────────────────────────────────────────────────────
const MAP_BG        = '#1e1b17'
const NO_DATA_FILL  = '#2e2820'
const BORDER_COLOR  = 'rgba(200, 150, 42, 0.25)'
const GRAD_LOW      = { r: 0x40, g: 0x28, b: 0x0e }  // dark brown  (0% approval)
const GRAD_HIGH     = { r: 0xc8, g: 0x96, b: 0x2a }  // ochre gold  (100% approval)
const SEL_FILL      = '#e8c870'                        // lighter gold — distinguishes selection
const SEL_BORDER    = '#f0d888'

const C = {
  card:        'rgba(14, 10, 6, 0.94)',
  text:        '#ede8df',
  textMuted:   'rgba(237, 232, 223, 0.65)',
  textFaint:   'rgba(237, 232, 223, 0.35)',
  accent:      '#c8962a',
  accentDim:   'rgba(200, 150, 42, 0.45)',
  accentFaint: 'rgba(200, 150, 42, 0.18)',
  border:      'rgba(200, 150, 42, 0.18)',
  rowHover:    'rgba(200, 150, 42, 0.06)',
  rowActive:   'rgba(200, 150, 42, 0.13)',
}

// ── Helpers ──────────────────────────────────────────────────────────────────
function getColor(pct: number): string {
  const t = Math.max(0, Math.min(1, pct / 100))
  const r = Math.round(GRAD_LOW.r + (GRAD_HIGH.r - GRAD_LOW.r) * t)
  const g = Math.round(GRAD_LOW.g + (GRAD_HIGH.g - GRAD_LOW.g) * t)
  const b = Math.round(GRAD_LOW.b + (GRAD_HIGH.b - GRAD_LOW.b) * t)
  return `rgb(${r},${g},${b})`
}

function normalise(name: string): string {
  return name.toLowerCase().replace(/[^a-z]/g, '')
}

function riskMeta(pct: number) {
  if (pct >= 70) return { label: 'High approval likelihood',     colour: '#6b8f5e' }
  if (pct >= 40) return { label: 'Moderate approval likelihood', colour: '#c8962a' }
  return               { label: 'Low approval likelihood',       colour: '#b85c38' }
}

// ── Types ────────────────────────────────────────────────────────────────────
interface MapState       { query: string; result: QueryResponse; region?: string }
interface SelectedBorough { name: string; pct: number | undefined }

// ── BoroughDetail panel ──────────────────────────────────────────────────────
function BoroughDetail({ name, pct, onClose }: {
  name: string
  pct:  number | undefined
  onClose: () => void
}) {
  const meta = pct !== undefined ? riskMeta(pct) : null

  return (
    <div style={{
      position:       'absolute',
      top:            '64px',
      right:          '16px',
      zIndex:         1000,
      background:     C.card,
      border:         `1px solid ${C.border}`,
      backdropFilter: 'blur(12px)',
      padding:        '20px 22px',
      minWidth:       '240px',
      maxWidth:       '280px',
    }}>
      {/* Corner brackets */}
      {([
        'top:0;left:0;border-top:1px;border-left:1px',
        'top:0;right:0;border-top:1px;border-right:1px',
        'bottom:0;left:0;border-bottom:1px;border-left:1px',
        'bottom:0;right:0;border-bottom:1px;border-right:1px',
      ] as const).map((_, i) => {
        const pos = [
          { top: 0, left: 0 }, { top: 0, right: 0 },
          { bottom: 0, left: 0 }, { bottom: 0, right: 0 },
        ][i]
        const borders = [
          { borderTop: `1px solid ${C.accentDim}`, borderLeft: `1px solid ${C.accentDim}` },
          { borderTop: `1px solid ${C.accentDim}`, borderRight: `1px solid ${C.accentDim}` },
          { borderBottom: `1px solid ${C.accentDim}`, borderLeft: `1px solid ${C.accentDim}` },
          { borderBottom: `1px solid ${C.accentDim}`, borderRight: `1px solid ${C.accentDim}` },
        ][i]
        return <span key={i} style={{ position: 'absolute', width: 12, height: 12, ...pos, ...borders }} />
      })}

      <p style={{ fontFamily: 'Inter, sans-serif', fontSize: 10, fontWeight: 500, letterSpacing: '0.22em', textTransform: 'uppercase', color: C.textFaint, marginBottom: 6 }}>
        Planning Analysis
      </p>
      <h3 style={{ fontFamily: '"Barlow Condensed", sans-serif', fontWeight: 600, fontSize: '1.5rem', textTransform: 'uppercase', letterSpacing: '-0.01em', color: C.text, lineHeight: 1.05, marginBottom: 16 }}>
        {name}
      </h3>

      {pct !== undefined && meta ? (
        <>
          <p style={{ fontFamily: 'Inter, sans-serif', fontSize: 10, fontWeight: 500, letterSpacing: '0.2em', textTransform: 'uppercase', color: C.textFaint, marginBottom: 8 }}>
            Approval Likelihood
          </p>
          <div style={{ fontFamily: '"Barlow Condensed", sans-serif', fontWeight: 700, fontSize: '2.6rem', color: C.accent, lineHeight: 1, marginBottom: 10 }}>
            {pct}%
          </div>

          {/* Progress bar */}
          <div style={{ height: 3, background: C.accentFaint, marginBottom: 10 }}>
            <div style={{ height: '100%', width: `${pct}%`, background: meta.colour, transition: 'width 0.5s ease' }} />
          </div>

          {/* Risk indicator */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 16 }}>
            <span style={{ width: 6, height: 6, borderRadius: '50%', background: meta.colour, flexShrink: 0 }} />
            <span style={{ fontFamily: 'Inter, sans-serif', fontSize: '0.78rem', color: meta.colour, fontWeight: 500 }}>
              {meta.label}
            </span>
          </div>
        </>
      ) : (
        <p style={{ fontFamily: 'Inter, sans-serif', fontSize: '0.85rem', color: C.textMuted, marginBottom: 16 }}>
          No data available for this council.
        </p>
      )}

      <button onClick={onClose} style={{
        fontFamily: 'Inter, sans-serif', fontSize: 10, letterSpacing: '0.14em',
        textTransform: 'uppercase', color: C.textFaint, background: 'none',
        border: 'none', cursor: 'pointer', padding: 0,
      }}>
        ← Back to overview
      </button>
    </div>
  )
}

// ── MapController ────────────────────────────────────────────────────────────
function MapController({ selected, selectedLayer, onReset, regionBounds, zoomToRegionTrigger }: {
  selected:              SelectedBorough | null
  selectedLayer:         L.Layer | null
  onReset:               () => void
  regionBounds?:         L.LatLngBoundsExpression
  zoomToRegionTrigger:   number
}) {
  const map = useMap()

  // Zoom to region on initial mount
  useEffect(() => {
    if (regionBounds) {
      map.fitBounds(regionBounds, { padding: [48, 48], animate: true, duration: 0.9 })
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Zoom to region when button pressed (skip on first render)
  useEffect(() => {
    if (zoomToRegionTrigger === 0) return
    if (regionBounds) {
      map.fitBounds(regionBounds, { padding: [48, 48], animate: true, duration: 0.7 })
    } else {
      map.flyTo(DEFAULT_CENTER, DEFAULT_ZOOM, { duration: 0.7 })
    }
  }, [zoomToRegionTrigger]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (selected && selectedLayer) {
      const bounds = (selectedLayer as L.GeoJSON).getBounds()
      map.flyToBounds(bounds, { padding: [80, 80], duration: 0.7 })
    } else if (!selected) {
      if (regionBounds) {
        map.fitBounds(regionBounds, { padding: [48, 48], animate: true, duration: 0.7 })
      } else {
        map.flyTo(DEFAULT_CENTER, DEFAULT_ZOOM, { duration: 0.7 })
      }
    }
  }, [selected, selectedLayer, map, regionBounds])

  useEffect(() => {
    function handleClick(e: L.LeafletMouseEvent) {
      if ((e.originalEvent.target as HTMLElement)?.classList?.contains('leaflet-container')) {
        onReset()
      }
    }
    map.on('click', handleClick)
    return () => { map.off('click', handleClick) }
  }, [map, onReset])

  return null
}

// ── Leaderboard ──────────────────────────────────────────────────────────────
function Leaderboard({ ranked, selected, onSelect, region }: {
  ranked:   [string, number][]
  selected: SelectedBorough | null
  onSelect: (name: string, pct: number) => void
  region?:  string
}) {
  return (
    <div style={{
      position:       'absolute',
      top:            48,
      left:           0,
      bottom:         0,
      width:          LEADERBOARD_W,
      zIndex:         1000,
      background:     C.card,
      borderRight:    `1px solid ${C.border}`,
      backdropFilter: 'blur(12px)',
      display:        'flex',
      flexDirection:  'column',
    }}>
      {/* Header */}
      <div style={{ padding: '18px 18px 12px', borderBottom: `1px solid ${C.border}`, flexShrink: 0 }}>
        <p style={{ fontFamily: 'Inter, sans-serif', fontSize: 10, fontWeight: 500, letterSpacing: '0.22em', textTransform: 'uppercase', color: C.textFaint, marginBottom: 4 }}>
          Approval Ranking
        </p>
        <h2 style={{ fontFamily: '"Barlow Condensed", sans-serif', fontWeight: 600, fontSize: '1.25rem', textTransform: 'uppercase', letterSpacing: '-0.01em', color: C.text, marginBottom: 2, lineHeight: 1.1 }}>
          {region ?? 'All England'}
        </h2>
        <p style={{ fontFamily: 'Inter, sans-serif', fontSize: '0.75rem', color: C.textMuted, lineHeight: 1.4 }}>
          {ranked.length} councils · highest approval first. Click to zoom.
        </p>
      </div>

      {/* Ranked list */}
      <div style={{ overflowY: 'auto', flex: 1 }}>
        {ranked.map(([name, pct], i) => {
          const isActive = selected?.name === name
          return (
            <button
              key={name}
              onClick={() => onSelect(name, pct)}
              style={{
                display:     'flex',
                alignItems:  'center',
                gap:         10,
                width:       '100%',
                padding:     '9px 14px 9px 0',
                background:  isActive ? C.rowActive : 'transparent',
                border:      'none',
                borderBottom:`1px solid rgba(200,150,42,0.07)`,
                borderLeft:  `3px solid ${isActive ? C.accent : 'transparent'}`,
                cursor:      'pointer',
                textAlign:   'left',
                paddingLeft: isActive ? '11px' : '13px',
                transition:  'background 0.15s, border-color 0.15s',
              }}
              onMouseEnter={(e) => { if (!isActive) e.currentTarget.style.background = C.rowHover }}
              onMouseLeave={(e) => { if (!isActive) e.currentTarget.style.background = 'transparent' }}
            >
              {/* Rank */}
              <span style={{ fontFamily: 'Inter, sans-serif', fontSize: 10, color: C.textFaint, width: 22, flexShrink: 0, letterSpacing: '0.04em', textAlign: 'right' }}>
                {i + 1}
              </span>

              {/* Name */}
              <span style={{ fontFamily: 'Inter, sans-serif', fontSize: '0.8rem', fontWeight: isActive ? 600 : 400, color: isActive ? C.text : C.textMuted, flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {name}
              </span>

              {/* Mini bar */}
              <div style={{ width: 38, height: 3, background: 'rgba(200,150,42,0.12)', borderRadius: 1, flexShrink: 0 }}>
                <div style={{ width: `${pct}%`, height: '100%', background: getColor(pct), borderRadius: 1 }} />
              </div>

              {/* Percentage */}
              <span style={{ fontFamily: 'Inter, sans-serif', fontSize: '0.75rem', fontWeight: 600, color: getColor(pct), width: 34, textAlign: 'right', flexShrink: 0 }}>
                {pct}%
              </span>
            </button>
          )
        })}
      </div>
    </div>
  )
}

// ── MapPage ──────────────────────────────────────────────────────────────────
export default function MapPage() {
  const { state } = useLocation() as { state: MapState | null }
  const [geoData, setGeoData]                   = useState<FeatureCollection | null>(null)
  const [loading, setLoading]                   = useState(true)
  const [selected, setSelected]                 = useState<SelectedBorough | null>(null)
  const [selectedLayer, setSelectedLayer]       = useState<L.Layer | null>(null)
  const [zoomToRegionTrigger, setZoomToRegion]  = useState(0)
  const selectedLayerRef = useRef<L.Path | null>(null)
  const geoJsonRef       = useRef<L.GeoJSON | null>(null)
  const layerMapRef      = useRef(new Map<string, L.Path>())

  useEffect(() => {
    fetch(GEOJSON_URL)
      .then((r) => r.json())
      .then((data: FeatureCollection) => { setGeoData(data); setLoading(false) })
      .catch(() => setLoading(false))
  }, [])

  // Apply selected highlight after state settles (avoids react-leaflet style override)
  useEffect(() => {
    if (selectedLayerRef.current) {
      selectedLayerRef.current.setStyle({ fillColor: SEL_FILL, weight: 2, color: SEL_BORDER, opacity: 1, fillOpacity: 0.92 })
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

  if (!state) return <Navigate to="/" replace />

  const { boroughs: allBoroughs, region } = { ...state.result, region: state.region }

  // Filter to selected region when one is set
  const regionSet = region ? new Set((REGIONS[region] ?? []).map(normalise)) : null
  const boroughs = regionSet
    ? Object.fromEntries(Object.entries(allBoroughs).filter(([n]) => regionSet.has(normalise(n))))
    : allBoroughs

  const boroughLookup = new Map<string, number>()
  for (const [name, pct] of Object.entries(boroughs)) boroughLookup.set(normalise(name), pct)

  const ranked = [...Object.entries(boroughs)].sort(([, a], [, b]) => b - a) as [string, number][]

  function getPct(featureName: string): number | undefined {
    return boroughLookup.get(normalise(featureName))
  }

  function featureStyle(feature: Feature<Geometry> | undefined): PathOptions {
    const name     = feature?.properties?.LAD13NM ?? feature?.properties?.name ?? ''
    const pct      = getPct(name)
    const inRegion = !regionSet || regionSet.has(normalise(name))

    if (!inRegion) {
      return {
        fillColor:   '#2b2520',
        weight:      0.9,
        opacity:     1,
        color:       'rgba(200, 150, 42, 0.28)',
        fillOpacity: 0.82,
      }
    }

    return {
      fillColor:   pct !== undefined ? getColor(pct) : NO_DATA_FILL,
      weight:      1.2,
      opacity:     1,
      color:       'rgba(200, 150, 42, 0.55)',
      fillOpacity: pct !== undefined ? 0.85 : 0.35,
    }
  }

  function onEachFeature(feature: Feature<Geometry>, layer: Layer) {
    const name     = feature.properties?.LAD13NM ?? feature.properties?.name ?? 'Unknown'
    const pct      = getPct(name)
    const inRegion = !regionSet || regionSet.has(normalise(name))

    layerMapRef.current.set(normalise(name), layer as L.Path)

    if (!inRegion) {
      // Out-of-region: no tooltip, no click, default cursor
      ;(layer as L.Path).options.interactive = false
      return
    }

    layer.bindTooltip(
      pct !== undefined
        ? `<span style="font-family:Inter,sans-serif;font-size:12px"><strong>${name}</strong><br/>${pct}% approval likelihood</span>`
        : `<span style="font-family:Inter,sans-serif;font-size:12px"><strong>${name}</strong><br/>No data</span>`,
      { sticky: true },
    )

    layer.on('click', (e) => {
      L.DomEvent.stopPropagation(e as L.LeafletEvent & { originalEvent: Event })
      if (selectedLayerRef.current && geoJsonRef.current) {
        geoJsonRef.current.resetStyle(selectedLayerRef.current)
      }
      selectedLayerRef.current = layer as L.Path
      setSelected({ name, pct })
      setSelectedLayer(layer)
    })
  }

  const handleLeaderboardSelect = useCallback((name: string, pct: number) => {
    if (selectedLayerRef.current && geoJsonRef.current) {
      geoJsonRef.current.resetStyle(selectedLayerRef.current)
    }
    const layer = layerMapRef.current.get(normalise(name))
    if (layer) {
      selectedLayerRef.current = layer
      setSelected({ name, pct })
      setSelectedLayer(layer)
    } else {
      selectedLayerRef.current = null
      setSelected({ name, pct })
      setSelectedLayer(null)
    }
  }, [])

  return (
    <div style={{ position: 'relative', height: '100vh', width: '100vw', background: MAP_BG }}>
      {loading ? (
        <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontFamily: 'Inter, sans-serif', color: C.textFaint, paddingTop: 48 }}>
          Loading map data…
        </div>
      ) : !geoData ? (
        <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontFamily: 'Inter, sans-serif', color: '#b85c38', paddingTop: 48 }}>
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
            style={{ background: MAP_BG }}
            zoomControl={false}
            attributionControl={false}
          >
            <MapController selected={selected} selectedLayer={selectedLayer} onReset={handleReset} regionBounds={region ? REGION_BOUNDS[region] : undefined} zoomToRegionTrigger={zoomToRegionTrigger} />
            <GeoJSON ref={geoJsonRef} data={geoData} style={featureStyle} onEachFeature={onEachFeature} />
          </MapContainer>

          <Leaderboard ranked={ranked} selected={selected} onSelect={handleLeaderboardSelect} region={region} />

          {/* Gradient legend */}
          <div style={{
            position:   'absolute',
            bottom:     20,
            left:       LEADERBOARD_W + 16,
            zIndex:     1000,
            display:    'flex',
            alignItems: 'center',
            gap:        8,
            fontFamily: 'Inter, sans-serif',
            fontSize:   10,
            letterSpacing: '0.1em',
            color:      C.textFaint,
          }}>
            <span>0%</span>
            <div style={{ width: 72, height: 3, background: `linear-gradient(to right, ${getColor(0)}, ${getColor(100)})` }} />
            <span>100%</span>
            <span style={{ marginLeft: 8 }}>Approval likelihood</span>
          </div>

          {/* Zoom-to-region button */}
          <button
            onClick={() => setZoomToRegion(t => t + 1)}
            title="Zoom to region"
            style={{
              position:       'absolute',
              bottom:         20,
              right:          20,
              zIndex:         1000,
              width:          38,
              height:         38,
              display:        'flex',
              alignItems:     'center',
              justifyContent: 'center',
              background:     C.card,
              border:         `1px solid ${C.border}`,
              backdropFilter: 'blur(10px)',
              cursor:         'pointer',
              transition:     'border-color 0.15s',
            }}
            onMouseEnter={(e) => { e.currentTarget.style.borderColor = 'rgba(200,150,42,0.55)' }}
            onMouseLeave={(e) => { e.currentTarget.style.borderColor = C.border }}
          >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <path d="M1 5V1h4M11 1h4v4M15 11v4h-4M5 15H1v-4" stroke="#c8962a" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </button>

          {selected && (
            <BoroughDetail name={selected.name} pct={selected.pct} onClose={handleReset} />
          )}
        </>
      )}
    </div>
  )
}
