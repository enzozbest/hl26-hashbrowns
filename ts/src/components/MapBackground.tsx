import { useEffect, useRef } from 'react'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'

// London waypoints — all at the same zoom so transitions are pure pans, no zoom jitter
const ZOOM = 13
const WAYPOINTS: { center: [number, number]; zoom: number }[] = [
  { center: [51.5045, -0.0184], zoom: ZOOM }, // Canary Wharf / Docklands
  { center: [51.5145, -0.1009], zoom: ZOOM }, // City of London
  { center: [51.5308, -0.1238], zoom: ZOOM }, // King's Cross
  { center: [51.4842, -0.1446], zoom: ZOOM }, // Battersea / Nine Elms
  { center: [51.5440, -0.0074], zoom: ZOOM }, // Stratford / Olympic Park
]

// ESRI World Imagery — free satellite tiles, no API key required
const TILE_URL =
  'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'

export default function MapBackground() {
  const containerRef = useRef<HTMLDivElement>(null)
  const mapRef       = useRef<L.Map | null>(null)
  const wpIndex      = useRef(0)
  const timeoutRef   = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    if (!containerRef.current) return

    const map = L.map(containerRef.current, {
      center:           WAYPOINTS[0].center,
      zoom:             WAYPOINTS[0].zoom,
      zoomControl:      false,
      attributionControl: false,
      dragging:         false,
      scrollWheelZoom:  false,
      doubleClickZoom:  false,
      touchZoom:        false,
      keyboard:         false,
    })

    L.tileLayer(TILE_URL, { maxZoom: 18 }).addTo(map)

    mapRef.current = map

    const driftToNext = () => {
      wpIndex.current = (wpIndex.current + 1) % WAYPOINTS.length
      const { center } = WAYPOINTS[wpIndex.current]
      // panTo keeps zoom fixed — no zoom-out/zoom-in jitter, just a slow lateral glide
      map.panTo(center, { animate: true, duration: 10, easeLinearity: 0.25 })
      // next drift fires after pan finishes (~10 s) plus a 6 s rest
      timeoutRef.current = setTimeout(driftToNext, 16000)
    }

    // begin first drift after tiles have had time to load
    timeoutRef.current = setTimeout(driftToNext, 5000)

    return () => {
      if (timeoutRef.current) clearTimeout(timeoutRef.current)
      map.remove()
    }
  }, [])

  return <div ref={containerRef} className="absolute inset-0 w-full h-full" style={{ zIndex: 0 }} />
}
