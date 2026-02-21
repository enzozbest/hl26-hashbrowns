import {useState, useRef, useEffect} from 'react'
import MapBackground from '../components/MapBackground'
import {submitQuery} from "../api/client.ts";
import {useNavigate} from 'react-router-dom'
import {REGION_NAMES} from '../data/regions'

// ── Palette ─────────────────────────────────────────────────────────────────
const C = {
    overlay: 'rgba(10, 7, 4, 0.78)',
    card: 'rgba(14, 10, 6, 0.87)',
    cardAlt: 'rgba(20, 14, 8, 0.75)',
    text: '#ede8df',
    textMuted: 'rgba(237, 232, 223, 0.65)',
    textFaint: 'rgba(237, 232, 223, 0.50)',
    accent: '#c8962a',
    accentDim: 'rgba(200, 150, 42, 0.45)',
    accentFaint: 'rgba(200, 150, 42, 0.2)',
    accentBtn: '#a97820',
    accentHover: '#bf8c24',
    border: 'rgba(200, 150, 42, 0.22)',
    borderFocus: 'rgba(200, 150, 42, 0.58)',
    inputBg: 'rgba(255, 255, 255, 0.03)',
}


const DEV_TYPES = ['Residential', 'Commercial', 'Mixed Use', 'Extension / Conversion', 'Industrial', 'Educational']
const SCALES    = ['Small (1–10 units)', 'Medium (11–50 units)', 'Large (50+ units)']

// Shared tile style helpers — keep all colours from palette
const tileSel   = { background: 'rgba(200, 150, 42, 0.15)', border: `1px solid ${C.accent}`,   color: C.accent  }
const tileUnsel = { background: 'rgba(255,255,255,0.03)',    border: `1px solid ${C.border}`,   color: C.textMuted }

function RegionDropdown({ value, onChange }: {
    value:    string | null
    onChange: (v: string | null) => void
}) {
    const [open, setOpen] = useState(false)
    const ref = useRef<HTMLDivElement>(null)

    useEffect(() => {
        if (!open) return
        function onOutsideClick(e: MouseEvent) {
            if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
        }
        document.addEventListener('mousedown', onOutsideClick)
        return () => document.removeEventListener('mousedown', onOutsideClick)
    }, [open])

    return (
        <div ref={ref} style={{position: 'relative'}}>

            {/* Trigger button */}
            <button
                type="button"
                onClick={() => setOpen(o => !o)}
                style={{
                    width: '100%',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '8px 12px',
                    fontFamily: 'Inter, sans-serif',
                    fontSize: '0.8rem',
                    fontWeight: 500,
                    cursor: 'pointer',
                    transition: 'all 0.15s',
                    ...(value ? tileSel : tileUnsel),
                    ...(open && {borderColor: C.borderFocus}),
                }}
            >
                <span style={{color: value ? C.accent : C.textMuted}}>
                    {value ?? 'Select a region…'}
                </span>
                <svg width="11" height="11" viewBox="0 0 12 12" fill="none"
                     style={{flexShrink: 0, transform: open ? 'rotate(180deg)' : 'none', transition: 'transform 0.18s'}}>
                    <path d="M2 4L6 8L10 4"
                          stroke={value ? C.accent : C.textFaint}
                          strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
            </button>

            {/* Option list */}
            {open && (
                <div style={{
                    position:       'absolute',
                    top:            'calc(100% + 3px)',
                    left:           0,
                    right:          0,
                    zIndex:         50,
                    background:     'rgba(12, 8, 4, 0.98)',
                    border:         `1px solid ${C.border}`,
                    backdropFilter: 'blur(12px)',
                    maxHeight:      '260px',
                    overflowY:      'auto',
                }}>
                    {REGION_NAMES.map((r) => {
                        const isSelected = value === r
                        return (
                            <button
                                key={r}
                                type="button"
                                onClick={() => { onChange(r); setOpen(false) }}
                                style={{
                                    display:     'block',
                                    width:       '100%',
                                    textAlign:   'left',
                                    padding:     '8px 12px',
                                    fontFamily:  'Inter, sans-serif',
                                    fontSize:    '0.8rem',
                                    fontWeight:  isSelected ? 600 : 400,
                                    color:       isSelected ? C.accent : C.textMuted,
                                    background:  isSelected ? 'rgba(200,150,42,0.12)' : 'transparent',
                                    border:      'none',
                                    borderLeft:  `2px solid ${isSelected ? C.accent : 'transparent'}`,
                                    cursor:      'pointer',
                                    transition:  'background 0.12s',
                                }}
                                onMouseEnter={(e) => { if (!isSelected) e.currentTarget.style.background = 'rgba(200,150,42,0.06)' }}
                                onMouseLeave={(e) => { if (!isSelected) e.currentTarget.style.background = 'transparent' }}
                            >
                                {r}
                            </button>
                        )
                    })}
                </div>
            )}
        </div>
    )
}

export default function Home() {
    const [devType,  setDevType]  = useState<string | null>(null)
    const [region,   setRegion]   = useState<string | null>(null)
    const [scale,    setScale]    = useState<string | null>(null)
    const navigate = useNavigate()

    const allSelected = !!(devType && region && scale)

    async function handleSubmit(e: React.FormEvent) {
        e.preventDefault()
        if (!allSelected) return
        const query = `${devType}, ${scale}, region: ${region}`
        try {
            const data = await submitQuery(query)
            navigate('/map', {state: {query, result: data, region}})
        } catch (err) {
            console.log("Error: ", err)
        }
    }

    return (
        <div className="relative min-h-screen flex overflow-hidden">

            {/* ── Layer 0: drifting map ── */}
            <MapBackground/>

            {/* ── Layer 1: warm dark overlay ── */}
            <div className="absolute inset-0" style={{background: C.overlay, zIndex: 10}}/>

            {/* ── Layer 2: page content ── */}
            <div className="absolute inset-0 flex w-full" style={{zIndex: 20}}>

                {/* ── Left: headline ── */}
                <div className="flex-1 flex items-center px-16 py-16">
                    <div className="max-w-lg">

                        {/* Annotation rule */}
                        <div className="flex items-center gap-3 mb-8">
                            <div className="h-px w-8" style={{background: C.accentDim}}/>
                            <span style={{fontFamily: 'Inter, sans-serif', letterSpacing: '0.26em', color: C.accentDim}}
                                  className="text-[10px] font-medium uppercase">
                Council Intelligence Platform
              </span>
                        </div>

                        {/* Main heading */}
                        <h1 className="leading-[1.04] uppercase" style={{
                            fontFamily: '"Barlow Condensed", sans-serif',
                            fontWeight: 600,
                            fontSize: 'clamp(2.8rem, 4.5vw, 4.8rem)',
                            letterSpacing: '-0.01em',
                            color: C.text,
                        }}>
                            Know before<br/>you build
                        </h1>

                        <p className="mt-6 text-[0.95rem] leading-relaxed max-w-[22rem] font-light"
                           style={{fontFamily: 'Inter, sans-serif', color: C.textMuted}}>
                            Describe your development proposal. Clearance ranks every UK council by approval likelihood,
                            decision speed, and planning constraints — so you know where to build before you commit to a
                            site.
                        </p>

                        {/* Capability tags */}
                        <div className="mt-9 flex flex-wrap gap-2">
                            {['Council ranking', 'Approval rates', 'Constraint mapping', 'Decision timelines'].map((tag) => (
                                <span key={tag} className="px-3 py-1.5 text-[10.5px] font-medium uppercase"
                                      style={{
                                          fontFamily: 'Inter, sans-serif',
                                          letterSpacing: '0.12em',
                                          border: `1px solid ${C.accentFaint}`,
                                          color: C.textMuted,
                                      }}>
                  {tag}
                </span>
                            ))}
                        </div>

                    </div>
                </div>

                {/* ── Right: form + history ── */}
                <div className="w-[55%] flex flex-col justify-center px-12 py-10 gap-5 overflow-y-auto">

                    {/* Form card */}
                    <div className="relative p-7" style={{
                        background: C.card,
                        border: `1px solid ${C.border}`,
                        backdropFilter: 'blur(12px)',
                    }}>
                        {/* Corner brackets */}
                        {[
                            'top-0 left-0 border-t border-l',
                            'top-0 right-0 border-t border-r',
                            'bottom-0 left-0 border-b border-l',
                            'bottom-0 right-0 border-b border-r',
                        ].map((cls, i) => (
                            <span key={i} className={`absolute w-3.5 h-3.5 ${cls}`}
                                  style={{borderColor: C.accentDim}}/>
                        ))}

                        <p className="text-[10px] font-medium uppercase mb-1"
                           style={{fontFamily: 'Inter, sans-serif', letterSpacing: '0.24em', color: C.textFaint}}>
                            Proposal Input
                        </p>
                        <h2 className="text-[1.05rem] font-semibold mb-5"
                            style={{fontFamily: 'Inter, sans-serif', color: C.text}}>
                            Define your development parameters
                        </h2>

                        {/* ── Step 1: Development Type ── */}
                        <p className="text-[10px] font-medium uppercase mb-2"
                           style={{fontFamily: 'Inter, sans-serif', letterSpacing: '0.22em', color: C.textFaint}}>
                            01 — Development Type
                        </p>
                        <div className="grid grid-cols-3 gap-2 mb-5">
                            {DEV_TYPES.map((t) => (
                                <button
                                    key={t}
                                    type="button"
                                    onClick={() => setDevType(t)}
                                    className="py-2 px-3 text-left text-[0.8rem] font-medium"
                                    style={{
                                        fontFamily: 'Inter, sans-serif',
                                        transition: 'all 0.15s',
                                        ...(devType === t ? tileSel : tileUnsel),
                                    }}
                                >
                                    {t}
                                </button>
                            ))}
                        </div>

                        {/* ── Step 2: Region ── */}
                        <p className="text-[10px] font-medium uppercase mb-2"
                           style={{fontFamily: 'Inter, sans-serif', letterSpacing: '0.22em', color: C.textFaint}}>
                            02 — Region
                        </p>
                        <div className="mb-5">
                            <RegionDropdown value={region} onChange={setRegion} />
                        </div>

                        {/* ── Step 3: Scale ── */}
                        <p className="text-[10px] font-medium uppercase mb-2"
                           style={{fontFamily: 'Inter, sans-serif', letterSpacing: '0.22em', color: C.textFaint}}>
                            03 — Scale
                        </p>
                        <div className="flex gap-2 mb-5">
                            {SCALES.map((s) => (
                                <button
                                    key={s}
                                    type="button"
                                    onClick={() => setScale(s)}
                                    className="flex-1 py-2 px-3 text-[0.8rem] font-medium"
                                    style={{
                                        fontFamily: 'Inter, sans-serif',
                                        transition: 'all 0.15s',
                                        ...(scale === s ? tileSel : tileUnsel),
                                    }}
                                >
                                    {s}
                                </button>
                            ))}
                        </div>

                        {/* ── Submit ── */}
                        <button
                            onClick={handleSubmit}
                            disabled={!allSelected}
                            className="w-full py-3.5 text-[0.875rem] font-medium cursor-pointer disabled:cursor-default"
                            style={{
                                fontFamily: 'Inter, sans-serif',
                                letterSpacing: '0.06em',
                                background: allSelected ? C.accentBtn : C.accentFaint,
                                color: C.text,
                                transition: 'background 0.2s',
                            }}
                            onMouseEnter={(e) => { if (allSelected) e.currentTarget.style.background = C.accentHover }}
                            onMouseLeave={(e) => { if (allSelected) e.currentTarget.style.background = C.accentBtn }}
                        >
                            Analyse Proposal
                        </button>
                    </div>

                </div>
            </div>
        </div>
    )
}
