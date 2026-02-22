import {useState, useEffect, useMemo} from 'react'
import MapBackground from '../components/MapBackground'
import FormCard from '../components/FormCard'
import {submitAnalyse, fetchCouncils, type CouncilInfo} from "../api/client.ts";
import {useNavigate} from 'react-router-dom'
import {REGION_ORDER} from '../data/regions'
import RegionsDropdown from '../components/inputs/RegionsDropdown'
import CouncilSearchDropdown from '../components/inputs/CouncilSearchDropdown'
import CouncilTag from '../components/inputs/CouncilTag'
import {C, tileSel, tileUnsel} from '../components/theme'

const DEV_TYPES = ['Residential', 'Commercial', 'Mixed Use', 'Extension / Conversion', 'Industrial', 'Educational']
const SCALES    = ['Small (1–10 units)', 'Medium (11–50 units)', 'Large (50+ units)']


// ── Home page ────────────────────────────────────────────────────────────────
export default function Home() {
    const [devType,         setDevType]         = useState<string | null>(null)
    const [regions,         setRegions]         = useState<string[]>([])
    const [scale,           setScale]           = useState<string | null>(null)
    const [addedCouncils,   setAddedCouncils]   = useState<string[]>([])
    const [removedCouncils, setRemovedCouncils] = useState<string[]>([])
    const [councilData,     setCouncilData]     = useState<CouncilInfo[]>([])
    const [specify,         setSpecify]         = useState('')
    const [loading,         setLoading]         = useState(false)
    const [error,           setError]           = useState<string | null>(null)
    const navigate = useNavigate()

    // Fetch council list from backend on mount
    useEffect(() => {
        fetchCouncils().then(setCouncilData).catch(console.error)
    }, [])

    // Derive region → [lad_name] mapping from API data
    const regionsMap = useMemo<Record<string, string[]>>(() => {
        const map: Record<string, string[]> = {}
        for (const c of councilData) {
            if (!c.region) continue
            ;(map[c.region] ??= []).push(c.lad_name)
        }
        return map
    }, [councilData])

    // Sorted list of region names (by canonical order)
    const regionNames = useMemo(
        () => REGION_ORDER.filter(r => r in regionsMap),
        [regionsMap]
    )

    const allSelected = !!(devType && regions.length > 0 && scale)

    // When regions change, prune council adjustments that are now invalid
    useEffect(() => {
        const rc = new Set(regions.flatMap(r => regionsMap[r] ?? []))
        setAddedCouncils(prev  => prev.filter(c => !rc.has(c)))
        setRemovedCouncils(prev => prev.filter(c =>  rc.has(c)))
    }, [regions, regionsMap])

    // Derived: all councils belonging to selected regions (deduped)
    const regionCouncils = [...new Set(regions.flatMap(r => regionsMap[r] ?? []))]

    // Councils available to add (not in any selected region, not already added)
    const addableCouncils = [...new Set(
        Object.values(regionsMap).flat().filter(c => !regionCouncils.includes(c) && !addedCouncils.includes(c))
    )].sort()

    // Councils available to exclude (in selected regions, not already excluded)
    const excludableCouncils = regionCouncils.filter(c => !removedCouncils.includes(c)).sort()

    async function handleSubmit(e: React.FormEvent) {
        e.preventDefault()
        if (!allSelected) return

        setLoading(true)
        setError(null)

        const queryParts: string[] = [
            `To build a ${devType} development at a ${scale} scale in regions ${regions.join(', ')}`,
        ]
        if (addedCouncils.length > 0) {
            queryParts.push(`including ${addedCouncils.join(', ')}`)
        }
        if (removedCouncils.length > 0) {
            queryParts.push(`excluding ${removedCouncils.join(', ')}`)
        }
        if (specify.trim().length > 0) {
            queryParts.push(`and ${specify.trim()}`)
        }
        const query = `${queryParts.join(', ')}.`
        try {
            // Compute effective council IDs: (region councils − removed) ∪ added
            const regionCouncilIds = councilData
                .filter(c => c.region && regions.includes(c.region))
                .map(c => c.council_id)
            const addedIds = councilData
                .filter(c => addedCouncils.includes(c.lad_name))
                .map(c => c.council_id)
            const removedIds = new Set(
                councilData.filter(c => removedCouncils.includes(c.lad_name)).map(c => c.council_id)
            )
            const effectiveIds = [
                ...regionCouncilIds.filter(id => !removedIds.has(id)),
                ...addedIds,
            ]
            const data = await submitAnalyse(effectiveIds, query)
            navigate('/map', {state: {query, analysisId: data.analysis_id, analyseResults: data.scores, regions, addedCouncils, removedCouncils}})
        } catch (err) {
            const errorMessage = err instanceof Error ? err.message : 'Failed to submit analysis'
            setError(errorMessage)
            console.error("Error: ", err)
        } finally {
            setLoading(false)
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

                        <div className="flex items-center gap-3 mb-8">
                            <div className="h-px w-8" style={{background: C.accentDim}}/>
                            <span style={{fontFamily: 'Inter, sans-serif', letterSpacing: '0.26em', color: C.accentDim}}
                                  className="text-[10px] font-medium uppercase">
                                Council Intelligence Platform
                            </span>
                        </div>

                        <h1 className="leading-[1.04] uppercase" style={{
                            fontFamily:    '"Barlow Condensed", sans-serif',
                            fontWeight:    600,
                            fontSize:      'clamp(2.8rem, 4.5vw, 4.8rem)',
                            letterSpacing: '-0.01em',
                            color:         C.text,
                        }}>
                            Know before<br/>you build
                        </h1>

                        <p className="mt-6 text-[0.95rem] leading-relaxed max-w-[22rem] font-light"
                           style={{fontFamily: 'Inter, sans-serif', color: C.textMuted}}>
                            Describe your development proposal. Clearance ranks every UK council by approval likelihood,
                            decision speed, and planning constraints — so you know where to build before you commit to a
                            site.
                        </p>

                        <div className="mt-9 flex flex-wrap gap-2">
                            {['Council ranking', 'Approval rates', 'Constraint mapping', 'Decision timelines'].map((tag) => (
                                <span key={tag} className="px-3 py-1.5 text-[10.5px] font-medium uppercase"
                                      style={{
                                          fontFamily:    'Inter, sans-serif',
                                          letterSpacing: '0.12em',
                                          border:        `1px solid ${C.accentFaint}`,
                                          color:         C.textMuted,
                                      }}>
                                    {tag}
                                </span>
                            ))}
                        </div>

                    </div>
                </div>

                {/* ── Right: form ── */}
                <div className="w-[55%] flex flex-col justify-center px-12 py-10 gap-5 overflow-y-auto">

                    <form onSubmit={handleSubmit} style={{ display: 'contents' }}>
                        <FormCard title="Proposal Input" subtitle="Define your development parameters">

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
                        <div className={regions.length > 0 ? 'mb-3' : 'mb-5'}>
                            <RegionsDropdown values={regions} onChange={setRegions} regionNames={regionNames}/>
                        </div>

                        {/* Council adjustments — visible once at least one region is selected */}
                        {regions.length > 0 && (
                            <div className="mb-5" style={{
                                background: 'rgba(255,255,255,0.018)',
                                border:     `1px solid ${C.border}`,
                                padding:    '10px 12px',
                            }}>

                                {/* Add councils from other regions */}
                                <p style={{
                                    fontFamily:    'Inter, sans-serif',
                                    fontSize:      '0.7rem',
                                    fontWeight:    500,
                                    letterSpacing: '0.18em',
                                    textTransform: 'uppercase',
                                    color:         C.textFaint,
                                    marginBottom:  6,
                                }}>
                                    Also include
                                </p>
                                <CouncilSearchDropdown
                                    placeholder="Add councils from other regions…"
                                    options={addableCouncils}
                                    onSelect={(c) => setAddedCouncils(prev => [...prev, c])}
                                />
                                {addedCouncils.length > 0 && (
                                    <div style={{display: 'flex', flexWrap: 'wrap', gap: 4, marginTop: 6}}>
                                        {addedCouncils.map(c => (
                                            <CouncilTag
                                                key={c}
                                                label={c}
                                                variant="add"
                                                onRemove={() => setAddedCouncils(prev => prev.filter(x => x !== c))}
                                            />
                                        ))}
                                    </div>
                                )}

                                <div style={{height: 1, background: C.border, margin: '10px 0'}}/>

                                {/* Exclude councils from selected regions */}
                                <p style={{
                                    fontFamily:    'Inter, sans-serif',
                                    fontSize:      '0.7rem',
                                    fontWeight:    500,
                                    letterSpacing: '0.18em',
                                    textTransform: 'uppercase',
                                    color:         C.textFaint,
                                    marginBottom:  6,
                                }}>
                                    Exclude
                                </p>
                                <CouncilSearchDropdown
                                    placeholder="Remove specific councils from selection…"
                                    options={excludableCouncils}
                                    onSelect={(c) => setRemovedCouncils(prev => [...prev, c])}
                                />
                                {removedCouncils.length > 0 && (
                                    <div style={{display: 'flex', flexWrap: 'wrap', gap: 4, marginTop: 6}}>
                                        {removedCouncils.map(c => (
                                            <CouncilTag
                                                key={c}
                                                label={c}
                                                variant="exclude"
                                                onRemove={() => setRemovedCouncils(prev => prev.filter(x => x !== c))}
                                            />
                                        ))}
                                    </div>
                                )}
                            </div>
                        )}

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

                        {/* ── Step 4: Specify (Optional) ── */}
                        <p className="text-[10px] font-medium uppercase mb-2"
                           style={{fontFamily: 'Inter, sans-serif', letterSpacing: '0.22em', color: C.textFaint}}>
                            04 — Specify (Optional)
                        </p>
                        <div className="mb-5">
                            <input
                                type="text"
                                placeholder="Specify more details about the potential development..."
                                className="w-full py-2 px-3 text-[0.8rem]"
                                style={{
                                    fontFamily: 'Inter, sans-serif',
                                    background: C.inputBg,
                                    borderWidth: '1px',
                                    borderStyle: 'solid',
                                    borderColor: C.border,
                                    color:      C.text,
                                    transition: 'border-color 0.15s',
                                }}
                                value={specify}
                                onChange={(e) => setSpecify(e.target.value)}
                                onFocus={(e) => e.currentTarget.style.borderColor = C.borderFocus}
                                onBlur={(e)  => e.currentTarget.style.borderColor = C.border}
                            />
                        </div>

                        {/* ── Error message ── */}
                        {error && (
                            <div style={{
                                padding: '8px 12px',
                                marginBottom: 12,
                                background: 'rgba(184, 92, 56, 0.15)',
                                border: '1px solid rgba(184, 92, 56, 0.45)',
                                color: '#b85c38',
                                fontSize: '0.8rem',
                                fontFamily: 'Inter, sans-serif',
                                borderRadius: '2px',
                            }}>
                                {error}
                            </div>
                        )}

                        {/* ── Submit ── */}
                        <button
                            type="submit"
                            disabled={!allSelected || loading}
                            className="w-full py-3.5 text-[0.875rem] font-medium cursor-pointer disabled:cursor-default"
                            style={{
                                fontFamily:    'Inter, sans-serif',
                                letterSpacing: '0.06em',
                                background:    (allSelected && !loading) ? C.accentBtn : C.accentFaint,
                                color:         C.text,
                                transition:    'background 0.2s',
                                opacity:       loading ? 0.6 : 1,
                            }}
                            onMouseEnter={(e) => { if (allSelected && !loading) e.currentTarget.style.background = C.accentHover }}
                            onMouseLeave={(e) => { if (allSelected && !loading) e.currentTarget.style.background = C.accentBtn  }}
                        >
                            {loading ? 'Analysing…' : 'Analyse Proposal'}
                        </button>
                    </FormCard>
                    </form>

                </div>
            </div>
        </div>
    )
}
