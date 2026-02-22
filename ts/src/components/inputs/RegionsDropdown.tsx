import {useState, useRef, useEffect} from 'react'
import {C, tileSel, tileUnsel} from '../theme'

// ── Multi-select region dropdown ─────────────────────────────────────────────
export default function RegionsDropdown({ values, onChange, regionNames }: {
    values:      string[]
    onChange:    (v: string[]) => void
    regionNames: string[]
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

    const label = values.length === 0
        ? 'Select regions…'
        : values.length === 1
        ? values[0]
        : `${values.length} regions selected`

    return (
        <div ref={ref} style={{position: 'relative'}}>
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
                    ...(values.length > 0 ? tileSel : tileUnsel),
                    ...(open && {borderColor: C.borderFocus}),
                }}
            >
                <span style={{color: values.length > 0 ? C.accent : C.textMuted}}>
                    {label}
                </span>
                <svg width="11" height="11" viewBox="0 0 12 12" fill="none"
                     style={{flexShrink: 0, transform: open ? 'rotate(180deg)' : 'none', transition: 'transform 0.18s'}}>
                    <path d="M2 4L6 8L10 4"
                          stroke={values.length > 0 ? C.accent : C.textFaint}
                          strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
            </button>

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
                    {regionNames.map((r) => {
                        const isSel = values.includes(r)
                        return (
                            <button
                                key={r}
                                type="button"
                                onClick={() => onChange(isSel ? values.filter(v => v !== r) : [...values, r])}
                                style={{
                                    display:    'flex',
                                    alignItems: 'center',
                                    gap:        8,
                                    width:      '100%',
                                    textAlign:  'left',
                                    padding:    '8px 12px',
                                    fontFamily: 'Inter, sans-serif',
                                    fontSize:   '0.8rem',
                                    fontWeight: isSel ? 600 : 400,
                                    color:      isSel ? C.accent : C.textMuted,
                                    background: isSel ? 'rgba(200,150,42,0.12)' : 'transparent',
                                    border:     'none',
                                    borderLeft: `2px solid ${isSel ? C.accent : 'transparent'}`,
                                    cursor:     'pointer',
                                    transition: 'background 0.12s',
                                }}
                                onMouseEnter={(e) => { if (!isSel) e.currentTarget.style.background = 'rgba(200,150,42,0.06)' }}
                                onMouseLeave={(e) => { e.currentTarget.style.background = isSel ? 'rgba(200,150,42,0.12)' : 'transparent' }}
                            >
                                {/* Checkbox */}
                                <span style={{
                                    flexShrink:      0,
                                    width:           13,
                                    height:          13,
                                    border:          `1.5px solid ${isSel ? C.accent : C.border}`,
                                    background:      isSel ? C.accent : 'transparent',
                                    display:         'flex',
                                    alignItems:      'center',
                                    justifyContent:  'center',
                                }}>
                                    {isSel && (
                                        <svg width="8" height="8" viewBox="0 0 8 8" fill="none">
                                            <path d="M1.5 4L3.5 6L6.5 2" stroke="#0a0704" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                                        </svg>
                                    )}
                                </span>
                                {r}
                            </button>
                        )
                    })}
                </div>
            )}
        </div>
    )
}

