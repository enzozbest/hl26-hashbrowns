import {useState, useRef, useEffect} from 'react'
import {C} from '../theme'

// ── Searchable council picker ─────────────────────────────────────────────────
export default function CouncilSearchDropdown({ placeholder, options, onSelect }: {
    placeholder: string
    options:     string[]
    onSelect:    (council: string) => void
}) {
    const [search, setSearch] = useState('')
    const [open,   setOpen]   = useState(false)
    const ref = useRef<HTMLDivElement>(null)

    useEffect(() => {
        if (!open) return
        function onOutside(e: MouseEvent) {
            if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
        }
        document.addEventListener('mousedown', onOutside)
        return () => document.removeEventListener('mousedown', onOutside)
    }, [open])

    const filtered = search
        ? options.filter(c => c.toLowerCase().includes(search.toLowerCase()))
        : options

    return (
        <div ref={ref} style={{position: 'relative'}}>
            <input
                type="text"
                placeholder={placeholder}
                value={search}
                onChange={e => { setSearch(e.target.value); setOpen(true) }}
                onFocus={(e) => { e.currentTarget.style.borderColor = C.borderFocus; setOpen(true) }}
                onBlur={(e)  => e.currentTarget.style.borderColor = C.border}
                style={{
                    width:       '100%',
                    padding:     '7px 12px',
                    fontFamily:  'Inter, sans-serif',
                    fontSize:    '0.8rem',
                    background:  C.inputBg,
                    borderWidth: '1px',
                    borderStyle: 'solid',
                    borderColor: C.border,
                    color:       C.text,
                    outline:     'none',
                    transition:  'border-color 0.15s',
                    boxSizing:   'border-box',
                }}
            />
            {open && filtered.length > 0 && (
                <div style={{
                    position:       'absolute',
                    top:            'calc(100% + 2px)',
                    left:           0,
                    right:          0,
                    zIndex:         60,
                    background:     'rgba(12, 8, 4, 0.98)',
                    border:         `1px solid ${C.border}`,
                    backdropFilter: 'blur(12px)',
                    maxHeight:      '160px',
                    overflowY:      'auto',
                }}>
                    {filtered.map(c => (
                        <button
                            key={c}
                            type="button"
                            onMouseDown={(e) => {
                                e.preventDefault()
                                onSelect(c)
                                setSearch('')
                                setOpen(false)
                            }}
                            style={{
                                display:    'block',
                                width:      '100%',
                                textAlign:  'left',
                                padding:    '7px 12px',
                                fontFamily: 'Inter, sans-serif',
                                fontSize:   '0.8rem',
                                color:      C.textMuted,
                                background: 'transparent',
                                border:     'none',
                                cursor:     'pointer',
                                transition: 'background 0.12s',
                            }}
                            onMouseEnter={(e) => e.currentTarget.style.background = 'rgba(200,150,42,0.08)'}
                            onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}
                        >
                            {c}
                        </button>
                    ))}
                </div>
            )}
        </div>
    )
}

