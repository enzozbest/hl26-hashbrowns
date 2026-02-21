import { RISK, REPORTS } from '../data/reports'

// Palette — mirrors Home.tsx
const C = {
    text:      '#ede8df',
    textMuted: 'rgba(237, 232, 223, 0.65)',
    textFaint: 'rgba(237, 232, 223, 0.55)',
    accent:    '#c8962a',
    accentDim: 'rgba(200, 150, 42, 0.45)',
    border:    'rgba(200, 150, 42, 0.18)',
    rowHover:  'rgba(200, 150, 42, 0.05)',
}

const COLS = {
    num:      { width: '2.5rem',  flexShrink: 0 },
    type:     { width: '9rem',    flexShrink: 0 },
    title:    { flex: 1,          minWidth: 0   },
    location: { width: '10.5rem', flexShrink: 0 },
    date:     { width: '7rem',    flexShrink: 0 },
    risk:     { width: '7.5rem',  flexShrink: 0 },
}

const HEADER_LABELS: [keyof typeof COLS, string][] = [
    ['num',      '#'           ],
    ['type',     'Type'        ],
    ['title',    'Report Title'],
    ['location', 'Scope'       ],
    ['date',     'Date'        ],
    ['risk',     'Risk Level'  ],
]

export default function History() {
    return (
        <div style={{ minHeight: '100vh', background: '#0e0a06', paddingTop: '48px' }}>
            <div style={{ maxWidth: '72rem', margin: '0 auto', padding: '3rem 3.5rem 5rem' }}>

                {/* ── Page header ── */}
                <div className="mb-5">
                    <h1 style={{
                        fontFamily: '"Barlow Condensed", sans-serif',
                        fontWeight: 600,
                        fontSize: 'clamp(2rem, 3.5vw, 3rem)',
                        letterSpacing: '-0.01em',
                        textTransform: 'uppercase',
                        color: C.text,
                        lineHeight: 1.05,
                        marginBottom: '0.75rem',
                    }}>
                        Previous Reports
                    </h1>
                </div>

                {/* ── Table ── */}
                <div>
                    {/* Section label + rule */}
                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '12px' }}>
                        <div style={{ flex: 1, height: '1px', background: 'transparent' }} />
                        <span style={{
                            fontFamily: 'Inter, sans-serif',
                            fontSize: '10px',
                            color: C.textFaint,
                            letterSpacing: '0.08em',
                        }}>
                            {REPORTS.length} in total
                        </span>
                    </div>

                    {/* Column headers */}
                    <div style={{
                        display: 'flex',
                        gap: '24px',
                        padding: '8px 16px 8px 20px',
                        borderBottom: `1px solid ${C.border}`,
                        borderTop: `1px solid ${C.border}`,
                    }}>
                        {HEADER_LABELS.map(([key, label]) => (
                            <span key={key} style={{
                                ...COLS[key],
                                fontFamily: 'Inter, sans-serif',
                                fontSize: '10px',
                                fontWeight: 500,
                                letterSpacing: '0.2em',
                                textTransform: 'uppercase',
                                color: C.textFaint,
                                overflow: 'hidden',
                                whiteSpace: 'nowrap',
                            }}>
                                {label}
                            </span>
                        ))}
                    </div>

                    {/* Rows */}
                    {REPORTS.map((item, i) => (
                        <div
                            key={item.id}
                            style={{
                                position: 'relative',
                                borderBottom: `1px solid ${C.border}`,
                                cursor: 'pointer',
                                transition: 'background 0.15s',
                            }}
                            onMouseEnter={(e) => (e.currentTarget.style.background = C.rowHover)}
                            onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
                        >
                            {/* Coloured left spine */}
                            <span style={{
                                position: 'absolute',
                                left: 0,
                                top: 0,
                                bottom: 0,
                                width: '3px',
                                background: RISK[item.risk].colour,
                            }} />

                            {/* Main row */}
                            <div style={{
                                display: 'flex',
                                alignItems: 'baseline',
                                gap: '24px',
                                padding: '16px 16px 10px 20px',
                            }}>
                                {/* # */}
                                <span style={{
                                    ...COLS.num,
                                    fontFamily: 'Inter, sans-serif',
                                    fontSize: '11px',
                                    color: C.textFaint,
                                    letterSpacing: '0.06em',
                                }}>
                                    {String(i + 1).padStart(2, '0')}
                                </span>

                                {/* Type */}
                                <span style={{
                                    ...COLS.type,
                                    fontFamily: 'Inter, sans-serif',
                                    fontSize: '0.75rem',
                                    fontWeight: 500,
                                    color: C.textMuted,
                                    overflow: 'hidden',
                                    whiteSpace: 'nowrap',
                                    textOverflow: 'ellipsis',
                                }}>
                                    {item.type}
                                </span>

                                {/* Title */}
                                <span style={{
                                    ...COLS.title,
                                    fontFamily: 'Inter, sans-serif',
                                    fontSize: '0.875rem',
                                    fontWeight: 600,
                                    color: C.text,
                                    overflow: 'hidden',
                                    whiteSpace: 'nowrap',
                                    textOverflow: 'ellipsis',
                                }}>
                                    {item.title}
                                </span>

                                {/* Location */}
                                <span style={{
                                    ...COLS.location,
                                    fontFamily: 'Inter, sans-serif',
                                    fontSize: '0.8rem',
                                    color: C.textMuted,
                                    overflow: 'hidden',
                                    whiteSpace: 'nowrap',
                                    textOverflow: 'ellipsis',
                                }}>
                                    {item.location}
                                </span>

                                {/* Date */}
                                <span style={{
                                    ...COLS.date,
                                    fontFamily: 'Inter, sans-serif',
                                    fontSize: '0.8rem',
                                    color: C.textFaint,
                                    letterSpacing: '0.02em',
                                }}>
                                    {item.date}
                                </span>

                                {/* Risk */}
                                <div style={{
                                    ...COLS.risk,
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: '6px',
                                }}>
                                    <span style={{
                                        width: '6px',
                                        height: '6px',
                                        borderRadius: '50%',
                                        flexShrink: 0,
                                        background: RISK[item.risk].colour,
                                    }} />
                                    <span style={{
                                        fontFamily: 'Inter, sans-serif',
                                        fontSize: '0.75rem',
                                        fontWeight: 500,
                                        color: RISK[item.risk].colour,
                                        letterSpacing: '0.04em',
                                    }}>
                                        {RISK[item.risk].label}
                                    </span>
                                </div>
                            </div>

                            {/* Snippet row */}
                            <p style={{
                                fontFamily: 'Inter, sans-serif',
                                fontSize: '0.82rem',
                                lineHeight: 1.65,
                                color: C.textMuted,
                                padding: '0 16px 16px',
                                paddingLeft: `calc(20px + 2.5rem + 9rem + 24px + 24px)`,
                                margin: 0,
                            }}>
                                {item.snippet}
                            </p>
                        </div>
                    ))}
                </div>

            </div>
        </div>
    )
}
