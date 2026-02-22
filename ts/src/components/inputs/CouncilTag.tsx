import {C} from '../theme'

// ── Council tag chip ──────────────────────────────────────────────────────────
export default function CouncilTag({ label, onRemove, variant }: {
    label:    string
    onRemove: () => void
    variant:  'add' | 'exclude'
}) {
    const isAdd = variant === 'add'
    return (
        <span style={{
            display:    'inline-flex',
            alignItems: 'center',
            gap:        4,
            padding:    '3px 6px 3px 8px',
            fontFamily: 'Inter, sans-serif',
            fontSize:   '0.72rem',
            fontWeight: 500,
            background: isAdd ? 'rgba(200,150,42,0.12)' : 'rgba(184,92,56,0.12)',
            border:     `1px solid ${isAdd ? C.accentDim : 'rgba(184,92,56,0.45)'}`,
            color:      isAdd ? C.accent : '#b85c38',
        }}>
            {label}
            <button
                type="button"
                onClick={onRemove}
                style={{
                    display:         'flex',
                    alignItems:      'center',
                    justifyContent:  'center',
                    background:      'none',
                    border:          'none',
                    cursor:          'pointer',
                    padding:         '0 2px',
                    color:           isAdd ? C.accent : '#b85c38',
                    fontSize:        '0.95rem',
                    lineHeight:      1,
                    opacity:         0.8,
                }}
            >×</button>
        </span>
    )
}

