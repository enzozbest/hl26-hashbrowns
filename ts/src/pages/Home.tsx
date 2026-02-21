import { useState } from 'react'
import MapBackground from '../components/MapBackground'

// ── Palette ─────────────────────────────────────────────────────────────────
const C = {
  overlay:     'rgba(10, 7, 4, 0.78)',
  card:        'rgba(14, 10, 6, 0.87)',
  cardAlt:     'rgba(20, 14, 8, 0.75)',
  text:        '#ede8df',
  textMuted:   'rgba(237, 232, 223, 0.65)',
  textFaint:   'rgba(237, 232, 223, 0.50)',
  accent:      '#c8962a',
  accentDim:   'rgba(200, 150, 42, 0.45)',
  accentFaint: 'rgba(200, 150, 42, 0.2)',
  accentBtn:   '#a97820',
  accentHover: '#bf8c24',
  border:      'rgba(200, 150, 42, 0.22)',
  borderFocus: 'rgba(200, 150, 42, 0.58)',
  inputBg:     'rgba(255, 255, 255, 0.03)',
}

const RISK = {
  low:    { colour: '#6b8f5e', label: 'Low risk'  },
  medium: { colour: '#c8962a', label: 'Med. risk' },
  high:   { colour: '#b85c38', label: 'High risk' },
}

const HISTORY: {
  id: number
  title: string
  location: string
  date: string
  snippet: string
  risk: keyof typeof RISK
  pinned: boolean
}[] = [
  {
    id: 1,
    title: 'Battersea Power Station',
    location: 'SW8 · Lambeth',
    date: '14 Feb 2026',
    snippet: '180-unit mixed-use residential scheme on former industrial land adjacent to the Thames riverside corridor.',
    risk: 'low',
    pinned: true,
  },
  {
    id: 2,
    title: 'Canary Wharf East',
    location: 'E14 · Tower Hamlets',
    date: '12 Feb 2026',
    snippet: 'Class A office tower, 32 storeys. Site falls within flood zone 3 with tidal surge risk implications.',
    risk: 'high',
    pinned: false,
  },
  {
    id: 3,
    title: "King's Cross Retail",
    location: 'N1C · Camden',
    date: '10 Feb 2026',
    snippet: 'Ground-floor retail and F&B units as part of a wider regeneration masterplan around St Pancras.',
    risk: 'medium',
    pinned: false,
  },
]

export default function Home() {
  const [proposal, setProposal] = useState('')

  const handleSubmit = () => {
    if (proposal.trim()) {
      // TODO: navigate to analysis page
    }
  }

  return (
    <div className="relative min-h-screen flex overflow-hidden">

      {/* ── Layer 0: drifting map ── */}
      <MapBackground />

      {/* ── Layer 1: warm dark overlay ── */}
      <div className="absolute inset-0" style={{ background: C.overlay, zIndex: 10 }} />

      {/* ── Layer 2: page content ── */}
      <div className="absolute inset-0 flex w-full" style={{ zIndex: 20 }}>

        {/* ── Left: headline ── */}
        <div className="flex-1 flex items-center px-16 py-16">
          <div className="max-w-lg">

            {/* Brand */}
            <div className="flex items-center gap-2.5 mb-12">
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                <rect x="1" y="1" width="12" height="12"
                  stroke={C.accent} strokeWidth="1.5" transform="rotate(45 7 7)" />
              </svg>
              <span style={{ fontFamily: 'Inter, sans-serif', letterSpacing: '0.18em', color: C.accent }}
                className="text-[11px] font-medium uppercase">
                Clearance
              </span>
            </div>

            {/* Annotation rule */}
            <div className="flex items-center gap-3 mb-8">
              <div className="h-px w-8" style={{ background: C.accentDim }} />
              <span style={{ fontFamily: 'Inter, sans-serif', letterSpacing: '0.26em', color: C.accentDim }}
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
              Know before<br />you build
            </h1>

            <p className="mt-6 text-[0.95rem] leading-relaxed max-w-[22rem] font-light"
              style={{ fontFamily: 'Inter, sans-serif', color: C.textMuted }}>
              Describe your development proposal. Clearance ranks every UK council by approval likelihood, decision speed, and planning constraints — so you know where to build before you commit to a site.
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
                style={{ borderColor: C.accentDim }} />
            ))}

            <p className="text-[10px] font-medium uppercase mb-1"
              style={{ fontFamily: 'Inter, sans-serif', letterSpacing: '0.24em', color: C.textFaint }}>
              Proposal Input
            </p>
            <h2 className="text-[1.05rem] font-semibold mb-4"
              style={{ fontFamily: 'Inter, sans-serif', color: C.text }}>
              Describe your proposal — development type, scale, and intended use. e.g. "50-unit residential scheme, mid-rise, predominantly affordable housing"
            </h2>

            <textarea
              value={proposal}
              onChange={(e) => setProposal(e.target.value)}
              placeholder="Enter the key details of your planning proposal — development type, location, scale, intended use, and any known site constraints or sensitivities..."
              rows={7}
              className="w-full p-4 text-[0.875rem] leading-relaxed resize-none focus:outline-none transition-colors duration-200"
              style={{
                fontFamily: 'Inter, sans-serif',
                background: C.inputBg,
                border: `1px solid ${C.border}`,
                color: C.text,
              }}
              onFocus={(e) => (e.currentTarget.style.borderColor = C.borderFocus)}
              onBlur={(e)  => (e.currentTarget.style.borderColor = C.border)}
            />

            <button
              onClick={handleSubmit}
              disabled={!proposal.trim()}
              className="mt-4 w-full py-3.5 text-[0.875rem] font-medium cursor-pointer disabled:cursor-default"
              style={{
                fontFamily: 'Inter, sans-serif',
                letterSpacing: '0.06em',
                background: proposal.trim() ? C.accentBtn : C.accentFaint,
                color: C.text,
                transition: 'background 0.2s',
              }}
              onMouseEnter={(e) => { if (proposal.trim()) e.currentTarget.style.background = C.accentHover }}
              onMouseLeave={(e) => { if (proposal.trim()) e.currentTarget.style.background = C.accentBtn }}
            >
              Analyse Proposal
            </button>
          </div>

          {/* ── History section ── */}
          <div>
            {/* Section heading */}
            <div className="flex items-center gap-3 mb-3">
              <span className="text-[10px] font-medium uppercase"
                style={{ fontFamily: 'Inter, sans-serif', letterSpacing: '0.22em', color: C.textFaint }}>
                Recent Reports
              </span>
              <div className="flex-1 h-px" style={{ background: C.border }} />
            </div>

            {/* History cards */}
            <div className="grid grid-cols-3 gap-3">
              {HISTORY.map((item) => (
                <button
                  key={item.id}
                  className="relative text-left cursor-pointer group"
                  style={{
                    background: C.cardAlt,
                    border: `1px solid ${C.border}`,
                    padding: '12px 14px',
                    transition: 'border-color 0.18s',
                  }}
                  onMouseEnter={(e) => (e.currentTarget.style.borderColor = C.accentDim)}
                  onMouseLeave={(e) => (e.currentTarget.style.borderColor = C.border)}
                >
                  {/* Coloured left spine */}
                  <span className="absolute left-0 top-0 bottom-0 w-[3px]"
                    style={{ background: RISK[item.risk].colour }} />

                  {/* Pinned badge */}
                  {item.pinned && (
                    <span className="absolute top-2 right-2 text-[8px] font-medium uppercase px-1.5 py-0.5"
                      style={{
                        fontFamily: 'Inter, sans-serif',
                        letterSpacing: '0.12em',
                        background: C.accentFaint,
                        color: C.accent,
                      }}>
                      Pinned
                    </span>
                  )}

                  <p className="text-[0.8rem] font-semibold leading-tight mb-0.5 pr-10"
                    style={{ fontFamily: 'Inter, sans-serif', color: C.text }}>
                    {item.title}
                  </p>
                  <p className="text-[10px] mb-2"
                    style={{ fontFamily: 'Inter, sans-serif', color: C.textFaint, letterSpacing: '0.04em' }}>
                    {item.location} · {item.date}
                  </p>
                  <p className="text-[0.75rem] leading-snug mb-3"
                    style={{
                      fontFamily: 'Inter, sans-serif',
                      color: C.textMuted,
                      display: '-webkit-box',
                      WebkitLineClamp: 2,
                      WebkitBoxOrient: 'vertical',
                      overflow: 'hidden',
                    }}>
                    {item.snippet}
                  </p>

                  {/* Risk badge */}
                  <div className="flex items-center gap-1.5">
                    <span className="w-1.5 h-1.5 rounded-full"
                      style={{ background: RISK[item.risk].colour }} />
                    <span className="text-[10px] font-medium"
                      style={{
                        fontFamily: 'Inter, sans-serif',
                        color: RISK[item.risk].colour,
                        letterSpacing: '0.06em',
                      }}>
                      {RISK[item.risk].label}
                    </span>
                  </div>
                </button>
              ))}
            </div>
          </div>

        </div>
      </div>
    </div>
  )
}
