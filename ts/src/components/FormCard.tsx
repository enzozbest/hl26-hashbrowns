import {C} from './theme'

// ── Form card wrapper with corner brackets ────────────────────────────────────
export default function FormCard({ title, subtitle, children }: {
    title:    string
    subtitle: string
    children: React.ReactNode
}) {
    return (
        <div className="relative p-7" style={{
            background:     C.card,
            border:         `1px solid ${C.border}`,
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
                {title}
            </p>
            <h2 className="text-[1.05rem] font-semibold mb-5"
                style={{fontFamily: 'Inter, sans-serif', color: C.text}}>
                {subtitle}
            </h2>

            {children}
        </div>
    )
}

