import { NavLink, Outlet } from 'react-router-dom'

const NAV_H = '48px'

const linkStyle = (isActive: boolean): React.CSSProperties => ({
    fontFamily: 'Inter, sans-serif',
    fontSize: '13px',
    fontWeight: 500,
    letterSpacing: '0.05em',
    textDecoration: 'none',
    color: isActive ? '#c8962a' : 'rgba(237, 232, 223, 0.50)',
    transition: 'color 0.15s',
})

export default function App() {
    return (
        <>
            <nav style={{
                position: 'fixed',
                top: 0, left: 0, right: 0,
                height: NAV_H,
                background: 'rgba(10, 7, 4, 0.97)',
                borderBottom: '1px solid rgba(200, 150, 42, 0.16)',
                backdropFilter: 'blur(10px)',
                zIndex: 2000,
                display: 'flex',
                alignItems: 'center',
                padding: '0 32px',
                gap: '28px',
            }}>
                {/* Brand mark */}
                <NavLink
                    to="/"
                    style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '8px',
                        marginRight: 'auto',
                        textDecoration: 'none',
                    }}
                >
                    <svg width="12" height="12" viewBox="0 0 14 14" fill="none">
                        <rect
                            x="1"
                            y="1"
                            width="12"
                            height="12"
                            stroke="#c8962a"
                            strokeWidth="1.5"
                            transform="rotate(45 7 7)"
                        />
                    </svg>

                    <span
                        style={{
                            fontFamily: 'Inter, sans-serif',
                            fontSize: '11px',
                            fontWeight: 500,
                            letterSpacing: '0.18em',
                            textTransform: 'uppercase',
                            color: '#c8962a',
                        }}
                    >
                        Siteline
                    </span>
                </NavLink>

                <NavLink to="/history" style={({ isActive }) => linkStyle(isActive)}>History</NavLink>
            </nav>

            <Outlet />
        </>
    )
}
