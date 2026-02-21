import { Link, Outlet, useLocation } from 'react-router-dom'

export default function App() {
  const { pathname } = useLocation()
  const isFullWidth = pathname === '/map'

  return (
    <div className="h-screen flex flex-col bg-gray-50 text-gray-900 dark:bg-gray-900 dark:text-gray-100">
      <nav className="flex gap-4 p-4 bg-white shadow dark:bg-gray-800 shrink-0 z-10">
        <Link to="/" className="hover:text-blue-500">Home</Link>
        <Link to="/chat" className="hover:text-blue-500">Chat</Link>
        <Link to="/about" className="hover:text-blue-500">About</Link>
      </nav>
      {isFullWidth ? (
        <Outlet />
      ) : (
        <main className="max-w-3xl mx-auto p-6 flex-1">
          <Outlet />
        </main>
      )}
    </div>
  )
}
