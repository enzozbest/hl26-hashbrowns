import { Link, Outlet } from 'react-router-dom'

export default function App() {
  return (
    <div className="min-h-screen bg-gray-50 text-gray-900 dark:bg-gray-900 dark:text-gray-100">
      <nav className="flex gap-4 p-4 bg-white shadow dark:bg-gray-800">
        <Link to="/" className="hover:text-blue-500">Home</Link>
        <Link to="/about" className="hover:text-blue-500">About</Link>
      </nav>
      <main className="max-w-3xl mx-auto p-6">
        <Outlet />
      </main>
    </div>
  )
}
