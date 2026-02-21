import { Link, Outlet } from 'react-router-dom'
import './App.css'

export default function App() {
  return (
    <>
      <nav>
        <Link to="/">Home</Link>
        {' | '}
        <Link to="/about">About</Link>
      </nav>
      <hr />
      <Outlet />
    </>
  )
}
