import { Link } from 'react-router-dom'

export default function NotFound() {
  return (
    <div>
      <h2>404 - Page Not Found</h2>
      <Link to="/">Go home</Link>
    </div>
  )
}
