import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom'
import ModelsPage from './pages/ModelsPage'
import ModelDetailPage from './pages/ModelDetailPage'
import AlertsPage from './pages/AlertsPage'
import OverviewPage from './pages/OverviewPage'



function NavItem({ to, label }: { to: string; label: string }) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) =>
        `px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
          isActive
            ? 'bg-blue-600 text-white'
            : 'text-gray-400 hover:text-white hover:bg-gray-800'
        }`
      }
    >
      {label}
    </NavLink>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gray-950 text-gray-100">
        <nav className="border-b border-gray-800 px-6 py-3 flex items-center gap-6">
          <span className="text-blue-400 font-bold text-lg tracking-tight">
            DriftWatch
          </span>
          <div className="flex gap-1">
            <NavItem to="/" label="Overview" />
            <NavItem to="/models" label="Models" />
            <NavItem to="/alerts" label="Alerts" />
          </div>
        </nav>
        <main className="px-6 py-8 max-w-7xl mx-auto">
          <Routes>
            <Route path="/" element={<OverviewPage />} />
            <Route path="/models" element={<ModelsPage />} />
            <Route path="/models/:id" element={<ModelDetailPage />} />
            <Route path="/alerts" element={<AlertsPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}