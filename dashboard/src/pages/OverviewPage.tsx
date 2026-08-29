import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { modelsApi, alertsApi } from '../api/endpoints'
import type { MLModel, Alert } from '../api/types'

export default function OverviewPage() {
  const [models, setModels] = useState<MLModel[]>([])
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([modelsApi.list(), alertsApi.list({ acknowledged: false })])
      .then(([m, a]) => {
        setModels(m.data)
        setAlerts(a.data)
      })
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div className="text-gray-400">Loading...</div>

  const high = alerts.filter(a => a.severity === 'high').length
  const medium = alerts.filter(a => a.severity === 'medium').length
  const low = alerts.filter(a => a.severity === 'low').length

  return (
    <div className="space-y-8">
      <h1 className="text-2xl font-semibold">Overview</h1>

      <div className="grid grid-cols-4 gap-4">
        <StatCard label="Registered Models" value={models.length} color="blue" />
        <StatCard label="High Severity Alerts" value={high} color="red" />
        <StatCard label="Medium Severity" value={medium} color="yellow" />
        <StatCard label="Low Severity" value={low} color="green" />
      </div>

      <div>
        <h2 className="text-lg font-medium mb-4">Registered Models</h2>
        <div className="space-y-2">
          {models.map(m => (
            <Link
              key={m.id}
              to={`/models/${m.id}`}
              className="flex items-center justify-between p-4 bg-gray-900 border border-gray-800 rounded-lg hover:border-blue-600 transition-colors"
            >
              <div>
                <div className="font-medium">{m.name}</div>
                <div className="text-sm text-gray-400">{m.model_type} · {m.feature_names.length} features</div>
              </div>
              <span className="text-blue-400 text-sm">View →</span>
            </Link>
          ))}
        </div>
      </div>
    </div>
  )
}

function StatCard({ label, value, color }: { label: string; value: number; color: string }) {
  const colors: Record<string, string> = {
    blue: 'text-blue-400',
    red: 'text-red-400',
    yellow: 'text-yellow-400',
    green: 'text-green-400',
  }
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
      <div className={`text-3xl font-bold ${colors[color]}`}>{value}</div>
      <div className="text-sm text-gray-400 mt-1">{label}</div>
    </div>
  )
}