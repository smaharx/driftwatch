import { useEffect, useState } from 'react'
import { alertsApi } from '../api/endpoints'
import type { Alert } from '../api/types'

const SEVERITY_STYLES: Record<string, string> = {
  high: 'bg-red-900 text-red-400',
  medium: 'bg-yellow-900 text-yellow-400',
  low: 'bg-green-900 text-green-400',
}

export default function AlertsPage() {
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState<'all' | 'unacknowledged'>('unacknowledged')

  useEffect(() => {
    const params = filter === 'unacknowledged' ? { acknowledged: false } : {}
    alertsApi.list(params).then(r => setAlerts(r.data)).finally(() => setLoading(false))
  }, [filter])

  const handleAcknowledge = async (alertId: string) => {
    await alertsApi.acknowledge(alertId)
    setAlerts(prev => prev.filter(a => a.id !== alertId))
  }

  if (loading) return <div className="text-gray-400">Loading...</div>

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold">Alerts</h1>
        <div className="flex gap-2">
          {(['all', 'unacknowledged'] as const).map(f => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`px-3 py-1.5 rounded text-sm font-medium transition-colors ${
                filter === f ? 'bg-blue-600 text-white' : 'bg-gray-800 text-gray-400 hover:text-white'
              }`}
            >
              {f === 'all' ? 'All' : 'Unacknowledged'}
            </button>
          ))}
        </div>
      </div>

      {alerts.length === 0 ? (
        <div className="text-gray-400 text-sm">No alerts found.</div>
      ) : (
        <div className="space-y-2">
          {alerts.map(alert => (
            <div key={alert.id} className="flex items-center justify-between p-4 bg-gray-900 border border-gray-800 rounded-lg">
              <div className="space-y-1">
                <div className="flex items-center gap-3">
                  <span className={`px-2 py-0.5 rounded text-xs font-medium ${SEVERITY_STYLES[alert.severity]}`}>
                    {alert.severity}
                  </span>
                  <span className="font-medium">{alert.feature_name}</span>
                  <span className="text-gray-400 text-sm">{alert.detector_type}</span>
                </div>
                <div className="text-sm text-gray-400">
                  Score: {alert.drift_score.toFixed(4)} · Threshold: {alert.threshold} · {new Date(alert.created_at).toLocaleString()}
                </div>
              </div>
              {!alert.acknowledged && (
                <button
                  onClick={() => handleAcknowledge(alert.id)}
                  className="px-3 py-1.5 bg-gray-800 hover:bg-gray-700 text-sm rounded transition-colors"
                >
                  Acknowledge
                </button>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}