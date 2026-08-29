import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'
import { modelsApi, runsApi } from '../api/endpoints'
import type { MLModel, Run } from '../api/types'

export default function ModelDetailPage() {
  const { id } = useParams<{ id: string }>()
  const [model, setModel] = useState<MLModel | null>(null)
  const [runs, setRuns] = useState<Run[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (!id) return
    Promise.all([modelsApi.get(id), runsApi.list(id)])
      .then(([m, r]) => {
        setModel(m.data)
        setRuns(r.data.reverse())
      })
      .finally(() => setLoading(false))
  }, [id])

  if (loading) return <div className="text-gray-400">Loading...</div>
  if (!model) return <div className="text-gray-400">Model not found</div>

  const chartData = runs.map((r, i) => ({
    run: `Run ${i + 1}`,
    score: r.overall_drift_score ?? 0,
    drifted: (r.drifted_features?.length ?? 0) > 0,
  }))

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-semibold">{model.name}</h1>
        <p className="text-gray-400 mt-1">{model.description ?? 'No description'}</p>
      </div>

      <div className="grid grid-cols-3 gap-4">
        <InfoCard label="Model Type" value={model.model_type} />
        <InfoCard label="Features" value={model.feature_names.length.toString()} />
        <InfoCard label="Total Runs" value={runs.length.toString()} />
      </div>

      <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
        <h2 className="text-lg font-medium mb-6">Drift Score Timeline</h2>
        {chartData.length === 0 ? (
          <div className="text-gray-400 text-sm">No runs yet. Submit production data to see drift scores.</div>
        ) : (
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
              <XAxis dataKey="run" stroke="#6b7280" fontSize={12} />
              <YAxis stroke="#6b7280" fontSize={12} />
              <Tooltip
                contentStyle={{ backgroundColor: '#111827', border: '1px solid #374151', borderRadius: '8px' }}
                labelStyle={{ color: '#f9fafb' }}
              />
              <ReferenceLine y={0.2} stroke="#ef4444" strokeDasharray="4 4" label={{ value: 'PSI threshold', fill: '#ef4444', fontSize: 11 }} />
              <Line type="monotone" dataKey="score" stroke="#3b82f6" strokeWidth={2} dot={{ fill: '#3b82f6', r: 4 }} />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>

      <div>
        <h2 className="text-lg font-medium mb-4">Run History</h2>
        <div className="space-y-2">
          {runs.map(r => (
            <div key={r.id} className="flex items-center justify-between p-4 bg-gray-900 border border-gray-800 rounded-lg">
              <div>
                <div className="text-sm font-medium">{r.id.slice(0, 8)}...</div>
                <div className="text-xs text-gray-400">{new Date(r.created_at).toLocaleString()}</div>
              </div>
              <div className="flex items-center gap-4">
                <span className="text-sm text-gray-400">Score: {r.overall_drift_score?.toFixed(4) ?? 'N/A'}</span>
                {r.drifted_features && r.drifted_features.length > 0 && (
                  <span className="px-2 py-0.5 bg-red-900 text-red-400 rounded text-xs">
                    {r.drifted_features.length} drifted
                  </span>
                )}
                <span className={`px-2 py-0.5 rounded text-xs ${r.status === 'completed' ? 'bg-green-900 text-green-400' : 'bg-yellow-900 text-yellow-400'}`}>
                  {r.status}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

function InfoCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
      <div className="text-2xl font-bold text-blue-400">{value}</div>
      <div className="text-sm text-gray-400 mt-1">{label}</div>
    </div>
  )
}