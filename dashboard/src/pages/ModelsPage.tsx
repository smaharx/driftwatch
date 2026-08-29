import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { modelsApi } from '../api/endpoints'
import type { MLModel } from '../api/types'

export default function ModelsPage() {
  const [models, setModels] = useState<MLModel[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    modelsApi.list().then(r => setModels(r.data)).finally(() => setLoading(false))
  }, [])

  if (loading) return <div className="text-gray-400">Loading...</div>

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold">Model Registry</h1>
      <div className="overflow-hidden border border-gray-800 rounded-lg">
        <table className="w-full text-sm">
          <thead className="bg-gray-900 text-gray-400 text-left">
            <tr>
              <th className="px-4 py-3">Name</th>
              <th className="px-4 py-3">Type</th>
              <th className="px-4 py-3">Features</th>
              <th className="px-4 py-3">Status</th>
              <th className="px-4 py-3">Created</th>
              <th className="px-4 py-3"></th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-800">
            {models.map(m => (
              <tr key={m.id} className="bg-gray-950 hover:bg-gray-900 transition-colors">
                <td className="px-4 py-3 font-medium">{m.name}</td>
                <td className="px-4 py-3 text-gray-400">{m.model_type}</td>
                <td className="px-4 py-3 text-gray-400">{m.feature_names.join(', ')}</td>
                <td className="px-4 py-3">
                  <span className={`px-2 py-0.5 rounded text-xs ${m.is_active ? 'bg-green-900 text-green-400' : 'bg-gray-800 text-gray-400'}`}>
                    {m.is_active ? 'active' : 'inactive'}
                  </span>
                </td>
                <td className="px-4 py-3 text-gray-400">{new Date(m.created_at).toLocaleDateString()}</td>
                <td className="px-4 py-3">
                  <Link to={`/models/${m.id}`} className="text-blue-400 hover:text-blue-300">View →</Link>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}