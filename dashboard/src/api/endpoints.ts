import api from './client'
import type { MLModel, Run, Alert, AlertSummary } from './types'

export const modelsApi = {
  list: () => api.get<MLModel[]>('/api/v1/models'),
  get: (id: string) => api.get<MLModel>(`/api/v1/models/${id}`),
}

export const runsApi = {
  list: (modelId: string) => api.get<Run[]>(`/api/v1/models/${modelId}/runs`),
  get: (runId: string) => api.get<Run>(`/api/v1/runs/${runId}`),
}

export const alertsApi = {
  list: (params?: { model_id?: string; severity?: string; acknowledged?: boolean }) =>
    api.get<Alert[]>('/api/v1/alerts', { params }),
  acknowledge: (alertId: string) =>
    api.patch<Alert>(`/api/v1/alerts/${alertId}/acknowledge`, { acknowledged: true }),
  summary: (modelId: string) =>
    api.get<AlertSummary>(`/api/v1/alerts/model/${modelId}/summary`),
}