export interface MLModel {
  id: string
  name: string
  description: string | null
  model_type: string
  feature_names: string[]
  is_active: boolean
  created_at: string
  updated_at: string
}

export interface Run {
  id: string
  model_id: string
  status: string
  sample_size: number
  overall_drift_score: number | null
  drifted_features: string[] | null
  drift_results: Record<string, any> | null
  error_message: string | null
  created_at: string
  completed_at: string | null
}

export interface Alert {
  id: string
  run_id: string
  model_id: string
  feature_name: string
  detector_type: string
  drift_score: number
  threshold: number
  severity: 'low' | 'medium' | 'high'
  acknowledged: boolean
  notification_sent: boolean
  created_at: string
}

export interface AlertSummary {
  model_id: string
  model_name: string
  total: number
  unacknowledged: number
  by_severity: {
    high: number
    medium: number
    low: number
  }
}