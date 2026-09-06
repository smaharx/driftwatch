import axios from 'axios'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'https://driftwatch-production-e733.up.railway.app',
  headers: { 'Content-Type': 'application/json' },
})

export default api