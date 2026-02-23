import { api } from './client'

export interface Strategy {
  id: string
  name: string
  type: string
  config: Record<string, any>
  is_active: boolean
  performance: number
  created_at: string
}

export interface CreateStrategyData {
  name: string
  type: string
  config: Record<string, any>
}

export const strategiesApi = {
  getStrategies: async (): Promise<{ strategies: Strategy[] }> => {
    const response = await api.get('/api/strategies')
    return response.data
  },
  
  createStrategy: async (data: CreateStrategyData) => {
    const response = await api.post('/api/strategies', data)
    return response.data
  },
  
  toggleStrategy: async (strategyId: string) => {
    const response = await api.post(`/api/strategies/${strategyId}/toggle`)
    return response.data
  },
  
  deleteStrategy: async (strategyId: string) => {
    const response = await api.delete(`/api/strategies/${strategyId}`)
    return response.data
  },
}
