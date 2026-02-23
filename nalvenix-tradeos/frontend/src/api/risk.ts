import { api } from './client'

export interface RiskSettings {
  max_position_size: number
  max_daily_loss: number
  var_limit: number
  kill_switch_enabled: boolean
}

export const riskApi = {
  getRiskSettings: async (): Promise<RiskSettings> => {
    const response = await api.get('/api/risk')
    return response.data
  },
  
  updateRiskSettings: async (settings: RiskSettings) => {
    const response = await api.put('/api/risk', settings)
    return response.data
  },
}
