import { api } from './client'

export interface AnalyticsData {
  daily_pnl: { date: string; pnl: number }[]
  strategy_performance: { name: string; return: number }[]
  symbol_distribution: { symbol: string; count: number }[]
  metrics: {
    sharpe_ratio: number
    max_drawdown: number
    win_rate: number
    profit_factor: number
    avg_trade: number
    total_trades: number
  }
}

export const analyticsApi = {
  getAnalytics: async (): Promise<AnalyticsData> => {
    const response = await api.get('/api/analytics')
    return response.data
  },
}
