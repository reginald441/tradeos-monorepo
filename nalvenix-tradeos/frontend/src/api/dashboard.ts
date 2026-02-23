import { api } from './client'

export interface Portfolio {
  total_equity: number
  buying_power: number
  day_pnl: number
  total_pnl: number
  day_pnl_percent: number
}

export interface DashboardData {
  portfolio: Portfolio
  stats: {
    open_positions: number
    active_strategies: number
    today_trades: number
    win_rate: number
  }
  market_overview: {
    crypto: MarketItem[]
    forex: MarketItem[]
    commodities: MarketItem[]
    indices: MarketItem[]
  }
}

export interface MarketItem {
  symbol: string
  price: number
  change_24h: number
  volume: number
}

export const dashboardApi = {
  getDashboard: async (): Promise<DashboardData> => {
    const response = await api.get('/api/dashboard')
    return response.data
  },
}
