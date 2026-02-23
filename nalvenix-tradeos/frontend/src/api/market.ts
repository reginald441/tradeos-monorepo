import { api } from './client'

export interface CandleData {
  timestamp: string
  open: number
  high: number
  low: number
  close: number
  volume: number
}

export interface MarketData {
  symbol: string
  timeframe: string
  data: CandleData[]
}

export const marketApi = {
  getMarketData: async (symbol?: string, timeframe: string = '1h'): Promise<MarketData> => {
    const response = await api.get('/api/market-data', { params: { symbol, timeframe } })
    return response.data
  },
}
