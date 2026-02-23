import { api } from './client'

export interface Trade {
  id: string
  symbol: string
  side: 'buy' | 'sell'
  quantity: number
  entry_price: number
  exit_price?: number
  pnl?: number
  status: 'open' | 'closed'
  opened_at: string
  closed_at?: string
}

export interface CreateTradeData {
  symbol: string
  side: string
  quantity: number
  order_type?: string
  price?: number
}

export const tradesApi = {
  getTrades: async (status?: string): Promise<{ trades: Trade[] }> => {
    const params = status ? { status } : {}
    const response = await api.get('/api/trades', { params })
    return response.data
  },
  
  createTrade: async (data: CreateTradeData) => {
    const response = await api.post('/api/trades', data)
    return response.data
  },
  
  closeTrade: async (tradeId: string) => {
    const response = await api.post(`/api/trades/${tradeId}/close`)
    return response.data
  },
}
