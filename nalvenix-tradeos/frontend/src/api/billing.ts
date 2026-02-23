import { api } from './client'

export interface BillingData {
  subscription: {
    plan: string
    status: string
    expires_at: string | null
  }
  usage: {
    api_calls: number
    api_limit: number
    strategies_used: number
    strategies_limit: number
  }
  invoices: {
    id: string
    date: string
    amount: number
    status: string
  }[]
}

export const billingApi = {
  getBilling: async (): Promise<BillingData> => {
    const response = await api.get('/api/billing')
    return response.data
  },
}
