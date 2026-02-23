import { api } from './client'

export interface LoginCredentials {
  email: string
  password: string
}

export interface AuthResponse {
  access_token: string
  token_type: string
  user: {
    id: string
    email: string
    full_name: string
    is_premium?: boolean
  }
}

export const authApi = {
  login: async (credentials: LoginCredentials): Promise<AuthResponse> => {
    const response = await api.post('/api/auth/login', credentials)
    return response.data
  },
  
  register: async (data: { email: string; password: string; full_name: string }): Promise<AuthResponse> => {
    const response = await api.post('/api/auth/register', data)
    return response.data
  },
  
  getMe: async () => {
    const response = await api.get('/api/auth/me')
    return response.data
  },
}
