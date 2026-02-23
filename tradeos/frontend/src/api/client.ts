import axios, { AxiosInstance, AxiosError, InternalAxiosRequestConfig } from 'axios';
import { useAuthStore } from '@/store/authStore';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

class ApiClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.setupInterceptors();
  }

  private setupInterceptors() {
    // Request interceptor
    this.client.interceptors.request.use(
      (config: InternalAxiosRequestConfig) => {
        const token = useAuthStore.getState().token;
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => response,
      (error: AxiosError) => {
        if (error.response?.status === 401) {
          // Handle unauthorized
          useAuthStore.getState().logout();
          window.location.href = '/login';
        }
        return Promise.reject(error);
      }
    );
  }

  // Auth endpoints
  async login(email: string, password: string) {
    const response = await this.client.post('/auth/login', { email, password });
    return response.data;
  }

  async register(name: string, email: string, password: string) {
    const response = await this.client.post('/auth/register', { name, email, password });
    return response.data;
  }

  async refreshToken() {
    const response = await this.client.post('/auth/refresh');
    return response.data;
  }

  // Portfolio endpoints
  async getPortfolio() {
    const response = await this.client.get('/portfolio');
    return response.data;
  }

  async getPositions() {
    const response = await this.client.get('/positions');
    return response.data;
  }

  async getTrades(limit?: number) {
    const response = await this.client.get('/trades', { params: { limit } });
    return response.data;
  }

  // Strategy endpoints
  async getStrategies() {
    const response = await this.client.get('/strategies');
    return response.data;
  }

  async createStrategy(strategy: any) {
    const response = await this.client.post('/strategies', strategy);
    return response.data;
  }

  async updateStrategy(id: string, updates: any) {
    const response = await this.client.patch(`/strategies/${id}`, updates);
    return response.data;
  }

  async deleteStrategy(id: string) {
    const response = await this.client.delete(`/strategies/${id}`);
    return response.data;
  }

  async toggleStrategy(id: string) {
    const response = await this.client.post(`/strategies/${id}/toggle`);
    return response.data;
  }

  // Trading endpoints
  async placeOrder(order: any) {
    const response = await this.client.post('/orders', order);
    return response.data;
  }

  async cancelOrder(orderId: string) {
    const response = await this.client.delete(`/orders/${orderId}`);
    return response.data;
  }

  async getOrderBook(symbol: string) {
    const response = await this.client.get(`/orderbook/${symbol}`);
    return response.data;
  }

  async getCandles(symbol: string, timeframe: string, limit?: number) {
    const response = await this.client.get(`/candles/${symbol}`, {
      params: { timeframe, limit },
    });
    return response.data;
  }

  // Backtest endpoints
  async runBacktest(config: any) {
    const response = await this.client.post('/backtest', config);
    return response.data;
  }

  async getBacktestResults() {
    const response = await this.client.get('/backtest');
    return response.data;
  }

  async getBacktestResult(id: string) {
    const response = await this.client.get(`/backtest/${id}`);
    return response.data;
  }

  // Risk endpoints
  async getRiskMetrics() {
    const response = await this.client.get('/risk/metrics');
    return response.data;
  }

  async getRiskLimits() {
    const response = await this.client.get('/risk/limits');
    return response.data;
  }

  async updateRiskLimit(type: string, config: any) {
    const response = await this.client.patch(`/risk/limits/${type}`, config);
    return response.data;
  }

  // User endpoints
  async getUserProfile() {
    const response = await this.client.get('/user/profile');
    return response.data;
  }

  async updateUserProfile(updates: any) {
    const response = await this.client.patch('/user/profile', updates);
    return response.data;
  }

  async changePassword(currentPassword: string, newPassword: string) {
    const response = await this.client.post('/user/change-password', {
      currentPassword,
      newPassword,
    });
    return response.data;
  }

  // API Key endpoints
  async getApiKeys() {
    const response = await this.client.get('/api-keys');
    return response.data;
  }

  async createApiKey(name: string, permissions: string[]) {
    const response = await this.client.post('/api-keys', { name, permissions });
    return response.data;
  }

  async revokeApiKey(id: string) {
    const response = await this.client.delete(`/api-keys/${id}`);
    return response.data;
  }

  // Billing endpoints
  async getSubscription() {
    const response = await this.client.get('/billing/subscription');
    return response.data;
  }

  async getInvoices() {
    const response = await this.client.get('/billing/invoices');
    return response.data;
  }

  async createCheckoutSession(plan: string) {
    const response = await this.client.post('/billing/checkout', { plan });
    return response.data;
  }

  async cancelSubscription() {
    const response = await this.client.post('/billing/cancel');
    return response.data;
  }

  // WebSocket connection (for real-time data)
  connectWebSocket(onMessage: (data: any) => void) {
    const wsUrl = API_BASE_URL.replace('http', 'ws') + '/ws';
    const token = useAuthStore.getState().token;
    const ws = new WebSocket(`${wsUrl}?token=${token}`);

    ws.onopen = () => {
      console.log('WebSocket connected');
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      onMessage(data);
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
    };

    return ws;
  }

  // Generic request methods
  get<T>(url: string, config?: any) {
    return this.client.get<T>(url, config);
  }

  post<T>(url: string, data?: any, config?: any) {
    return this.client.post<T>(url, data, config);
  }

  patch<T>(url: string, data?: any, config?: any) {
    return this.client.patch<T>(url, data, config);
  }

  delete<T>(url: string, config?: any) {
    return this.client.delete<T>(url, config);
  }
}

export const apiClient = new ApiClient();
export default apiClient;
