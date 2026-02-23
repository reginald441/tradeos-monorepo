export interface User {
  id: string;
  email: string;
  name: string;
  avatar?: string;
  role: 'admin' | 'trader' | 'viewer';
  createdAt: string;
  lastLogin: string;
}

export interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
}

export interface Position {
  id: string;
  symbol: string;
  side: 'long' | 'short';
  size: number;
  entryPrice: number;
  markPrice: number;
  liquidationPrice?: number;
  margin: number;
  leverage: number;
  unrealizedPnl: number;
  unrealizedPnlPercent: number;
  openedAt: string;
}

export interface Trade {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  size: number;
  price: number;
  value: number;
  fee: number;
  pnl?: number;
  timestamp: string;
  strategy?: string;
}

export interface Strategy {
  id: string;
  name: string;
  description: string;
  type: 'trend' | 'mean_reversion' | 'arbitrage' | 'market_making';
  status: 'active' | 'paused' | 'stopped';
  config: StrategyConfig;
  performance: StrategyPerformance;
  createdAt: string;
  updatedAt: string;
}

export interface StrategyConfig {
  symbol: string;
  timeframe: string;
  positionSize: number;
  maxPositions: number;
  stopLoss?: number;
  takeProfit?: number;
  leverage: number;
  parameters: Record<string, number | string | boolean>;
}

export interface StrategyPerformance {
  totalReturn: number;
  sharpeRatio: number;
  maxDrawdown: number;
  winRate: number;
  totalTrades: number;
  profitFactor: number;
  avgTrade: number;
}

export interface Portfolio {
  totalValue: number;
  availableBalance: number;
  marginUsed: number;
  unrealizedPnl: number;
  realizedPnl24h: number;
  realizedPnl7d: number;
  realizedPnl30d: number;
}

export interface BacktestResult {
  id: string;
  strategyId: string;
  startDate: string;
  endDate: string;
  initialCapital: number;
  finalCapital: number;
  totalReturn: number;
  sharpeRatio: number;
  maxDrawdown: number;
  winRate: number;
  totalTrades: number;
  profitFactor: number;
  equityCurve: EquityPoint[];
  trades: Trade[];
  metrics: BacktestMetrics;
}

export interface EquityPoint {
  timestamp: string;
  value: number;
  drawdown: number;
}

export interface BacktestMetrics {
  avgTrade: number;
  avgWin: number;
  avgLoss: number;
  largestWin: number;
  largestLoss: number;
  avgTradeDuration: number;
  maxConsecutiveWins: number;
  maxConsecutiveLosses: number;
}

export interface RiskMetrics {
  totalExposure: number;
  marginUtilization: number;
  var95: number;
  var99: number;
  expectedShortfall: number;
  beta: number;
  correlationMatrix: Record<string, Record<string, number>>;
}

export interface RiskLimit {
  type: 'position_size' | 'daily_loss' | 'total_exposure' | 'leverage';
  value: number;
  enabled: boolean;
}

export interface OrderBook {
  symbol: string;
  bids: [number, number][];
  asks: [number, number][];
  timestamp: string;
}

export interface Candle {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface Subscription {
  id: string;
  plan: 'free' | 'pro' | 'enterprise';
  status: 'active' | 'cancelled' | 'past_due';
  currentPeriodStart: string;
  currentPeriodEnd: string;
  cancelAtPeriodEnd: boolean;
  features: string[];
}

export interface Invoice {
  id: string;
  amount: number;
  currency: string;
  status: 'paid' | 'pending' | 'failed';
  description: string;
  createdAt: string;
  paidAt?: string;
  pdfUrl?: string;
}

export interface ApiKey {
  id: string;
  name: string;
  key: string;
  permissions: string[];
  createdAt: string;
  lastUsedAt?: string;
  expiresAt?: string;
}
