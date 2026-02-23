import { create } from 'zustand';
import type { 
  Position, 
  Trade, 
  Strategy, 
  Portfolio, 
  BacktestResult,
  RiskMetrics,
  Candle,
  OrderBook 
} from '@/types';

interface TradingState {
  // Portfolio
  portfolio: Portfolio | null;
  
  // Positions & Trades
  positions: Position[];
  trades: Trade[];
  
  // Strategies
  strategies: Strategy[];
  
  // Market Data
  candles: Record<string, Candle[]>;
  orderBook: OrderBook | null;
  selectedSymbol: string;
  
  // Backtest
  backtestResults: BacktestResult[];
  currentBacktest: BacktestResult | null;
  
  // Risk
  riskMetrics: RiskMetrics | null;
  
  // UI State
  isLoading: boolean;
  error: string | null;
}

interface TradingStore extends TradingState {
  // Portfolio actions
  setPortfolio: (portfolio: Portfolio) => void;
  updatePortfolio: (updates: Partial<Portfolio>) => void;
  
  // Position actions
  setPositions: (positions: Position[]) => void;
  addPosition: (position: Position) => void;
  updatePosition: (id: string, updates: Partial<Position>) => void;
  removePosition: (id: string) => void;
  
  // Trade actions
  setTrades: (trades: Trade[]) => void;
  addTrade: (trade: Trade) => void;
  
  // Strategy actions
  setStrategies: (strategies: Strategy[]) => void;
  addStrategy: (strategy: Strategy) => void;
  updateStrategy: (id: string, updates: Partial<Strategy>) => void;
  removeStrategy: (id: string) => void;
  toggleStrategy: (id: string) => void;
  
  // Market data actions
  setCandles: (symbol: string, candles: Candle[]) => void;
  addCandle: (symbol: string, candle: Candle) => void;
  setOrderBook: (orderBook: OrderBook) => void;
  setSelectedSymbol: (symbol: string) => void;
  
  // Backtest actions
  setBacktestResults: (results: BacktestResult[]) => void;
  addBacktestResult: (result: BacktestResult) => void;
  setCurrentBacktest: (result: BacktestResult | null) => void;
  
  // Risk actions
  setRiskMetrics: (metrics: RiskMetrics) => void;
  
  // UI actions
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
}

// Mock data generators
const generateMockPortfolio = (): Portfolio => ({
  totalValue: 125430.50,
  availableBalance: 45670.25,
  marginUsed: 79760.25,
  unrealizedPnl: 3245.80,
  realizedPnl24h: 1250.30,
  realizedPnl7d: 8750.60,
  realizedPnl30d: 28450.90,
});

const generateMockPositions = (): Position[] => [
  {
    id: 'pos-1',
    symbol: 'BTC-USD',
    side: 'long',
    size: 0.5,
    entryPrice: 43250.00,
    markPrice: 45120.50,
    liquidationPrice: 35000.00,
    margin: 10000,
    leverage: 2,
    unrealizedPnl: 935.25,
    unrealizedPnlPercent: 4.32,
    openedAt: '2024-01-15T10:30:00Z',
  },
  {
    id: 'pos-2',
    symbol: 'ETH-USD',
    side: 'long',
    size: 5,
    entryPrice: 2580.00,
    markPrice: 2645.30,
    liquidationPrice: 2000.00,
    margin: 5000,
    leverage: 2.5,
    unrealizedPnl: 326.50,
    unrealizedPnlPercent: 2.53,
    openedAt: '2024-01-16T14:20:00Z',
  },
  {
    id: 'pos-3',
    symbol: 'SOL-USD',
    side: 'short',
    size: 100,
    entryPrice: 98.50,
    markPrice: 95.20,
    liquidationPrice: 130.00,
    margin: 3000,
    leverage: 3,
    unrealizedPnl: 330.00,
    unrealizedPnlPercent: 3.35,
    openedAt: '2024-01-17T09:15:00Z',
  },
];

const generateMockTrades = (): Trade[] => [
  {
    id: 'trade-1',
    symbol: 'BTC-USD',
    side: 'buy',
    size: 0.25,
    price: 42800.00,
    value: 10700.00,
    fee: 10.70,
    pnl: 245.50,
    timestamp: '2024-01-17T16:30:00Z',
    strategy: 'Trend Follower',
  },
  {
    id: 'trade-2',
    symbol: 'ETH-USD',
    side: 'sell',
    size: 2,
    price: 2620.00,
    value: 5240.00,
    fee: 5.24,
    pnl: 80.00,
    timestamp: '2024-01-17T15:45:00Z',
    strategy: 'Mean Reversion',
  },
  {
    id: 'trade-3',
    symbol: 'SOL-USD',
    side: 'buy',
    size: 50,
    price: 94.80,
    value: 4740.00,
    fee: 4.74,
    pnl: 20.00,
    timestamp: '2024-01-17T14:20:00Z',
    strategy: 'Breakout',
  },
];

const generateMockStrategies = (): Strategy[] => [
  {
    id: 'strat-1',
    name: 'Trend Follower',
    description: 'Follows established trends using moving average crossovers',
    type: 'trend',
    status: 'active',
    config: {
      symbol: 'BTC-USD',
      timeframe: '1h',
      positionSize: 1000,
      maxPositions: 3,
      stopLoss: 2,
      takeProfit: 6,
      leverage: 2,
      parameters: {
        fastMA: 20,
        slowMA: 50,
        atrPeriod: 14,
      },
    },
    performance: {
      totalReturn: 45.2,
      sharpeRatio: 1.85,
      maxDrawdown: -12.5,
      winRate: 58.3,
      totalTrades: 156,
      profitFactor: 1.72,
      avgTrade: 2.89,
    },
    createdAt: '2024-01-01T00:00:00Z',
    updatedAt: '2024-01-17T12:00:00Z',
  },
  {
    id: 'strat-2',
    name: 'Mean Reversion',
    description: 'Trades price reversions to the mean using Bollinger Bands',
    type: 'mean_reversion',
    status: 'active',
    config: {
      symbol: 'ETH-USD',
      timeframe: '15m',
      positionSize: 500,
      maxPositions: 5,
      stopLoss: 1.5,
      takeProfit: 3,
      leverage: 2.5,
      parameters: {
        bbPeriod: 20,
        bbStdDev: 2,
        rsiPeriod: 14,
      },
    },
    performance: {
      totalReturn: 32.8,
      sharpeRatio: 1.52,
      maxDrawdown: -8.3,
      winRate: 62.1,
      totalTrades: 234,
      profitFactor: 1.58,
      avgTrade: 1.40,
    },
    createdAt: '2024-01-05T00:00:00Z',
    updatedAt: '2024-01-17T10:30:00Z',
  },
  {
    id: 'strat-3',
    name: 'Breakout Hunter',
    description: 'Identifies and trades breakouts from consolidation patterns',
    type: 'trend',
    status: 'paused',
    config: {
      symbol: 'SOL-USD',
      timeframe: '30m',
      positionSize: 750,
      maxPositions: 2,
      stopLoss: 3,
      takeProfit: 9,
      leverage: 3,
      parameters: {
        lookbackPeriod: 20,
        volumeThreshold: 1.5,
        atrMultiplier: 1.5,
      },
    },
    performance: {
      totalReturn: 28.5,
      sharpeRatio: 1.28,
      maxDrawdown: -15.2,
      winRate: 45.8,
      totalTrades: 89,
      profitFactor: 1.85,
      avgTrade: 3.20,
    },
    createdAt: '2024-01-10T00:00:00Z',
    updatedAt: '2024-01-16T18:00:00Z',
  },
];

const generateMockCandles = (symbol: string): Candle[] => {
  const candles: Candle[] = [];
  const basePrice = symbol.includes('BTC') ? 45000 : symbol.includes('ETH') ? 2600 : 95;
  let price = basePrice;
  
  const now = new Date();
  for (let i = 100; i >= 0; i--) {
    const timestamp = new Date(now.getTime() - i * 3600000).toISOString();
    const volatility = basePrice * 0.02;
    const change = (Math.random() - 0.48) * volatility;
    
    const open = price;
    const close = price + change;
    const high = Math.max(open, close) + Math.random() * volatility * 0.3;
    const low = Math.min(open, close) - Math.random() * volatility * 0.3;
    const volume = Math.random() * 1000 + 500;
    
    candles.push({
      timestamp,
      open,
      high,
      low,
      close,
      volume,
    });
    
    price = close;
  }
  
  return candles;
};

const generateMockRiskMetrics = (): RiskMetrics => ({
  totalExposure: 180000,
  marginUtilization: 63.5,
  var95: -8500,
  var99: -14200,
  expectedShortfall: -18500,
  beta: 0.85,
  correlationMatrix: {
    'BTC-USD': { 'BTC-USD': 1, 'ETH-USD': 0.85, 'SOL-USD': 0.72 },
    'ETH-USD': { 'BTC-USD': 0.85, 'ETH-USD': 1, 'SOL-USD': 0.78 },
    'SOL-USD': { 'BTC-USD': 0.72, 'ETH-USD': 0.78, 'SOL-USD': 1 },
  },
});

export const useTradingStore = create<TradingStore>((set, get) => ({
  // Initial state
  portfolio: generateMockPortfolio(),
  positions: generateMockPositions(),
  trades: generateMockTrades(),
  strategies: generateMockStrategies(),
  candles: {
    'BTC-USD': generateMockCandles('BTC-USD'),
    'ETH-USD': generateMockCandles('ETH-USD'),
    'SOL-USD': generateMockCandles('SOL-USD'),
  },
  orderBook: null,
  selectedSymbol: 'BTC-USD',
  backtestResults: [],
  currentBacktest: null,
  riskMetrics: generateMockRiskMetrics(),
  isLoading: false,
  error: null,

  // Portfolio actions
  setPortfolio: (portfolio) => set({ portfolio }),
  updatePortfolio: (updates) => {
    const { portfolio } = get();
    if (portfolio) {
      set({ portfolio: { ...portfolio, ...updates } });
    }
  },

  // Position actions
  setPositions: (positions) => set({ positions }),
  addPosition: (position) => {
    const { positions } = get();
    set({ positions: [position, ...positions] });
  },
  updatePosition: (id, updates) => {
    const { positions } = get();
    set({
      positions: positions.map(p => p.id === id ? { ...p, ...updates } : p),
    });
  },
  removePosition: (id) => {
    const { positions } = get();
    set({ positions: positions.filter(p => p.id !== id) });
  },

  // Trade actions
  setTrades: (trades) => set({ trades }),
  addTrade: (trade) => {
    const { trades } = get();
    set({ trades: [trade, ...trades] });
  },

  // Strategy actions
  setStrategies: (strategies) => set({ strategies }),
  addStrategy: (strategy) => {
    const { strategies } = get();
    set({ strategies: [...strategies, strategy] });
  },
  updateStrategy: (id, updates) => {
    const { strategies } = get();
    set({
      strategies: strategies.map(s => s.id === id ? { ...s, ...updates } : s),
    });
  },
  removeStrategy: (id) => {
    const { strategies } = get();
    set({ strategies: strategies.filter(s => s.id !== id) });
  },
  toggleStrategy: (id) => {
    const { strategies } = get();
    set({
      strategies: strategies.map(s => 
        s.id === id 
          ? { ...s, status: s.status === 'active' ? 'paused' : 'active' as const }
          : s
      ),
    });
  },

  // Market data actions
  setCandles: (symbol, candles) => {
    set((state) => ({
      candles: { ...state.candles, [symbol]: candles },
    }));
  },
  addCandle: (symbol, candle) => {
    set((state) => ({
      candles: {
        ...state.candles,
        [symbol]: [...(state.candles[symbol] || []), candle].slice(-500),
      },
    }));
  },
  setOrderBook: (orderBook) => set({ orderBook }),
  setSelectedSymbol: (symbol) => set({ selectedSymbol: symbol }),

  // Backtest actions
  setBacktestResults: (results) => set({ backtestResults: results }),
  addBacktestResult: (result) => {
    const { backtestResults } = get();
    set({ backtestResults: [result, ...backtestResults] });
  },
  setCurrentBacktest: (result) => set({ currentBacktest: result }),

  // Risk actions
  setRiskMetrics: (metrics) => set({ riskMetrics: metrics }),

  // UI actions
  setLoading: (loading) => set({ isLoading: loading }),
  setError: (error) => set({ error }),
}));
