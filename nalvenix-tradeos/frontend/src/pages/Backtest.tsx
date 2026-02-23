import { useState } from 'react'
import { 
  Play, 
  Calendar, 
  Settings, 
  TrendingUp, 
  TrendingDown,
  Target,
  Activity,
  BarChart3,
  Check
} from 'lucide-react'

const SYMBOLS = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'EUR/USD', 'XAU/USD', 'SPX500']
const STRATEGIES = ['Momentum', 'Mean Reversion', 'Trend Following', 'Breakout', 'Scalping']
const TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h', '1d']

interface BacktestResult {
  totalReturn: number
  sharpeRatio: number
  maxDrawdown: number
  winRate: number
  totalTrades: number
  profitFactor: number
  equity: { date: string; value: number }[]
  trades: { date: string; pnl: number; type: 'win' | 'loss' }[]
}

export default function Backtest() {
  const [running, setRunning] = useState(false)
  const [completed, setCompleted] = useState(false)
  const [result, setResult] = useState<BacktestResult | null>(null)
  const [config, setConfig] = useState({
    symbol: 'BTC-USD',
    strategy: 'Momentum',
    timeframe: '1h',
    startDate: '2023-01-01',
    endDate: '2024-01-01',
    initialCapital: 100000
  })

  const runBacktest = async () => {
    setRunning(true)
    setCompleted(false)
    
    // Simulate backtest execution
    await new Promise(resolve => setTimeout(resolve, 3000))
    
    // Generate mock results
    setResult({
      totalReturn: 45.8,
      sharpeRatio: 1.85,
      maxDrawdown: -12.5,
      winRate: 62.5,
      totalTrades: 156,
      profitFactor: 1.75,
      equity: Array.from({ length: 50 }, (_, i) => ({
        date: new Date(2023, 0, i * 7).toISOString().split('T')[0],
        value: 100000 + Math.random() * 50000 * (i / 50)
      })),
      trades: Array.from({ length: 20 }, (_, i) => ({
        date: new Date(2023, Math.floor(Math.random() * 12), Math.floor(Math.random() * 28) + 1).toISOString().split('T')[0],
        pnl: (Math.random() - 0.4) * 2000,
        type: Math.random() > 0.4 ? 'win' : 'loss'
      }))
    })
    
    setRunning(false)
    setCompleted(true)
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-white">Backtest</h1>
        <p className="text-slate-400">Test your strategies against historical data</p>
      </div>

      {/* Configuration */}
      <div className="glass-card rounded-xl p-6">
        <div className="flex items-center gap-2 mb-6">
          <Settings className="w-5 h-5 text-cyan-400" />
          <h2 className="text-lg font-semibold text-white">Configuration</h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">Symbol</label>
            <select
              value={config.symbol}
              onChange={(e) => setConfig({ ...config, symbol: e.target.value })}
              className="w-full bg-slate-800/50 border border-white/10 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-cyan-500/50"
            >
              {SYMBOLS.map(s => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">Strategy</label>
            <select
              value={config.strategy}
              onChange={(e) => setConfig({ ...config, strategy: e.target.value })}
              className="w-full bg-slate-800/50 border border-white/10 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-cyan-500/50"
            >
              {STRATEGIES.map(s => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">Timeframe</label>
            <select
              value={config.timeframe}
              onChange={(e) => setConfig({ ...config, timeframe: e.target.value })}
              className="w-full bg-slate-800/50 border border-white/10 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-cyan-500/50"
            >
              {TIMEFRAMES.map(t => <option key={t} value={t}>{t}</option>)}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">Start Date</label>
            <div className="relative">
              <Calendar className="w-5 h-5 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
              <input
                type="date"
                value={config.startDate}
                onChange={(e) => setConfig({ ...config, startDate: e.target.value })}
                className="w-full bg-slate-800/50 border border-white/10 rounded-lg pl-10 pr-4 py-3 text-white focus:outline-none focus:border-cyan-500/50"
              />
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">End Date</label>
            <div className="relative">
              <Calendar className="w-5 h-5 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
              <input
                type="date"
                value={config.endDate}
                onChange={(e) => setConfig({ ...config, endDate: e.target.value })}
                className="w-full bg-slate-800/50 border border-white/10 rounded-lg pl-10 pr-4 py-3 text-white focus:outline-none focus:border-cyan-500/50"
              />
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">Initial Capital</label>
            <input
              type="number"
              value={config.initialCapital}
              onChange={(e) => setConfig({ ...config, initialCapital: parseInt(e.target.value) })}
              className="w-full bg-slate-800/50 border border-white/10 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-cyan-500/50"
            />
          </div>
        </div>

        <button
          onClick={runBacktest}
          disabled={running}
          className="mt-6 flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-cyan-500 to-violet-600 text-white rounded-lg hover:from-cyan-400 hover:to-violet-500 transition-all disabled:opacity-50"
        >
          {running ? (
            <>
              <div className="animate-spin w-5 h-5 border-2 border-white border-t-transparent rounded-full"></div>
              Running Backtest...
            </>
          ) : (
            <>
              <Play className="w-5 h-5" />
              Run Backtest
            </>
          )}
        </button>
      </div>

      {/* Results */}
      {completed && result && (
        <div className="space-y-6 animate-slide-in">
          <div className="flex items-center gap-2">
            <Check className="w-5 h-5 text-emerald-400" />
            <h2 className="text-lg font-semibold text-white">Backtest Results</h2>
          </div>

          {/* Metrics Grid */}
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
            <div className="glass-card rounded-xl p-4 text-center">
              <TrendingUp className="w-5 h-5 text-emerald-400 mx-auto mb-2" />
              <p className="text-2xl font-bold text-emerald-400 font-mono">+{result.totalReturn}%</p>
              <p className="text-xs text-slate-400">Total Return</p>
            </div>
            <div className="glass-card rounded-xl p-4 text-center">
              <Activity className="w-5 h-5 text-cyan-400 mx-auto mb-2" />
              <p className="text-2xl font-bold text-cyan-400 font-mono">{result.sharpeRatio}</p>
              <p className="text-xs text-slate-400">Sharpe Ratio</p>
            </div>
            <div className="glass-card rounded-xl p-4 text-center">
              <TrendingDown className="w-5 h-5 text-red-400 mx-auto mb-2" />
              <p className="text-2xl font-bold text-red-400 font-mono">{result.maxDrawdown}%</p>
              <p className="text-xs text-slate-400">Max Drawdown</p>
            </div>
            <div className="glass-card rounded-xl p-4 text-center">
              <Target className="w-5 h-5 text-violet-400 mx-auto mb-2" />
              <p className="text-2xl font-bold text-violet-400 font-mono">{result.winRate}%</p>
              <p className="text-xs text-slate-400">Win Rate</p>
            </div>
            <div className="glass-card rounded-xl p-4 text-center">
              <BarChart3 className="w-5 h-5 text-orange-400 mx-auto mb-2" />
              <p className="text-2xl font-bold text-orange-400 font-mono">{result.totalTrades}</p>
              <p className="text-xs text-slate-400">Total Trades</p>
            </div>
            <div className="glass-card rounded-xl p-4 text-center">
              <TrendingUp className="w-5 h-5 text-blue-400 mx-auto mb-2" />
              <p className="text-2xl font-bold text-blue-400 font-mono">{result.profitFactor}</p>
              <p className="text-xs text-slate-400">Profit Factor</p>
            </div>
          </div>

          {/* Trade Distribution */}
          <div className="glass-card rounded-xl p-6">
            <h3 className="text-lg font-semibold text-white mb-4">Trade Distribution</h3>
            <div className="space-y-2">
              {result.trades.slice(0, 10).map((trade, i) => (
                <div key={i} className="flex items-center justify-between p-3 rounded-lg bg-white/5">
                  <span className="text-sm text-slate-400">{trade.date}</span>
                  <span className={`font-mono font-medium ${trade.pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                    {trade.pnl >= 0 ? '+' : ''}${trade.pnl.toFixed(2)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
