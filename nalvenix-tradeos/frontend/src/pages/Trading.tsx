import { useEffect, useState } from 'react'
import { tradesApi, Trade, CreateTradeData } from '../api/trades'
import PriceChart from '../components/PriceChart'
import { 
  ArrowUpRight, 
  ArrowDownRight, 
  Plus, 
  X,
  Loader2,
  TrendingUp,
  TrendingDown
} from 'lucide-react'

const SYMBOLS = [
  'BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'DOT-USD',
  'EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD',
  'XAU/USD', 'XAG/USD', 'WTI/USD',
  'SPX500', 'NAS100', 'DJ30'
]

export default function Trading() {
  const [trades, setTrades] = useState<Trade[]>([])
  const [loading, setLoading] = useState(true)
  const [showModal, setShowModal] = useState(false)
  const [selectedSymbol, setSelectedSymbol] = useState('BTC-USD')
  const [formData, setFormData] = useState<CreateTradeData>({
    symbol: 'BTC-USD',
    side: 'buy',
    quantity: 0.1,
    order_type: 'market'
  })
  const [submitting, setSubmitting] = useState(false)

  const fetchTrades = async () => {
    try {
      const response = await tradesApi.getTrades()
      setTrades(response.trades)
    } catch (error) {
      console.error('Failed to fetch trades:', error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchTrades()
  }, [])

  const handleCreateTrade = async (e: React.FormEvent) => {
    e.preventDefault()
    setSubmitting(true)
    try {
      await tradesApi.createTrade(formData)
      setShowModal(false)
      fetchTrades()
    } catch (error) {
      console.error('Failed to create trade:', error)
    } finally {
      setSubmitting(false)
    }
  }

  const handleCloseTrade = async (tradeId: string) => {
    try {
      await tradesApi.closeTrade(tradeId)
      fetchTrades()
    } catch (error) {
      console.error('Failed to close trade:', error)
    }
  }

  const openTrades = trades.filter(t => t.status === 'open')
  const closedTrades = trades.filter(t => t.status === 'closed')

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Trading</h1>
          <p className="text-slate-400">Manage your positions and execute trades</p>
        </div>
        <button
          onClick={() => setShowModal(true)}
          className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-cyan-500 to-violet-600 text-white rounded-lg hover:from-cyan-400 hover:to-violet-500 transition-all"
        >
          <Plus className="w-5 h-5" />
          New Trade
        </button>
      </div>

      {/* Chart */}
      <div className="glass-card rounded-xl p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-white">Price Chart</h2>
          <select
            value={selectedSymbol}
            onChange={(e) => setSelectedSymbol(e.target.value)}
            className="bg-slate-800/50 border border-white/10 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:border-cyan-500/50"
          >
            {SYMBOLS.map(s => <option key={s} value={s}>{s}</option>)}
          </select>
        </div>
        <PriceChart symbol={selectedSymbol} height={300} />
      </div>

      {/* Open Positions */}
      <div className="glass-card rounded-xl overflow-hidden">
        <div className="px-6 py-4 border-b border-white/5 flex items-center justify-between">
          <h3 className="text-lg font-semibold text-white flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-emerald-400" />
            Open Positions ({openTrades.length})
          </h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="text-left text-xs text-slate-400 border-b border-white/5">
                <th className="px-6 py-3 font-medium">Symbol</th>
                <th className="px-6 py-3 font-medium">Side</th>
                <th className="px-6 py-3 font-medium text-right">Quantity</th>
                <th className="px-6 py-3 font-medium text-right">Entry Price</th>
                <th className="px-6 py-3 font-medium text-right">P&L</th>
                <th className="px-6 py-3 font-medium text-center">Action</th>
              </tr>
            </thead>
            <tbody>
              {openTrades.length === 0 ? (
                <tr>
                  <td colSpan={6} className="px-6 py-8 text-center text-slate-400">
                    No open positions
                  </td>
                </tr>
              ) : (
                openTrades.map((trade) => (
                  <tr key={trade.id} className="border-b border-white/5 last:border-0 hover:bg-white/5">
                    <td className="px-6 py-4 font-mono text-white">{trade.symbol}</td>
                    <td className="px-6 py-4">
                      <span className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${
                        trade.side === 'buy' ? 'bg-emerald-500/20 text-emerald-400' : 'bg-red-500/20 text-red-400'
                      }`}>
                        {trade.side === 'buy' ? <ArrowUpRight className="w-3 h-3" /> : <ArrowDownRight className="w-3 h-3" />}
                        {trade.side.toUpperCase()}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-right font-mono text-white">{trade.quantity}</td>
                    <td className="px-6 py-4 text-right font-mono text-white">${trade.entry_price.toFixed(2)}</td>
                    <td className="px-6 py-4 text-right font-mono text-slate-400">-</td>
                    <td className="px-6 py-4 text-center">
                      <button
                        onClick={() => handleCloseTrade(trade.id)}
                        className="px-3 py-1 bg-red-500/20 text-red-400 rounded-lg text-sm hover:bg-red-500/30 transition-colors"
                      >
                        Close
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Trade History */}
      <div className="glass-card rounded-xl overflow-hidden">
        <div className="px-6 py-4 border-b border-white/5">
          <h3 className="text-lg font-semibold text-white flex items-center gap-2">
            <TrendingDown className="w-5 h-5 text-slate-400" />
            Trade History ({closedTrades.length})
          </h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="text-left text-xs text-slate-400 border-b border-white/5">
                <th className="px-6 py-3 font-medium">Symbol</th>
                <th className="px-6 py-3 font-medium">Side</th>
                <th className="px-6 py-3 font-medium text-right">Quantity</th>
                <th className="px-6 py-3 font-medium text-right">Entry</th>
                <th className="px-6 py-3 font-medium text-right">Exit</th>
                <th className="px-6 py-3 font-medium text-right">P&L</th>
              </tr>
            </thead>
            <tbody>
              {closedTrades.length === 0 ? (
                <tr>
                  <td colSpan={6} className="px-6 py-8 text-center text-slate-400">
                    No closed trades
                  </td>
                </tr>
              ) : (
                closedTrades.map((trade) => (
                  <tr key={trade.id} className="border-b border-white/5 last:border-0 hover:bg-white/5">
                    <td className="px-6 py-4 font-mono text-white">{trade.symbol}</td>
                    <td className="px-6 py-4">
                      <span className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${
                        trade.side === 'buy' ? 'bg-emerald-500/20 text-emerald-400' : 'bg-red-500/20 text-red-400'
                      }`}>
                        {trade.side.toUpperCase()}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-right font-mono text-white">{trade.quantity}</td>
                    <td className="px-6 py-4 text-right font-mono text-white">${trade.entry_price.toFixed(2)}</td>
                    <td className="px-6 py-4 text-right font-mono text-white">${trade.exit_price?.toFixed(2)}</td>
                    <td className="px-6 py-4 text-right font-mono">
                      <span className={(trade.pnl || 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}>
                        {(trade.pnl || 0) >= 0 ? '+' : ''}${trade.pnl?.toFixed(2)}
                      </span>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* New Trade Modal */}
      {showModal && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="glass-card rounded-xl p-6 w-full max-w-md">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-semibold text-white">New Trade</h3>
              <button onClick={() => setShowModal(false)} className="text-slate-400 hover:text-white">
                <X className="w-5 h-5" />
              </button>
            </div>

            <form onSubmit={handleCreateTrade} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-1">Symbol</label>
                <select
                  value={formData.symbol}
                  onChange={(e) => setFormData({ ...formData, symbol: e.target.value })}
                  className="w-full bg-slate-800/50 border border-white/10 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-cyan-500/50"
                >
                  {SYMBOLS.map(s => <option key={s} value={s}>{s}</option>)}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-300 mb-1">Side</label>
                <div className="flex gap-2">
                  <button
                    type="button"
                    onClick={() => setFormData({ ...formData, side: 'buy' })}
                    className={`flex-1 py-3 rounded-lg font-medium transition-all ${
                      formData.side === 'buy'
                        ? 'bg-emerald-500 text-white'
                        : 'bg-slate-800/50 text-slate-400 hover:text-white'
                    }`}
                  >
                    BUY
                  </button>
                  <button
                    type="button"
                    onClick={() => setFormData({ ...formData, side: 'sell' })}
                    className={`flex-1 py-3 rounded-lg font-medium transition-all ${
                      formData.side === 'sell'
                        ? 'bg-red-500 text-white'
                        : 'bg-slate-800/50 text-slate-400 hover:text-white'
                    }`}
                  >
                    SELL
                  </button>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-300 mb-1">Quantity</label>
                <input
                  type="number"
                  step="0.01"
                  value={formData.quantity}
                  onChange={(e) => setFormData({ ...formData, quantity: parseFloat(e.target.value) })}
                  className="w-full bg-slate-800/50 border border-white/10 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-cyan-500/50"
                />
              </div>

              <button
                type="submit"
                disabled={submitting}
                className="w-full bg-gradient-to-r from-cyan-500 to-violet-600 text-white font-semibold py-3 rounded-lg hover:from-cyan-400 hover:to-violet-500 transition-all disabled:opacity-50 flex items-center justify-center gap-2"
              >
                {submitting && <Loader2 className="w-5 h-5 animate-spin" />}
                Execute Trade
              </button>
            </form>
          </div>
        </div>
      )}
    </div>
  )
}
