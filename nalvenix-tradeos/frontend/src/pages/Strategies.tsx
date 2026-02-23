import { useEffect, useState } from 'react'
import { strategiesApi, Strategy, CreateStrategyData } from '../api/strategies'
import { 
  Play, 
  Pause, 
  Trash2, 
  Plus, 
  Cpu, 
  TrendingUp, 
  Activity,
  X,
  Loader2,
  Check
} from 'lucide-react'

const STRATEGY_TYPES = [
  { value: 'trend_following', label: 'Trend Following', icon: TrendingUp },
  { value: 'mean_reversion', label: 'Mean Reversion', icon: Activity },
  { value: 'arbitrage', label: 'Arbitrage', icon: Cpu },
  { value: 'momentum', label: 'Momentum', icon: TrendingUp },
  { value: 'volatility', label: 'Volatility', icon: Activity },
]

export default function Strategies() {
  const [strategies, setStrategies] = useState<Strategy[]>([])
  const [loading, setLoading] = useState(true)
  const [showModal, setShowModal] = useState(false)
  const [formData, setFormData] = useState<CreateStrategyData>({
    name: '',
    type: 'trend_following',
    config: {}
  })
  const [submitting, setSubmitting] = useState(false)

  const fetchStrategies = async () => {
    try {
      const response = await strategiesApi.getStrategies()
      setStrategies(response.strategies)
    } catch (error) {
      console.error('Failed to fetch strategies:', error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchStrategies()
  }, [])

  const handleToggle = async (strategyId: string) => {
    try {
      await strategiesApi.toggleStrategy(strategyId)
      fetchStrategies()
    } catch (error) {
      console.error('Failed to toggle strategy:', error)
    }
  }

  const handleDelete = async (strategyId: string) => {
    if (!confirm('Are you sure you want to delete this strategy?')) return
    try {
      await strategiesApi.deleteStrategy(strategyId)
      fetchStrategies()
    } catch (error) {
      console.error('Failed to delete strategy:', error)
    }
  }

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault()
    setSubmitting(true)
    try {
      await strategiesApi.createStrategy(formData)
      setShowModal(false)
      setFormData({ name: '', type: 'trend_following', config: {} })
      fetchStrategies()
    } catch (error) {
      console.error('Failed to create strategy:', error)
    } finally {
      setSubmitting(false)
    }
  }

  const getStrategyIcon = (type: string) => {
    const st = STRATEGY_TYPES.find(s => s.value === type)
    const Icon = st?.icon || Cpu
    return <Icon className="w-5 h-5" />
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Strategies</h1>
          <p className="text-slate-400">Manage your algorithmic trading strategies</p>
        </div>
        <button
          onClick={() => setShowModal(true)}
          className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-cyan-500 to-violet-600 text-white rounded-lg hover:from-cyan-400 hover:to-violet-500 transition-all"
        >
          <Plus className="w-5 h-5" />
          New Strategy
        </button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="glass-card rounded-xl p-6">
          <p className="text-sm text-slate-400 mb-1">Total Strategies</p>
          <p className="text-3xl font-bold text-white font-mono">{strategies.length}</p>
        </div>
        <div className="glass-card rounded-xl p-6">
          <p className="text-sm text-slate-400 mb-1">Active</p>
          <p className="text-3xl font-bold text-emerald-400 font-mono">
            {strategies.filter(s => s.is_active).length}
          </p>
        </div>
        <div className="glass-card rounded-xl p-6">
          <p className="text-sm text-slate-400 mb-1">Avg Performance</p>
          <p className="text-3xl font-bold text-cyan-400 font-mono">
            {(strategies.reduce((acc, s) => acc + s.performance, 0) / (strategies.length || 1)).toFixed(1)}%
          </p>
        </div>
      </div>

      {/* Strategies Grid */}
      {loading ? (
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin w-8 h-8 border-2 border-cyan-500 border-t-transparent rounded-full"></div>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {strategies.map((strategy) => (
            <div key={strategy.id} className="glass-card rounded-xl p-6 hover:border-cyan-500/30 transition-all">
              <div className="flex items-start justify-between mb-4">
                <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-cyan-500/20 to-violet-600/20 flex items-center justify-center text-cyan-400">
                  {getStrategyIcon(strategy.type)}
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => handleToggle(strategy.id)}
                    className={`p-2 rounded-lg transition-colors ${
                      strategy.is_active
                        ? 'bg-emerald-500/20 text-emerald-400 hover:bg-emerald-500/30'
                        : 'bg-slate-700/50 text-slate-400 hover:text-white'
                    }`}
                  >
                    {strategy.is_active ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                  </button>
                  <button
                    onClick={() => handleDelete(strategy.id)}
                    className="p-2 rounded-lg bg-red-500/20 text-red-400 hover:bg-red-500/30 transition-colors"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>

              <h3 className="text-lg font-semibold text-white mb-1">{strategy.name}</h3>
              <p className="text-sm text-slate-400 capitalize mb-4">{strategy.type.replace('_', ' ')}</p>

              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs text-slate-400">Performance</p>
                  <p className={`text-lg font-bold font-mono ${strategy.performance >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                    {strategy.performance >= 0 ? '+' : ''}{strategy.performance.toFixed(1)}%
                  </p>
                </div>
                <div className="text-right">
                  <p className="text-xs text-slate-400">Status</p>
                  <span className={`inline-flex items-center gap-1 text-sm font-medium ${
                    strategy.is_active ? 'text-emerald-400' : 'text-slate-400'
                  }`}>
                    {strategy.is_active ? <Check className="w-4 h-4" /> : <Pause className="w-4 h-4" />}
                    {strategy.is_active ? 'Running' : 'Paused'}
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* New Strategy Modal */}
      {showModal && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="glass-card rounded-xl p-6 w-full max-w-md">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-semibold text-white">Create Strategy</h3>
              <button onClick={() => setShowModal(false)} className="text-slate-400 hover:text-white">
                <X className="w-5 h-5" />
              </button>
            </div>

            <form onSubmit={handleCreate} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-1">Strategy Name</label>
                <input
                  type="text"
                  value={formData.name}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                  placeholder="e.g., Momentum Alpha"
                  className="w-full bg-slate-800/50 border border-white/10 rounded-lg px-4 py-3 text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500/50"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-300 mb-1">Strategy Type</label>
                <select
                  value={formData.type}
                  onChange={(e) => setFormData({ ...formData, type: e.target.value })}
                  className="w-full bg-slate-800/50 border border-white/10 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-cyan-500/50"
                >
                  {STRATEGY_TYPES.map(type => (
                    <option key={type.value} value={type.value}>{type.label}</option>
                  ))}
                </select>
              </div>

              <button
                type="submit"
                disabled={submitting}
                className="w-full bg-gradient-to-r from-cyan-500 to-violet-600 text-white font-semibold py-3 rounded-lg hover:from-cyan-400 hover:to-violet-500 transition-all disabled:opacity-50 flex items-center justify-center gap-2"
              >
                {submitting && <Loader2 className="w-5 h-5 animate-spin" />}
                Create Strategy
              </button>
            </form>
          </div>
        </div>
      )}
    </div>
  )
}
