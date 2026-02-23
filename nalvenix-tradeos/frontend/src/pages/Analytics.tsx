import { useEffect, useState } from 'react'
import { analyticsApi, AnalyticsData } from '../api/analytics'
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell
} from 'recharts'
import { TrendingUp, TrendingDown, Target, Activity, BarChart3, DollarSign } from 'lucide-react'

const COLORS = ['#22d3ee', '#8b5cf6', '#10b981', '#f59e0b', '#ef4444']

export default function Analytics() {
  const [data, setData] = useState<AnalyticsData | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await analyticsApi.getAnalytics()
        setData(response)
      } catch (error) {
        console.error('Failed to fetch analytics:', error)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="animate-spin w-8 h-8 border-2 border-cyan-500 border-t-transparent rounded-full"></div>
      </div>
    )
  }

  const metrics = data?.metrics

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-white">Analytics</h1>
        <p className="text-slate-400">Comprehensive trading performance analysis</p>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
        <div className="glass-card rounded-xl p-4 text-center">
          <Activity className="w-5 h-5 text-cyan-400 mx-auto mb-2" />
          <p className="text-2xl font-bold text-cyan-400 font-mono">{metrics?.sharpe_ratio}</p>
          <p className="text-xs text-slate-400">Sharpe Ratio</p>
        </div>
        <div className="glass-card rounded-xl p-4 text-center">
          <TrendingDown className="w-5 h-5 text-red-400 mx-auto mb-2" />
          <p className="text-2xl font-bold text-red-400 font-mono">{metrics?.max_drawdown}%</p>
          <p className="text-xs text-slate-400">Max Drawdown</p>
        </div>
        <div className="glass-card rounded-xl p-4 text-center">
          <Target className="w-5 h-5 text-emerald-400 mx-auto mb-2" />
          <p className="text-2xl font-bold text-emerald-400 font-mono">{metrics?.win_rate}%</p>
          <p className="text-xs text-slate-400">Win Rate</p>
        </div>
        <div className="glass-card rounded-xl p-4 text-center">
          <BarChart3 className="w-5 h-5 text-violet-400 mx-auto mb-2" />
          <p className="text-2xl font-bold text-violet-400 font-mono">{metrics?.profit_factor}</p>
          <p className="text-xs text-slate-400">Profit Factor</p>
        </div>
        <div className="glass-card rounded-xl p-4 text-center">
          <DollarSign className="w-5 h-5 text-orange-400 mx-auto mb-2" />
          <p className="text-2xl font-bold text-orange-400 font-mono">${metrics?.avg_trade}</p>
          <p className="text-xs text-slate-400">Avg Trade</p>
        </div>
        <div className="glass-card rounded-xl p-4 text-center">
          <TrendingUp className="w-5 h-5 text-blue-400 mx-auto mb-2" />
          <p className="text-2xl font-bold text-blue-400 font-mono">{metrics?.total_trades}</p>
          <p className="text-xs text-slate-400">Total Trades</p>
        </div>
      </div>

      {/* P&L Chart */}
      <div className="glass-card rounded-xl p-6">
        <h2 className="text-lg font-semibold text-white mb-4">Daily P&L</h2>
        <ResponsiveContainer width="100%" height={300}>
          <AreaChart data={data?.daily_pnl}>
            <defs>
              <linearGradient id="colorPnl" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#22d3ee" stopOpacity={0.3}/>
                <stop offset="95%" stopColor="#22d3ee" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
            <XAxis 
              dataKey="date" 
              tickFormatter={(value) => new Date(value).toLocaleDateString()}
              stroke="#64748b"
              fontSize={12}
            />
            <YAxis 
              tickFormatter={(value) => `$${value}`}
              stroke="#64748b"
              fontSize={12}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: 'rgba(15, 23, 42, 0.95)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                borderRadius: '8px',
                color: '#fff'
              }}
              formatter={(value: number) => [`$${value.toFixed(2)}`, 'P&L']}
            />
            <Area
              type="monotone"
              dataKey="pnl"
              stroke="#22d3ee"
              strokeWidth={2}
              fillOpacity={1}
              fill="url(#colorPnl)"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Strategy Performance & Symbol Distribution */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="glass-card rounded-xl p-6">
          <h2 className="text-lg font-semibold text-white mb-4">Strategy Performance</h2>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={data?.strategy_performance}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="name" stroke="#64748b" fontSize={11} angle={-45} textAnchor="end" height={80} />
              <YAxis stroke="#64748b" fontSize={12} tickFormatter={(v) => `${v}%`} />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'rgba(15, 23, 42, 0.95)',
                  border: '1px solid rgba(255, 255, 255, 0.1)',
                  borderRadius: '8px',
                  color: '#fff'
                }}
                formatter={(value: number) => [`${value.toFixed(1)}%`, 'Return']}
              />
              <Bar dataKey="return" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="glass-card rounded-xl p-6">
          <h2 className="text-lg font-semibold text-white mb-4">Symbol Distribution</h2>
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie
                data={data?.symbol_distribution}
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={100}
                paddingAngle={5}
                dataKey="count"
                nameKey="symbol"
              >
                {data?.symbol_distribution.map((_, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{
                  backgroundColor: 'rgba(15, 23, 42, 0.95)',
                  border: '1px solid rgba(255, 255, 255, 0.1)',
                  borderRadius: '8px',
                  color: '#fff'
                }}
              />
            </PieChart>
          </ResponsiveContainer>
          <div className="flex flex-wrap justify-center gap-4 mt-4">
            {data?.symbol_distribution.map((item, index) => (
              <div key={item.symbol} className="flex items-center gap-2">
                <span 
                  className="w-3 h-3 rounded-full" 
                  style={{ backgroundColor: COLORS[index % COLORS.length] }}
                />
                <span className="text-sm text-slate-400">{item.symbol}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Monthly Summary */}
      <div className="glass-card rounded-xl p-6">
        <h2 className="text-lg font-semibold text-white mb-4">Monthly Performance</h2>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="text-left text-xs text-slate-400 border-b border-white/5">
                <th className="px-4 py-3 font-medium">Month</th>
                <th className="px-4 py-3 font-medium text-right">Trades</th>
                <th className="px-4 py-3 font-medium text-right">Wins</th>
                <th className="px-4 py-3 font-medium text-right">Losses</th>
                <th className="px-4 py-3 font-medium text-right">Win Rate</th>
                <th className="px-4 py-3 font-medium text-right">P&L</th>
              </tr>
            </thead>
            <tbody>
              {['January', 'February', 'March', 'April', 'May'].map((month, i) => {
                const trades = 25 + i * 3
                const wins = 15 + i * 2
                const losses = trades - wins
                const winRate = ((wins / trades) * 100).toFixed(1)
                const pnl = (Math.random() * 5000 - 1000).toFixed(2)
                const isPositive = parseFloat(pnl) > 0
                
                return (
                  <tr key={month} className="border-b border-white/5 last:border-0 hover:bg-white/5">
                    <td className="px-4 py-3 text-white">{month} 2024</td>
                    <td className="px-4 py-3 text-right font-mono text-white">{trades}</td>
                    <td className="px-4 py-3 text-right font-mono text-emerald-400">{wins}</td>
                    <td className="px-4 py-3 text-right font-mono text-red-400">{losses}</td>
                    <td className="px-4 py-3 text-right font-mono text-cyan-400">{winRate}%</td>
                    <td className={`px-4 py-3 text-right font-mono ${isPositive ? 'text-emerald-400' : 'text-red-400'}`}>
                      {isPositive ? '+' : ''}${pnl}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
