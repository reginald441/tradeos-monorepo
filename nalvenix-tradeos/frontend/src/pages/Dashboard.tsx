import { useEffect, useState } from 'react'
import { dashboardApi, DashboardData } from '../api/dashboard'
import StatCard from '../components/StatCard'
import MarketTable from '../components/MarketTable'
import PriceChart from '../components/PriceChart'
import { 
  Wallet, 
  TrendingUp, 
  Activity, 
  Target,
  Bitcoin,
  Globe,
  Gem,
  BarChart3
} from 'lucide-react'

export default function Dashboard() {
  const [data, setData] = useState<DashboardData | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await dashboardApi.getDashboard()
        setData(response)
      } catch (error) {
        console.error('Failed to fetch dashboard:', error)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="animate-spin w-12 h-12 border-3 border-cyan-500 border-t-transparent rounded-full"></div>
      </div>
    )
  }

  const portfolio = data?.portfolio

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Dashboard</h1>
          <p className="text-slate-400">Welcome back to your trading dashboard</p>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse"></span>
          <span className="text-sm text-emerald-400">Live Market Data</span>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          title="Total Equity"
          value={portfolio?.total_equity.toLocaleString('en-US', { minimumFractionDigits: 2 }) || '0.00'}
          prefix="$"
          change={2.35}
          icon={Wallet}
        />
        <StatCard
          title="Day P&L"
          value={portfolio?.day_pnl.toLocaleString('en-US', { minimumFractionDigits: 2 }) || '0.00'}
          prefix="$"
          change={portfolio?.day_pnl_percent}
          icon={TrendingUp}
        />
        <StatCard
          title="Buying Power"
          value={portfolio?.buying_power.toLocaleString('en-US', { minimumFractionDigits: 2 }) || '0.00'}
          prefix="$"
          icon={Activity}
        />
        <StatCard
          title="Win Rate"
          value={data?.stats.win_rate.toString() || '0'}
          suffix="%"
          change={5.2}
          icon={Target}
        />
      </div>

      {/* Chart Section */}
      <div className="glass-card rounded-xl p-6">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-lg font-semibold text-white">Portfolio Performance</h2>
            <p className="text-sm text-slate-400">Real-time price action</p>
          </div>
          <div className="flex items-center gap-2">
            <span className="px-3 py-1 rounded-full bg-cyan-500/20 text-cyan-400 text-sm font-medium">BTC-USD</span>
          </div>
        </div>
        <PriceChart symbol="BTC-USD" height={350} />
      </div>

      {/* Market Overview */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <MarketTable 
          title={
            <div className="flex items-center gap-2">
              <Bitcoin className="w-5 h-5 text-orange-400" />
              <span>Crypto Markets</span>
            </div>
          } 
          data={data?.market_overview.crypto || []} 
        />
        <MarketTable 
          title={
            <div className="flex items-center gap-2">
              <Globe className="w-5 h-5 text-blue-400" />
              <span>Forex Markets</span>
            </div>
          } 
          data={data?.market_overview.forex || []} 
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <MarketTable 
          title={
            <div className="flex items-center gap-2">
              <Gem className="w-5 h-5 text-yellow-400" />
              <span>Commodities</span>
            </div>
          } 
          data={data?.market_overview.commodities || []} 
        />
        <MarketTable 
          title={
            <div className="flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-purple-400" />
              <span>Indices</span>
            </div>
          } 
          data={data?.market_overview.indices || []} 
        />
      </div>

      {/* Quick Stats */}
      <div className="glass-card rounded-xl p-6">
        <h3 className="text-lg font-semibold text-white mb-4">Trading Activity</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="text-center p-4 rounded-lg bg-white/5">
            <p className="text-3xl font-bold text-cyan-400 font-mono">{data?.stats.open_positions}</p>
            <p className="text-sm text-slate-400 mt-1">Open Positions</p>
          </div>
          <div className="text-center p-4 rounded-lg bg-white/5">
            <p className="text-3xl font-bold text-violet-400 font-mono">{data?.stats.active_strategies}</p>
            <p className="text-sm text-slate-400 mt-1">Active Strategies</p>
          </div>
          <div className="text-center p-4 rounded-lg bg-white/5">
            <p className="text-3xl font-bold text-emerald-400 font-mono">{data?.stats.today_trades}</p>
            <p className="text-sm text-slate-400 mt-1">Today's Trades</p>
          </div>
        </div>
      </div>
    </div>
  )
}
