import { useEffect, useState } from 'react'
import { riskApi, RiskSettings } from '../api/risk'
import { 
  Shield, 
  AlertTriangle, 
  TrendingDown, 
  DollarSign,
  Percent,
  Activity,
  Save,
  Loader2
} from 'lucide-react'

export default function Risk() {
  const [settings, setSettings] = useState<RiskSettings>({
    max_position_size: 25000,
    max_daily_loss: 5000,
    var_limit: 0.02,
    kill_switch_enabled: true
  })
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)

  useEffect(() => {
    const fetchSettings = async () => {
      try {
        const response = await riskApi.getRiskSettings()
        setSettings(response)
      } catch (error) {
        console.error('Failed to fetch risk settings:', error)
      } finally {
        setLoading(false)
      }
    }

    fetchSettings()
  }, [])

  const handleSave = async () => {
    setSaving(true)
    setSaved(false)
    try {
      await riskApi.updateRiskSettings(settings)
      setSaved(true)
      setTimeout(() => setSaved(false), 3000)
    } catch (error) {
      console.error('Failed to save risk settings:', error)
    } finally {
      setSaving(false)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="animate-spin w-8 h-8 border-2 border-cyan-500 border-t-transparent rounded-full"></div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-white">Risk Management</h1>
        <p className="text-slate-400">Configure your risk parameters and limits</p>
      </div>

      {/* Risk Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="glass-card rounded-xl p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 rounded-lg bg-emerald-500/20 flex items-center justify-center">
              <Shield className="w-5 h-5 text-emerald-400" />
            </div>
            <div>
              <p className="text-sm text-slate-400">Risk Level</p>
              <p className="text-lg font-semibold text-emerald-400">Low</p>
            </div>
          </div>
        </div>
        <div className="glass-card rounded-xl p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 rounded-lg bg-cyan-500/20 flex items-center justify-center">
              <Activity className="w-5 h-5 text-cyan-400" />
            </div>
            <div>
              <p className="text-sm text-slate-400">Current VaR</p>
              <p className="text-lg font-semibold text-cyan-400 font-mono">1.85%</p>
            </div>
          </div>
        </div>
        <div className="glass-card rounded-xl p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 rounded-lg bg-violet-500/20 flex items-center justify-center">
              <TrendingDown className="w-5 h-5 text-violet-400" />
            </div>
            <div>
              <p className="text-sm text-slate-400">Max Drawdown</p>
              <p className="text-lg font-semibold text-violet-400 font-mono">-8.5%</p>
            </div>
          </div>
        </div>
        <div className="glass-card rounded-xl p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 rounded-lg bg-orange-500/20 flex items-center justify-center">
              <AlertTriangle className="w-5 h-5 text-orange-400" />
            </div>
            <div>
              <p className="text-sm text-slate-400">Alerts</p>
              <p className="text-lg font-semibold text-orange-400">0 Active</p>
            </div>
          </div>
        </div>
      </div>

      {/* Risk Settings */}
      <div className="glass-card rounded-xl p-6">
        <div className="flex items-center gap-2 mb-6">
          <Shield className="w-5 h-5 text-cyan-400" />
          <h2 className="text-lg font-semibold text-white">Risk Parameters</h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              <DollarSign className="w-4 h-4 inline mr-1" />
              Max Position Size
            </label>
            <input
              type="number"
              value={settings.max_position_size}
              onChange={(e) => setSettings({ ...settings, max_position_size: parseFloat(e.target.value) })}
              className="w-full bg-slate-800/50 border border-white/10 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-cyan-500/50"
            />
            <p className="text-xs text-slate-400 mt-1">Maximum capital allocated per position</p>
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              <TrendingDown className="w-4 h-4 inline mr-1" />
              Max Daily Loss
            </label>
            <input
              type="number"
              value={settings.max_daily_loss}
              onChange={(e) => setSettings({ ...settings, max_daily_loss: parseFloat(e.target.value) })}
              className="w-full bg-slate-800/50 border border-white/10 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-cyan-500/50"
            />
            <p className="text-xs text-slate-400 mt-1">Maximum loss allowed per trading day</p>
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              <Percent className="w-4 h-4 inline mr-1" />
              VaR Limit
            </label>
            <input
              type="number"
              step="0.01"
              value={settings.var_limit}
              onChange={(e) => setSettings({ ...settings, var_limit: parseFloat(e.target.value) })}
              className="w-full bg-slate-800/50 border border-white/10 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-cyan-500/50"
            />
            <p className="text-xs text-slate-400 mt-1">Value at Risk limit (95% confidence)</p>
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              <AlertTriangle className="w-4 h-4 inline mr-1" />
              Kill Switch
            </label>
            <div className="flex items-center gap-4">
              <button
                onClick={() => setSettings({ ...settings, kill_switch_enabled: !settings.kill_switch_enabled })}
                className={`relative w-14 h-8 rounded-full transition-colors ${
                  settings.kill_switch_enabled ? 'bg-emerald-500' : 'bg-slate-600'
                }`}
              >
                <span className={`absolute top-1 w-6 h-6 bg-white rounded-full transition-transform ${
                  settings.kill_switch_enabled ? 'translate-x-7' : 'translate-x-1'
                }`} />
              </button>
              <span className={`text-sm ${settings.kill_switch_enabled ? 'text-emerald-400' : 'text-slate-400'}`}>
                {settings.kill_switch_enabled ? 'Enabled' : 'Disabled'}
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-1">Automatically stop trading on risk breach</p>
          </div>
        </div>

        <div className="mt-6 flex items-center gap-4">
          <button
            onClick={handleSave}
            disabled={saving}
            className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-cyan-500 to-violet-600 text-white rounded-lg hover:from-cyan-400 hover:to-violet-500 transition-all disabled:opacity-50"
          >
            {saving ? <Loader2 className="w-5 h-5 animate-spin" /> : <Save className="w-5 h-5" />}
            Save Settings
          </button>
          {saved && (
            <span className="text-emerald-400 text-sm">Settings saved successfully!</span>
          )}
        </div>
      </div>

      {/* Risk Alerts */}
      <div className="glass-card rounded-xl p-6">
        <h2 className="text-lg font-semibold text-white mb-4">Recent Risk Events</h2>
        <div className="space-y-3">
          <div className="flex items-center gap-4 p-4 rounded-lg bg-emerald-500/10 border border-emerald-500/30">
            <Shield className="w-5 h-5 text-emerald-400" />
            <div className="flex-1">
              <p className="text-sm font-medium text-white">Risk parameters within normal range</p>
              <p className="text-xs text-slate-400">All risk metrics are within acceptable limits</p>
            </div>
            <span className="text-xs text-slate-400">Just now</span>
          </div>
          <div className="flex items-center gap-4 p-4 rounded-lg bg-slate-700/30">
            <Activity className="w-5 h-5 text-cyan-400" />
            <div className="flex-1">
              <p className="text-sm font-medium text-white">Daily VaR calculated</p>
              <p className="text-xs text-slate-400">Current VaR: 1.85% of portfolio</p>
            </div>
            <span className="text-xs text-slate-400">2 hours ago</span>
          </div>
          <div className="flex items-center gap-4 p-4 rounded-lg bg-slate-700/30">
            <TrendingDown className="w-5 h-5 text-violet-400" />
            <div className="flex-1">
              <p className="text-sm font-medium text-white">Position sizing adjusted</p>
              <p className="text-xs text-slate-400">Reduced BTC position by 15%</p>
            </div>
            <span className="text-xs text-slate-400">5 hours ago</span>
          </div>
        </div>
      </div>
    </div>
  )
}
