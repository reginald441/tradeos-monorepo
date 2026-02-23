import { useEffect, useState } from 'react'
import { billingApi, BillingData } from '../api/billing'
import { CreditCard, Check, Zap, Crown, Building2, Download } from 'lucide-react'

const PLANS = [
  {
    name: 'Free',
    price: 0,
    features: ['3 Strategies', '1,000 API calls/day', 'Basic backtesting', 'Email support'],
    icon: Zap,
    color: 'slate'
  },
  {
    name: 'Pro',
    price: 99,
    features: ['10 Strategies', '10,000 API calls/day', 'Advanced backtesting', 'Priority support', 'Real-time data'],
    icon: Crown,
    color: 'cyan',
    popular: true
  },
  {
    name: 'Enterprise',
    price: 299,
    features: ['Unlimited Strategies', '100,000 API calls/day', 'AI-powered insights', '24/7 dedicated support', 'Custom integrations'],
    icon: Building2,
    color: 'violet'
  }
]

export default function Billing() {
  const [data, setData] = useState<BillingData | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await billingApi.getBilling()
        setData(response)
      } catch (error) {
        console.error('Failed to fetch billing:', error)
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

  const usage = data?.usage
  const subscription = data?.subscription

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-white">Billing</h1>
        <p className="text-slate-400">Manage your subscription and billing</p>
      </div>

      {/* Current Plan */}
      <div className="glass-card rounded-xl p-6">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-4">
            <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-cyan-500 to-violet-600 flex items-center justify-center">
              <Crown className="w-7 h-7 text-white" />
            </div>
            <div>
              <p className="text-sm text-slate-400">Current Plan</p>
              <h2 className="text-2xl font-bold text-white capitalize">{subscription?.plan}</h2>
              <p className="text-sm text-cyan-400">
                {subscription?.status === 'active' ? 'Active' : 'Inactive'}
                {subscription?.expires_at && ` · Expires ${new Date(subscription.expires_at).toLocaleDateString()}`}
              </p>
            </div>
          </div>
          <button className="px-4 py-2 bg-gradient-to-r from-cyan-500 to-violet-600 text-white rounded-lg hover:from-cyan-400 hover:to-violet-500 transition-all">
            Upgrade Plan
          </button>
        </div>

        {/* Usage */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-slate-400">API Calls</span>
              <span className="text-sm text-white font-mono">{usage?.api_calls.toLocaleString()} / {usage?.api_limit.toLocaleString()}</span>
            </div>
            <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
              <div 
                className="h-full bg-gradient-to-r from-cyan-500 to-violet-600 rounded-full transition-all"
                style={{ width: `${((usage?.api_calls || 0) / (usage?.api_limit || 1)) * 100}%` }}
              />
            </div>
          </div>
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-slate-400">Strategies</span>
              <span className="text-sm text-white font-mono">{usage?.strategies_used} / {usage?.strategies_limit}</span>
            </div>
            <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
              <div 
                className="h-full bg-gradient-to-r from-emerald-500 to-cyan-500 rounded-full transition-all"
                style={{ width: `${((usage?.strategies_used || 0) / (usage?.strategies_limit || 1)) * 100}%` }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Plans */}
      <div>
        <h2 className="text-lg font-semibold text-white mb-4">Available Plans</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {PLANS.map((plan) => {
            const Icon = plan.icon
            const isCurrentPlan = subscription?.plan.toLowerCase() === plan.name.toLowerCase()
            
            return (
              <div 
                key={plan.name} 
                className={`glass-card rounded-xl p-6 relative ${plan.popular ? 'border-cyan-500/50' : ''}`}
              >
                {plan.popular && (
                  <span className="absolute -top-3 left-1/2 -translate-x-1/2 px-3 py-1 bg-gradient-to-r from-cyan-500 to-violet-600 text-white text-xs font-medium rounded-full">
                    Most Popular
                  </span>
                )}
                
                <div className="text-center mb-6">
                  <div className={`w-12 h-12 rounded-lg mx-auto mb-4 flex items-center justify-center ${
                    plan.color === 'cyan' ? 'bg-cyan-500/20 text-cyan-400' :
                    plan.color === 'violet' ? 'bg-violet-500/20 text-violet-400' :
                    'bg-slate-700 text-slate-400'
                  }`}>
                    <Icon className="w-6 h-6" />
                  </div>
                  <h3 className="text-xl font-semibold text-white">{plan.name}</h3>
                  <div className="mt-2">
                    <span className="text-3xl font-bold text-white">${plan.price}</span>
                    <span className="text-slate-400">/month</span>
                  </div>
                </div>

                <ul className="space-y-3 mb-6">
                  {plan.features.map((feature) => (
                    <li key={feature} className="flex items-center gap-2 text-sm text-slate-300">
                      <Check className="w-4 h-4 text-emerald-400" />
                      {feature}
                    </li>
                  ))}
                </ul>

                <button
                  disabled={isCurrentPlan}
                  className={`w-full py-3 rounded-lg font-medium transition-all ${
                    isCurrentPlan
                      ? 'bg-emerald-500/20 text-emerald-400 cursor-default'
                      : plan.popular
                        ? 'bg-gradient-to-r from-cyan-500 to-violet-600 text-white hover:from-cyan-400 hover:to-violet-500'
                        : 'bg-slate-700 text-white hover:bg-slate-600'
                  }`}
                >
                  {isCurrentPlan ? 'Current Plan' : 'Select Plan'}
                </button>
              </div>
            )
          })}
        </div>
      </div>

      {/* Invoices */}
      <div className="glass-card rounded-xl overflow-hidden">
        <div className="px-6 py-4 border-b border-white/5">
          <h2 className="text-lg font-semibold text-white flex items-center gap-2">
            <CreditCard className="w-5 h-5 text-cyan-400" />
            Billing History
          </h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="text-left text-xs text-slate-400 border-b border-white/5">
                <th className="px-6 py-3 font-medium">Invoice ID</th>
                <th className="px-6 py-3 font-medium">Date</th>
                <th className="px-6 py-3 font-medium text-right">Amount</th>
                <th className="px-6 py-3 font-medium text-center">Status</th>
                <th className="px-6 py-3 font-medium text-center">Action</th>
              </tr>
            </thead>
            <tbody>
              {data?.invoices.map((invoice) => (
                <tr key={invoice.id} className="border-b border-white/5 last:border-0 hover:bg-white/5">
                  <td className="px-6 py-4 font-mono text-white">{invoice.id}</td>
                  <td className="px-6 py-4 text-slate-300">{invoice.date}</td>
                  <td className="px-6 py-4 text-right font-mono text-white">${invoice.amount.toFixed(2)}</td>
                  <td className="px-6 py-4 text-center">
                    <span className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${
                      invoice.status === 'paid' 
                        ? 'bg-emerald-500/20 text-emerald-400' 
                        : 'bg-orange-500/20 text-orange-400'
                    }`}>
                      {invoice.status === 'paid' && <Check className="w-3 h-3" />}
                      {invoice.status.charAt(0).toUpperCase() + invoice.status.slice(1)}
                    </span>
                  </td>
                  <td className="px-6 py-4 text-center">
                    <button className="p-2 text-slate-400 hover:text-white transition-colors">
                      <Download className="w-4 h-4" />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
