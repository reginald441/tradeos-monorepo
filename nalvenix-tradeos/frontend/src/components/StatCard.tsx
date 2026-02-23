import { TrendingUp, TrendingDown } from 'lucide-react'

interface StatCardProps {
  title: string
  value: string
  change?: number
  prefix?: string
  suffix?: string
  icon: React.ElementType
}

export default function StatCard({ title, value, change, prefix = '', suffix = '', icon: Icon }: StatCardProps) {
  const isPositive = change && change >= 0
  
  return (
    <div className="glass-card rounded-xl p-6 hover:border-cyan-500/30 transition-all duration-300">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm text-slate-400 mb-1">{title}</p>
          <p className="text-2xl font-bold text-white font-mono">
            {prefix}{value}{suffix}
          </p>
          {change !== undefined && (
            <div className={`flex items-center gap-1 mt-2 ${isPositive ? 'text-emerald-400' : 'text-red-400'}`}>
              {isPositive ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
              <span className="text-sm font-medium">{isPositive ? '+' : ''}{change.toFixed(2)}%</span>
            </div>
          )}
        </div>
        <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-cyan-500/20 to-violet-600/20 flex items-center justify-center">
          <Icon className="w-6 h-6 text-cyan-400" />
        </div>
      </div>
    </div>
  )
}
