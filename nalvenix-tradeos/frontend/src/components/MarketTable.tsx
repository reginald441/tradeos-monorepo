import { TrendingUp, TrendingDown } from 'lucide-react'

interface MarketItem {
  symbol: string
  price: number
  change_24h: number
  volume: number
}

interface MarketTableProps {
  title: string
  data: MarketItem[]
}

export default function MarketTable({ title, data }: MarketTableProps) {
  const formatPrice = (price: number) => {
    return price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
  }

  const formatVolume = (volume: number) => {
    if (volume >= 1e9) return `$${(volume / 1e9).toFixed(2)}B`
    if (volume >= 1e6) return `$${(volume / 1e6).toFixed(2)}M`
    return `$${(volume / 1e3).toFixed(2)}K`
  }

  return (
    <div className="glass-card rounded-xl overflow-hidden">
      <div className="px-6 py-4 border-b border-white/5">
        <h3 className="text-lg font-semibold text-white">{title}</h3>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="text-left text-xs text-slate-400 border-b border-white/5">
              <th className="px-6 py-3 font-medium">Symbol</th>
              <th className="px-6 py-3 font-medium text-right">Price</th>
              <th className="px-6 py-3 font-medium text-right">24h Change</th>
              <th className="px-6 py-3 font-medium text-right">Volume</th>
            </tr>
          </thead>
          <tbody>
            {data.map((item, index) => {
              const isPositive = item.change_24h >= 0
              return (
                <tr key={index} className="border-b border-white/5 last:border-0 hover:bg-white/5 transition-colors">
                  <td className="px-6 py-4">
                    <span className="font-mono font-medium text-white">{item.symbol}</span>
                  </td>
                  <td className="px-6 py-4 text-right">
                    <span className="font-mono text-white">${formatPrice(item.price)}</span>
                  </td>
                  <td className="px-6 py-4 text-right">
                    <div className={`flex items-center justify-end gap-1 ${isPositive ? 'text-emerald-400' : 'text-red-400'}`}>
                      {isPositive ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
                      <span className="font-mono">{isPositive ? '+' : ''}{item.change_24h.toFixed(2)}%</span>
                    </div>
                  </td>
                  <td className="px-6 py-4 text-right">
                    <span className="font-mono text-slate-400">{formatVolume(item.volume)}</span>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
