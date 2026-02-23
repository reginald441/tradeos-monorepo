import { useEffect, useState } from 'react'
import { marketApi, CandleData } from '../api/market'
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer
} from 'recharts'

interface PriceChartProps {
  symbol?: string
  height?: number
}

export default function PriceChart({ symbol = 'BTC-USD', height = 400 }: PriceChartProps) {
  const [data, setData] = useState<CandleData[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await marketApi.getMarketData(symbol)
        setData(response.data)
      } catch (error) {
        console.error('Failed to fetch market data:', error)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [symbol])

  if (loading) {
    return (
      <div className="flex items-center justify-center" style={{ height }}>
        <div className="animate-spin w-8 h-8 border-2 border-cyan-500 border-t-transparent rounded-full"></div>
      </div>
    )
  }

  const formatPrice = (value: number) => {
    return value.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
  }

  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={data}>
        <defs>
          <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#22d3ee" stopOpacity={0.3}/>
            <stop offset="95%" stopColor="#22d3ee" stopOpacity={0}/>
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
        <XAxis 
          dataKey="timestamp" 
          tickFormatter={(value) => new Date(value).toLocaleDateString()}
          stroke="#64748b"
          fontSize={12}
        />
        <YAxis 
          tickFormatter={formatPrice}
          stroke="#64748b"
          fontSize={12}
          domain={['auto', 'auto']}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: 'rgba(15, 23, 42, 0.95)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            borderRadius: '8px',
            color: '#fff'
          }}
          formatter={(value: number) => [formatPrice(value), 'Price']}
          labelFormatter={(label) => new Date(label).toLocaleString()}
        />
        <Area
          type="monotone"
          dataKey="close"
          stroke="#22d3ee"
          strokeWidth={2}
          fillOpacity={1}
          fill="url(#colorPrice)"
        />
      </AreaChart>
    </ResponsiveContainer>
  )
}
