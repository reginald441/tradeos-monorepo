import React from 'react';
import {
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import type { Candle } from '@/types';

interface PriceChartProps {
  data: Candle[];
  height?: number;
  showVolume?: boolean;
  showGrid?: boolean;
  symbol?: string;
}

interface ChartDataPoint {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  ma20?: number;
  ma50?: number;
}

const CustomTooltip: React.FC<{
  active?: boolean;
  payload?: any[];
  label?: string;
}> = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    const isUp = data.close >= data.open;
    
    return (
      <div className="bg-slate-900 border border-slate-700 rounded-lg p-3 shadow-xl">
        <p className="text-xs text-slate-500 mb-2">
          {new Date(data.timestamp).toLocaleString()}
        </p>
        <div className="space-y-1 text-sm">
          <div className="flex justify-between gap-4">
            <span className="text-slate-400">Open:</span>
            <span className="font-mono">{data.open.toFixed(2)}</span>
          </div>
          <div className="flex justify-between gap-4">
            <span className="text-slate-400">High:</span>
            <span className="font-mono">{data.high.toFixed(2)}</span>
          </div>
          <div className="flex justify-between gap-4">
            <span className="text-slate-400">Low:</span>
            <span className="font-mono">{data.low.toFixed(2)}</span>
          </div>
          <div className="flex justify-between gap-4">
            <span className="text-slate-400">Close:</span>
            <span className={`font-mono ${isUp ? 'text-emerald-400' : 'text-red-400'}`}>
              {data.close.toFixed(2)}
            </span>
          </div>
          <div className="flex justify-between gap-4">
            <span className="text-slate-400">Volume:</span>
            <span className="font-mono">{data.volume.toFixed(0)}</span>
          </div>
        </div>
      </div>
    );
  }
  return null;
};

export const PriceChart: React.FC<PriceChartProps> = ({
  data,
  height = 400,
  showVolume = true,
  showGrid = true,
  symbol = 'BTC-USD',
}) => {
  // Calculate moving averages
  const chartData: ChartDataPoint[] = React.useMemo(() => {
    return data.map((candle, index) => {
      const ma20 = index >= 19
        ? data.slice(index - 19, index + 1).reduce((sum, c) => sum + c.close, 0) / 20
        : undefined;
      const ma50 = index >= 49
        ? data.slice(index - 49, index + 1).reduce((sum, c) => sum + c.close, 0) / 50
        : undefined;
      
      return {
        ...candle,
        ma20,
        ma50,
      };
    });
  }, [data]);

  const currentPrice = data[data.length - 1]?.close || 0;
  const priceChange = data.length > 1 
    ? currentPrice - data[data.length - 2].close 
    : 0;
  const priceChangePercent = data.length > 1
    ? (priceChange / data[data.length - 2].close) * 100
    : 0;

  const formatDate = (timestamp: string) => {
    const date = new Date(timestamp);
    return `${date.getHours()}:${date.getMinutes().toString().padStart(2, '0')}`;
  };

  const formatPrice = (value: number) => {
    return value.toLocaleString('en-US', {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    });
  };

  return (
    <div className="w-full">
      {/* Price Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-4">
          <div>
            <h2 className="text-xl font-bold text-slate-100">{symbol}</h2>
            <p className="text-sm text-slate-500">Perpetual</p>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-2xl font-mono font-bold text-slate-100">
              {formatPrice(currentPrice)}
            </span>
            <span className={`text-sm font-medium ${
              priceChange >= 0 ? 'text-emerald-400' : 'text-red-400'
            }`}>
              {priceChange >= 0 ? '+' : ''}{priceChangePercent.toFixed(2)}%
            </span>
          </div>
        </div>
        
        {/* Timeframe Selector */}
        <div className="flex items-center gap-1 bg-slate-900 rounded-lg p-1">
          {['1m', '5m', '15m', '1h', '4h', '1d'].map((tf) => (
            <button
              key={tf}
              className={`px-3 py-1 text-xs font-medium rounded-md transition-all ${
                tf === '1h'
                  ? 'bg-slate-700 text-slate-100'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {tf}
            </button>
          ))}
        </div>
      </div>

      {/* Chart */}
      <div style={{ height }}>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
            {showGrid && (
              <CartesianGrid 
                strokeDasharray="3 3" 
                stroke="#1e293b" 
                vertical={false}
              />
            )}
            <XAxis
              dataKey="timestamp"
              tickFormatter={formatDate}
              stroke="#475569"
              tick={{ fill: '#64748b', fontSize: 11 }}
              tickLine={false}
              axisLine={{ stroke: '#1e293b' }}
              minTickGap={30}
            />
            <YAxis
              domain={['auto', 'auto']}
              stroke="#475569"
              tick={{ fill: '#64748b', fontSize: 11 }}
              tickLine={false}
              axisLine={{ stroke: '#1e293b' }}
              tickFormatter={formatPrice}
              width={70}
            />
            <Tooltip content={<CustomTooltip />} />
            
            {/* Price Area */}
            <Area
              type="monotone"
              dataKey="close"
              stroke="#3b82f6"
              strokeWidth={2}
              fill="url(#priceGradient)"
              dot={false}
              activeDot={{ r: 4, fill: '#3b82f6', stroke: '#fff', strokeWidth: 2 }}
            />
            
            {/* Moving Average 20 */}
            <Line
              type="monotone"
              dataKey="ma20"
              stroke="#f59e0b"
              strokeWidth={1.5}
              dot={false}
              strokeDasharray="5 5"
            />
            
            {/* Moving Average 50 */}
            <Line
              type="monotone"
              dataKey="ma50"
              stroke="#8b5cf6"
              strokeWidth={1.5}
              dot={false}
              strokeDasharray="5 5"
            />
            
            {/* Current Price Line */}
            <ReferenceLine
              y={currentPrice}
              stroke="#3b82f6"
              strokeDasharray="3 3"
              strokeOpacity={0.5}
            />
            
            <defs>
              <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
              </linearGradient>
            </defs>
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Legend */}
      <div className="flex items-center gap-4 mt-3 text-xs">
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-0.5 bg-blue-500" />
          <span className="text-slate-400">Price</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-0.5 bg-amber-500" style={{ borderStyle: 'dashed' }} />
          <span className="text-slate-400">MA20</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-0.5 bg-violet-500" style={{ borderStyle: 'dashed' }} />
          <span className="text-slate-400">MA50</span>
        </div>
      </div>
    </div>
  );
};
