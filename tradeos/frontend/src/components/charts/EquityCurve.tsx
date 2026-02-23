import React from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import type { EquityPoint } from '@/types';

interface EquityCurveProps {
  data: EquityPoint[];
  height?: number;
  showDrawdown?: boolean;
  initialCapital?: number;
}

interface ChartDataPoint {
  timestamp: string;
  value: number;
  drawdown: number;
  return: number;
}

const CustomTooltip: React.FC<{
  active?: boolean;
  payload?: any[];
  label?: string;
}> = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    
    return (
      <div className="bg-slate-900 border border-slate-700 rounded-lg p-3 shadow-xl">
        <p className="text-xs text-slate-500 mb-2">
          {new Date(data.timestamp).toLocaleDateString()}
        </p>
        <div className="space-y-1 text-sm">
          <div className="flex justify-between gap-4">
            <span className="text-slate-400">Equity:</span>
            <span className="font-mono text-blue-400">
              ${data.value.toLocaleString('en-US', { minimumFractionDigits: 2 })}
            </span>
          </div>
          <div className="flex justify-between gap-4">
            <span className="text-slate-400">Return:</span>
            <span className={`font-mono ${data.return >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
              {data.return >= 0 ? '+' : ''}{data.return.toFixed(2)}%
            </span>
          </div>
          <div className="flex justify-between gap-4">
            <span className="text-slate-400">Drawdown:</span>
            <span className="font-mono text-red-400">
              {data.drawdown.toFixed(2)}%
            </span>
          </div>
        </div>
      </div>
    );
  }
  return null;
};

export const EquityCurve: React.FC<EquityCurveProps> = ({
  data,
  height = 300,
  showDrawdown = true,
  initialCapital = 10000,
}) => {
  const chartData: ChartDataPoint[] = React.useMemo(() => {
    return data.map((point) => ({
      ...point,
      return: ((point.value - initialCapital) / initialCapital) * 100,
    }));
  }, [data, initialCapital]);

  const finalValue = data[data.length - 1]?.value || initialCapital;
  const totalReturn = ((finalValue - initialCapital) / initialCapital) * 100;
  const maxDrawdown = Math.max(...data.map(d => d.drawdown));

  const formatDate = (timestamp: string) => {
    const date = new Date(timestamp);
    return `${date.getMonth() + 1}/${date.getDate()}`;
  };

  return (
    <div className="w-full">
      {/* Metrics Header */}
      <div className="flex items-center gap-6 mb-4">
        <div>
          <p className="text-xs text-slate-500 uppercase tracking-wider">Final Equity</p>
          <p className="text-lg font-mono font-bold text-slate-100">
            ${finalValue.toLocaleString('en-US', { minimumFractionDigits: 2 })}
          </p>
        </div>
        <div>
          <p className="text-xs text-slate-500 uppercase tracking-wider">Total Return</p>
          <p className={`text-lg font-mono font-bold ${
            totalReturn >= 0 ? 'text-emerald-400' : 'text-red-400'
          }`}>
            {totalReturn >= 0 ? '+' : ''}{totalReturn.toFixed(2)}%
          </p>
        </div>
        <div>
          <p className="text-xs text-slate-500 uppercase tracking-wider">Max Drawdown</p>
          <p className="text-lg font-mono font-bold text-red-400">
            -{maxDrawdown.toFixed(2)}%
          </p>
        </div>
      </div>

      {/* Chart */}
      <div style={{ height }}>
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
            <CartesianGrid 
              strokeDasharray="3 3" 
              stroke="#1e293b" 
              vertical={false}
            />
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
              tickFormatter={(value) => `$${(value / 1000).toFixed(1)}k`}
              width={60}
            />
            <Tooltip content={<CustomTooltip />} />
            
            {/* Initial Capital Reference */}
            <ReferenceLine
              y={initialCapital}
              stroke="#64748b"
              strokeDasharray="3 3"
              strokeOpacity={0.5}
            />
            
            {/* Equity Area */}
            <Area
              type="monotone"
              dataKey="value"
              stroke="#3b82f6"
              strokeWidth={2}
              fill="url(#equityGradient)"
              dot={false}
              activeDot={{ r: 4, fill: '#3b82f6', stroke: '#fff', strokeWidth: 2 }}
            />
            
            <defs>
              <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
              </linearGradient>
            </defs>
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Drawdown Chart (if enabled) */}
      {showDrawdown && (
        <div className="mt-4" style={{ height: 100 }}>
          <p className="text-xs text-slate-500 mb-2">Drawdown</p>
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData} margin={{ top: 0, right: 10, left: 0, bottom: 0 }}>
              <XAxis
                dataKey="timestamp"
                tickFormatter={formatDate}
                stroke="#475569"
                tick={{ fill: '#64748b', fontSize: 10 }}
                tickLine={false}
                axisLine={{ stroke: '#1e293b' }}
                minTickGap={30}
              />
              <YAxis
                domain={[0, 'auto']}
                stroke="#475569"
                tick={{ fill: '#64748b', fontSize: 10 }}
                tickLine={false}
                axisLine={{ stroke: '#1e293b' }}
                tickFormatter={(value) => `${value.toFixed(0)}%`}
                width={40}
                reversed
              />
              <Area
                type="monotone"
                dataKey="drawdown"
                stroke="#ef4444"
                strokeWidth={1}
                fill="url(#drawdownGradient)"
                dot={false}
              />
              <defs>
                <linearGradient id="drawdownGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#ef4444" stopOpacity={0.4} />
                  <stop offset="95%" stopColor="#ef4444" stopOpacity={0.1} />
                </linearGradient>
              </defs>
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
};
