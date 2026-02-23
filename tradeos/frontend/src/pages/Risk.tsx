import React, { useState } from 'react';
import {
  Shield,
  AlertTriangle,
  TrendingDown,
  Activity,
  BarChart3,
  Settings,
  Lock,
  Unlock,
  Info,
} from 'lucide-react';
import { useTradingStore } from '@/store/tradingStore';
import { Card, CardHeader, CardContent } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from 'recharts';

const exposureData = [
  { name: 'BTC', value: 45, color: '#f7931a' },
  { name: 'ETH', value: 30, color: '#627eea' },
  { name: 'SOL', value: 15, color: '#00ffa3' },
  { name: 'Other', value: 10, color: '#64748b' },
];

const varData = [
  { confidence: '95%', value: -8500, label: 'VaR 95%' },
  { confidence: '99%', value: -14200, label: 'VaR 99%' },
  { confidence: 'ES', value: -18500, label: 'Expected Shortfall' },
];

export const Risk: React.FC = () => {
  const riskMetrics = useTradingStore((state) => state.riskMetrics);
  const positions = useTradingStore((state) => state.positions);
  const [limits, setLimits] = useState({
    maxPositionSize: 50000,
    maxDailyLoss: 10000,
    maxTotalExposure: 200000,
    maxLeverage: 10,
  });

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
    }).format(value);
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-100">Risk Management</h1>
          <p className="text-slate-400 mt-1">Monitor and control your risk exposure</p>
        </div>
        <Button variant="secondary" leftIcon={<Settings className="w-4 h-4" />}>
          Configure Limits
        </Button>
      </div>

      {/* Risk Alerts */}
      <div className="bg-amber-500/10 border border-amber-500/30 rounded-lg p-4 flex items-start gap-3">
        <AlertTriangle className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
        <div>
          <p className="text-sm font-medium text-amber-400">Risk Alert</p>
          <p className="text-sm text-slate-400 mt-1">
            Your margin utilization is at 63.5%. Consider reducing exposure or adding collateral.
          </p>
        </div>
      </div>

      {/* Key Risk Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-start justify-between">
              <div>
                <p className="text-xs text-slate-500 uppercase">Total Exposure</p>
                <p className="text-2xl font-mono font-bold text-slate-100 mt-1">
                  {formatCurrency(riskMetrics?.totalExposure || 0)}
                </p>
              </div>
              <div className="w-10 h-10 rounded-lg bg-blue-500/10 flex items-center justify-center">
                <BarChart3 className="w-5 h-5 text-blue-400" />
              </div>
            </div>
            <div className="mt-3">
              <div className="flex justify-between text-xs mb-1">
                <span className="text-slate-500">Limit</span>
                <span className="text-slate-400">{formatCurrency(limits.maxTotalExposure)}</span>
              </div>
              <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-blue-500 rounded-full"
                  style={{ width: `${Math.min((riskMetrics?.totalExposure || 0) / limits.maxTotalExposure * 100, 100)}%` }}
                />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-start justify-between">
              <div>
                <p className="text-xs text-slate-500 uppercase">Margin Utilization</p>
                <p className="text-2xl font-mono font-bold text-slate-100 mt-1">
                  {riskMetrics?.marginUtilization.toFixed(1)}%
                </p>
              </div>
              <div className="w-10 h-10 rounded-lg bg-amber-500/10 flex items-center justify-center">
                <Activity className="w-5 h-5 text-amber-400" />
              </div>
            </div>
            <div className="mt-3">
              <div className="flex justify-between text-xs mb-1">
                <span className="text-slate-500">Warning at</span>
                <span className="text-slate-400">70%</span>
              </div>
              <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                <div 
                  className={`h-full rounded-full ${
                    (riskMetrics?.marginUtilization || 0) > 70 ? 'bg-red-500' : 'bg-amber-500'
                  }`}
                  style={{ width: `${riskMetrics?.marginUtilization || 0}%` }}
                />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-start justify-between">
              <div>
                <p className="text-xs text-slate-500 uppercase">VaR (95%)</p>
                <p className="text-2xl font-mono font-bold text-red-400 mt-1">
                  {formatCurrency(riskMetrics?.var95 || 0)}
                </p>
              </div>
              <div className="w-10 h-10 rounded-lg bg-red-500/10 flex items-center justify-center">
                <TrendingDown className="w-5 h-5 text-red-400" />
              </div>
            </div>
            <p className="text-xs text-slate-500 mt-3">
              Maximum expected loss with 95% confidence
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-start justify-between">
              <div>
                <p className="text-xs text-slate-500 uppercase">Portfolio Beta</p>
                <p className="text-2xl font-mono font-bold text-slate-100 mt-1">
                  {riskMetrics?.beta.toFixed(2)}
                </p>
              </div>
              <div className="w-10 h-10 rounded-lg bg-violet-500/10 flex items-center justify-center">
                <Shield className="w-5 h-5 text-violet-400" />
              </div>
            </div>
            <p className="text-xs text-slate-500 mt-3">
              Sensitivity to market movements
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Exposure Distribution */}
        <Card>
          <CardHeader title="Exposure Distribution" />
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={exposureData}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={80}
                    paddingAngle={5}
                    dataKey="value"
                  >
                    {exposureData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#0f172a', 
                      border: '1px solid #1e293b',
                      borderRadius: '8px'
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="flex justify-center gap-4 mt-4">
              {exposureData.map((item) => (
                <div key={item.name} className="flex items-center gap-2">
                  <div 
                    className="w-3 h-3 rounded-full" 
                    style={{ backgroundColor: item.color }}
                  />
                  <span className="text-sm text-slate-400">{item.name}</span>
                  <span className="text-sm font-mono text-slate-100">{item.value}%</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* VaR Analysis */}
        <Card>
          <CardHeader title="Value at Risk Analysis" />
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={varData} layout="vertical" margin={{ left: 80 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
                  <XAxis 
                    type="number" 
                    stroke="#475569"
                    tick={{ fill: '#64748b', fontSize: 11 }}
                    tickFormatter={(value) => `$${Math.abs(value / 1000).toFixed(0)}k`}
                  />
                  <YAxis 
                    type="category" 
                    dataKey="label"
                    stroke="#475569"
                    tick={{ fill: '#94a3b8', fontSize: 11 }}
                    width={80}
                  />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#0f172a', 
                      border: '1px solid #1e293b',
                      borderRadius: '8px'
                    }}
                    formatter={(value: number) => [formatCurrency(value), 'Potential Loss']}
                  />
                  <Bar dataKey="value" fill="#ef4444" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Risk Limits */}
      <Card>
        <CardHeader 
          title="Risk Limits Configuration" 
          subtitle="Set automatic limits to protect your portfolio"
        />
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div>
              <label className="flex items-center gap-2 mb-2">
                <span className="text-sm font-medium text-slate-300">Max Position Size</span>
                <Info className="w-4 h-4 text-slate-500" />
              </label>
              <Input
                type="number"
                value={limits.maxPositionSize}
                onChange={(e) => setLimits({ ...limits, maxPositionSize: Number(e.target.value) })}
                rightIcon={<span className="text-xs text-slate-500">USD</span>}
              />
              <p className="text-xs text-slate-500 mt-1">Maximum size per position</p>
            </div>

            <div>
              <label className="flex items-center gap-2 mb-2">
                <span className="text-sm font-medium text-slate-300">Max Daily Loss</span>
                <Info className="w-4 h-4 text-slate-500" />
              </label>
              <Input
                type="number"
                value={limits.maxDailyLoss}
                onChange={(e) => setLimits({ ...limits, maxDailyLoss: Number(e.target.value) })}
                rightIcon={<span className="text-xs text-slate-500">USD</span>}
              />
              <p className="text-xs text-slate-500 mt-1">Trading stops after this loss</p>
            </div>

            <div>
              <label className="flex items-center gap-2 mb-2">
                <span className="text-sm font-medium text-slate-300">Max Total Exposure</span>
                <Info className="w-4 h-4 text-slate-500" />
              </label>
              <Input
                type="number"
                value={limits.maxTotalExposure}
                onChange={(e) => setLimits({ ...limits, maxTotalExposure: Number(e.target.value) })}
                rightIcon={<span className="text-xs text-slate-500">USD</span>}
              />
              <p className="text-xs text-slate-500 mt-1">Maximum portfolio exposure</p>
            </div>

            <div>
              <label className="flex items-center gap-2 mb-2">
                <span className="text-sm font-medium text-slate-300">Max Leverage</span>
                <Info className="w-4 h-4 text-slate-500" />
              </label>
              <Input
                type="number"
                value={limits.maxLeverage}
                onChange={(e) => setLimits({ ...limits, maxLeverage: Number(e.target.value) })}
                rightIcon={<span className="text-xs text-slate-500">x</span>}
              />
              <p className="text-xs text-slate-500 mt-1">Maximum allowed leverage</p>
            </div>
          </div>

          <div className="flex justify-end gap-3 mt-6">
            <Button variant="secondary">Reset to Defaults</Button>
            <Button leftIcon={<Lock className="w-4 h-4" />}>Save Limits</Button>
          </div>
        </CardContent>
      </Card>

      {/* Position Risk Table */}
      <Card>
        <CardHeader title="Position Risk Analysis" />
        <CardContent>
          <table className="w-full data-table">
            <thead>
              <tr>
                <th>Symbol</th>
                <th>Size</th>
                <th>Notional</th>
                <th>Leverage</th>
                <th>Liq. Price</th>
                <th>Distance to Liq</th>
                <th>Risk Score</th>
              </tr>
            </thead>
            <tbody>
              {positions.map((position) => {
                const notional = position.size * position.markPrice;
                const liqDistance = position.liquidationPrice 
                  ? Math.abs((position.markPrice - position.liquidationPrice) / position.markPrice * 100)
                  : 0;
                const riskScore = liqDistance < 10 ? 'High' : liqDistance < 25 ? 'Medium' : 'Low';
                
                return (
                  <tr key={position.id}>
                    <td className="font-medium">{position.symbol}</td>
                    <td className="font-mono">{position.size}</td>
                    <td className="font-mono">{formatCurrency(notional)}</td>
                    <td className="font-mono">{position.leverage}x</td>
                    <td className="font-mono">${position.liquidationPrice?.toFixed(2) || 'N/A'}</td>
                    <td className="font-mono">{liqDistance.toFixed(2)}%</td>
                    <td>
                      <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                        riskScore === 'High' ? 'bg-red-500/20 text-red-400' :
                        riskScore === 'Medium' ? 'bg-amber-500/20 text-amber-400' :
                        'bg-emerald-500/20 text-emerald-400'
                      }`}>
                        {riskScore}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </CardContent>
      </Card>
    </div>
  );
};
