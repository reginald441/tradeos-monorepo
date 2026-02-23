import React from 'react';
import {
  TrendingUp,
  TrendingDown,
  Wallet,
  Activity,
  BarChart3,
  Clock,
  ArrowUpRight,
  ArrowDownRight,
} from 'lucide-react';
import { useTradingStore } from '@/store/tradingStore';
import { Card, CardHeader, CardContent } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';

const PnLTooltip: React.FC<{
  active?: boolean;
  payload?: any[];
  label?: string;
}> = ({ active, payload }) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-slate-900 border border-slate-700 rounded-lg p-3 shadow-xl">
        <p className="text-sm font-mono text-blue-400">
          ${payload[0].value.toLocaleString('en-US', { minimumFractionDigits: 2 })}
        </p>
      </div>
    );
  }
  return null;
};

export const Dashboard: React.FC = () => {
  const portfolio = useTradingStore((state) => state.portfolio);
  const positions = useTradingStore((state) => state.positions);
  const trades = useTradingStore((state) => state.trades);
  const strategies = useTradingStore((state) => state.strategies);

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value);
  };

  // Generate PnL chart data
  const pnlData = React.useMemo(() => {
    const data = [];
    let value = portfolio?.totalValue || 100000;
    const now = new Date();
    for (let i = 30; i >= 0; i--) {
      const date = new Date(now.getTime() - i * 24 * 60 * 60 * 1000);
      const change = (Math.random() - 0.45) * 2000;
      value += change;
      data.push({
        date: date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
        value: value,
      });
    }
    return data;
  }, [portfolio]);

  const activeStrategies = strategies.filter(s => s.status === 'active').length;

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-100">Dashboard</h1>
          <p className="text-slate-400 mt-1">Overview of your trading portfolio</p>
        </div>
        <div className="flex items-center gap-3">
          <Button variant="secondary" leftIcon={<BarChart3 className="w-4 h-4" />}>
            Export Report
          </Button>
          <Button leftIcon={<Activity className="w-4 h-4" />}>
            Live Trading
          </Button>
        </div>
      </div>

      {/* Portfolio Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-5">
            <div className="flex items-start justify-between">
              <div>
                <p className="text-xs text-slate-500 uppercase tracking-wider">Total Balance</p>
                <p className="text-2xl font-mono font-bold text-slate-100 mt-1">
                  {formatCurrency(portfolio?.totalValue || 0)}
                </p>
              </div>
              <div className="w-10 h-10 rounded-lg bg-blue-500/10 flex items-center justify-center">
                <Wallet className="w-5 h-5 text-blue-400" />
              </div>
            </div>
            <div className="flex items-center gap-2 mt-3">
              <span className="text-xs text-slate-500">Available:</span>
              <span className="text-sm font-mono text-slate-300">
                {formatCurrency(portfolio?.availableBalance || 0)}
              </span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-5">
            <div className="flex items-start justify-between">
              <div>
                <p className="text-xs text-slate-500 uppercase tracking-wider">Unrealized P&L</p>
                <p className={`text-2xl font-mono font-bold mt-1 ${
                  (portfolio?.unrealizedPnl || 0) >= 0 ? 'text-emerald-400' : 'text-red-400'
                }`}>
                  {(portfolio?.unrealizedPnl || 0) >= 0 ? '+' : ''}
                  {formatCurrency(portfolio?.unrealizedPnl || 0)}
                </p>
              </div>
              <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
                (portfolio?.unrealizedPnl || 0) >= 0 ? 'bg-emerald-500/10' : 'bg-red-500/10'
              }`}>
                {(portfolio?.unrealizedPnl || 0) >= 0 ? (
                  <TrendingUp className="w-5 h-5 text-emerald-400" />
                ) : (
                  <TrendingDown className="w-5 h-5 text-red-400" />
                )}
              </div>
            </div>
            <div className="flex items-center gap-2 mt-3">
              <span className={`text-sm font-medium flex items-center gap-1 ${
                (portfolio?.unrealizedPnl || 0) >= 0 ? 'text-emerald-400' : 'text-red-400'
              }`}>
                {(portfolio?.unrealizedPnl || 0) >= 0 ? (
                  <ArrowUpRight className="w-3 h-3" />
                ) : (
                  <ArrowDownRight className="w-3 h-3" />
                )}
                {((portfolio?.unrealizedPnl || 0) / (portfolio?.totalValue || 1) * 100).toFixed(2)}%
              </span>
              <span className="text-xs text-slate-500">of portfolio</span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-5">
            <div className="flex items-start justify-between">
              <div>
                <p className="text-xs text-slate-500 uppercase tracking-wider">24h P&L</p>
                <p className={`text-2xl font-mono font-bold mt-1 ${
                  (portfolio?.realizedPnl24h || 0) >= 0 ? 'text-emerald-400' : 'text-red-400'
                }`}>
                  {(portfolio?.realizedPnl24h || 0) >= 0 ? '+' : ''}
                  {formatCurrency(portfolio?.realizedPnl24h || 0)}
                </p>
              </div>
              <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
                (portfolio?.realizedPnl24h || 0) >= 0 ? 'bg-emerald-500/10' : 'bg-red-500/10'
              }`}>
                {(portfolio?.realizedPnl24h || 0) >= 0 ? (
                  <TrendingUp className="w-5 h-5 text-emerald-400" />
                ) : (
                  <TrendingDown className="w-5 h-5 text-red-400" />
                )}
              </div>
            </div>
            <div className="flex items-center gap-2 mt-3">
              <span className="text-xs text-slate-500">7d:</span>
              <span className={`text-sm font-mono ${
                (portfolio?.realizedPnl7d || 0) >= 0 ? 'text-emerald-400' : 'text-red-400'
              }`}>
                {(portfolio?.realizedPnl7d || 0) >= 0 ? '+' : ''}
                {formatCurrency(portfolio?.realizedPnl7d || 0)}
              </span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-5">
            <div className="flex items-start justify-between">
              <div>
                <p className="text-xs text-slate-500 uppercase tracking-wider">Active Strategies</p>
                <p className="text-2xl font-mono font-bold text-slate-100 mt-1">
                  {activeStrategies}
                </p>
              </div>
              <div className="w-10 h-10 rounded-lg bg-violet-500/10 flex items-center justify-center">
                <Activity className="w-5 h-5 text-violet-400" />
              </div>
            </div>
            <div className="flex items-center gap-2 mt-3">
              <span className="text-xs text-slate-500">Total:</span>
              <span className="text-sm font-mono text-slate-300">{strategies.length}</span>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* PnL Chart */}
        <Card className="lg:col-span-2">
          <CardHeader 
            title="Portfolio Performance" 
            subtitle="30-day equity curve"
            action={
              <div className="flex items-center gap-2">
                <span className="text-xs text-slate-500">1M</span>
                <span className="text-xs text-slate-500">3M</span>
                <span className="text-xs text-slate-500">1Y</span>
                <span className="text-xs text-blue-400 font-medium">ALL</span>
              </div>
            }
          />
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={pnlData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                  <defs>
                    <linearGradient id="pnlGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                  <XAxis
                    dataKey="date"
                    stroke="#475569"
                    tick={{ fill: '#64748b', fontSize: 11 }}
                    tickLine={false}
                    axisLine={{ stroke: '#1e293b' }}
                  />
                  <YAxis
                    stroke="#475569"
                    tick={{ fill: '#64748b', fontSize: 11 }}
                    tickLine={false}
                    axisLine={{ stroke: '#1e293b' }}
                    tickFormatter={(value) => `$${(value / 1000).toFixed(0)}k`}
                    width={60}
                  />
                  <Tooltip content={<PnLTooltip />} />
                  <Area
                    type="monotone"
                    dataKey="value"
                    stroke="#3b82f6"
                    strokeWidth={2}
                    fill="url(#pnlGradient)"
                    dot={false}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Active Positions */}
        <Card>
          <CardHeader 
            title="Active Positions" 
            subtitle={`${positions.length} open positions`}
          />
          <CardContent>
            <div className="space-y-3">
              {positions.map((position) => (
                <div
                  key={position.id}
                  className="flex items-center justify-between p-3 bg-slate-900/50 rounded-lg border border-slate-800/50"
                >
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-slate-100">{position.symbol}</span>
                      <span className={`text-xs px-1.5 py-0.5 rounded ${
                        position.side === 'long' 
                          ? 'bg-emerald-500/20 text-emerald-400' 
                          : 'bg-red-500/20 text-red-400'
                      }`}>
                        {position.side.toUpperCase()}
                      </span>
                    </div>
                    <p className="text-xs text-slate-500 mt-0.5">
                      {position.size} @ {position.entryPrice.toFixed(2)}
                    </p>
                  </div>
                  <div className="text-right">
                    <p className={`font-mono font-medium ${
                      position.unrealizedPnl >= 0 ? 'text-emerald-400' : 'text-red-400'
                    }`}>
                      {position.unrealizedPnl >= 0 ? '+' : ''}
                      {formatCurrency(position.unrealizedPnl)}
                    </p>
                    <p className={`text-xs ${
                      position.unrealizedPnlPercent >= 0 ? 'text-emerald-400' : 'text-red-400'
                    }`}>
                      {position.unrealizedPnlPercent >= 0 ? '+' : ''}
                      {position.unrealizedPnlPercent.toFixed(2)}%
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Bottom Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Recent Trades */}
        <Card>
          <CardHeader 
            title="Recent Trades" 
            action={
              <Button variant="ghost" size="sm">
                View All
              </Button>
            }
          />
          <CardContent>
            <table className="w-full">
              <thead>
                <tr className="text-left border-b border-slate-800">
                  <th className="pb-2 text-xs font-medium text-slate-500 uppercase">Time</th>
                  <th className="pb-2 text-xs font-medium text-slate-500 uppercase">Symbol</th>
                  <th className="pb-2 text-xs font-medium text-slate-500 uppercase">Side</th>
                  <th className="pb-2 text-xs font-medium text-slate-500 uppercase text-right">P&L</th>
                </tr>
              </thead>
              <tbody>
                {trades.slice(0, 5).map((trade) => (
                  <tr key={trade.id} className="border-b border-slate-800/50 last:border-0">
                    <td className="py-3 text-sm text-slate-400">
                      <div className="flex items-center gap-1.5">
                        <Clock className="w-3 h-3" />
                        {new Date(trade.timestamp).toLocaleTimeString()}
                      </div>
                    </td>
                    <td className="py-3 text-sm font-medium text-slate-100">{trade.symbol}</td>
                    <td className="py-3">
                      <span className={`text-xs px-2 py-0.5 rounded ${
                        trade.side === 'buy' 
                          ? 'bg-emerald-500/20 text-emerald-400' 
                          : 'bg-red-500/20 text-red-400'
                      }`}>
                        {trade.side.toUpperCase()}
                      </span>
                    </td>
                    <td className="py-3 text-right">
                      {trade.pnl !== undefined && (
                        <span className={`font-mono text-sm ${
                          trade.pnl >= 0 ? 'text-emerald-400' : 'text-red-400'
                        }`}>
                          {trade.pnl >= 0 ? '+' : ''}
                          {formatCurrency(trade.pnl)}
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </CardContent>
        </Card>

        {/* Strategy Performance */}
        <Card>
          <CardHeader 
            title="Strategy Performance" 
            action={
              <Button variant="ghost" size="sm">
                Manage
              </Button>
            }
          />
          <CardContent>
            <div className="space-y-3">
              {strategies.map((strategy) => (
                <div
                  key={strategy.id}
                  className="flex items-center justify-between p-3 bg-slate-900/50 rounded-lg border border-slate-800/50"
                >
                  <div className="flex items-center gap-3">
                    <div className={`w-2 h-2 rounded-full ${
                      strategy.status === 'active' ? 'bg-emerald-400' : 'bg-amber-400'
                    }`} />
                    <div>
                      <p className="font-medium text-slate-100">{strategy.name}</p>
                      <p className="text-xs text-slate-500">{strategy.type}</p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className={`font-mono font-medium ${
                      strategy.performance.totalReturn >= 0 ? 'text-emerald-400' : 'text-red-400'
                    }`}>
                      {strategy.performance.totalReturn >= 0 ? '+' : ''}
                      {strategy.performance.totalReturn.toFixed(2)}%
                    </p>
                    <p className="text-xs text-slate-500">
                      {strategy.performance.totalTrades} trades
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};
