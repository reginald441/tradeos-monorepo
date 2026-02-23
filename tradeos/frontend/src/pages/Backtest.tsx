import React, { useState } from 'react';
import {
  Play,
  Calendar,
  Settings,
  BarChart3,
  TrendingUp,
  TrendingDown,
  Target,
  Activity,
  Download,
  History,
} from 'lucide-react';
import { useTradingStore } from '@/store/tradingStore';
import { Card, CardHeader, CardContent } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Input, Select } from '@/components/ui/Input';
import { EquityCurve } from '@/components/charts/EquityCurve';
import type { BacktestResult, EquityPoint } from '@/types';

// Generate mock backtest result
const generateMockBacktestResult = (): BacktestResult => {
  const equityCurve: EquityPoint[] = [];
  let value = 10000;
  let peak = value;
  
  const startDate = new Date('2023-01-01');
  const endDate = new Date('2023-12-31');
  const days = Math.floor((endDate.getTime() - startDate.getTime()) / (1000 * 60 * 60 * 24));
  
  for (let i = 0; i <= days; i += 7) {
    const date = new Date(startDate.getTime() + i * 24 * 60 * 60 * 1000);
    const change = (Math.random() - 0.42) * 500;
    value += change;
    if (value > peak) peak = value;
    const drawdown = ((peak - value) / peak) * 100;
    
    equityCurve.push({
      timestamp: date.toISOString(),
      value,
      drawdown,
    });
  }

  const finalValue = equityCurve[equityCurve.length - 1]?.value || 10000;
  const totalReturn = ((finalValue - 10000) / 10000) * 100;
  const maxDrawdown = Math.max(...equityCurve.map(e => e.drawdown));

  return {
    id: 'bt-1',
    strategyId: 'strat-1',
    startDate: startDate.toISOString(),
    endDate: endDate.toISOString(),
    initialCapital: 10000,
    finalCapital: finalValue,
    totalReturn,
    sharpeRatio: 1.85 + Math.random() * 0.5,
    maxDrawdown,
    winRate: 55 + Math.random() * 15,
    totalTrades: Math.floor(100 + Math.random() * 200),
    profitFactor: 1.5 + Math.random() * 0.8,
    equityCurve,
    trades: [],
    metrics: {
      avgTrade: totalReturn / (100 + Math.random() * 200),
      avgWin: 2.5 + Math.random() * 2,
      avgLoss: -1.5 - Math.random(),
      largestWin: 15 + Math.random() * 10,
      largestLoss: -8 - Math.random() * 5,
      avgTradeDuration: 24 + Math.random() * 48,
      maxConsecutiveWins: Math.floor(5 + Math.random() * 10),
      maxConsecutiveLosses: Math.floor(3 + Math.random() * 5),
    },
  };
};

const MetricCard: React.FC<{
  label: string;
  value: string | number;
  suffix?: string;
  positive?: boolean;
  negative?: boolean;
}> = ({ label, value, suffix = '', positive, negative }) => (
  <div className="bg-slate-900/50 rounded-lg p-4">
    <p className="text-xs text-slate-500 uppercase tracking-wider">{label}</p>
    <p className={`text-xl font-mono font-bold mt-1 ${
      positive ? 'text-emerald-400' : negative ? 'text-red-400' : 'text-slate-100'
    }`}>
      {value}{suffix}
    </p>
  </div>
);

export const Backtest: React.FC = () => {
  const strategies = useTradingStore((state) => state.strategies);
  const [isRunning, setIsRunning] = useState(false);
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [showHistory, setShowHistory] = useState(false);

  const [config, setConfig] = useState({
    strategyId: '',
    symbol: 'BTC-USD',
    timeframe: '1h',
    startDate: '2023-01-01',
    endDate: '2023-12-31',
    initialCapital: 10000,
  });

  const runBacktest = async () => {
    setIsRunning(true);
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 2000));
    setResult(generateMockBacktestResult());
    setIsRunning(false);
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-100">Backtest</h1>
          <p className="text-slate-400 mt-1">Test your strategies on historical data</p>
        </div>
        <div className="flex items-center gap-2">
          <Button 
            variant="secondary" 
            leftIcon={<History className="w-4 h-4" />}
            onClick={() => setShowHistory(!showHistory)}
          >
            History
          </Button>
          <Button 
            leftIcon={<Download className="w-4 h-4" />}
            disabled={!result}
          >
            Export
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Configuration Panel */}
        <Card>
          <CardHeader title="Configuration" />
          <CardContent>
            <div className="space-y-4">
              <Select
                label="Strategy"
                options={strategies.map(s => ({ value: s.id, label: s.name }))}
                value={config.strategyId}
                onChange={(e) => setConfig({ ...config, strategyId: e.target.value })}
              />

              <Select
                label="Symbol"
                options={[
                  { value: 'BTC-USD', label: 'BTC-USD' },
                  { value: 'ETH-USD', label: 'ETH-USD' },
                  { value: 'SOL-USD', label: 'SOL-USD' },
                ]}
                value={config.symbol}
                onChange={(e) => setConfig({ ...config, symbol: e.target.value })}
              />

              <Select
                label="Timeframe"
                options={[
                  { value: '1m', label: '1 Minute' },
                  { value: '5m', label: '5 Minutes' },
                  { value: '15m', label: '15 Minutes' },
                  { value: '1h', label: '1 Hour' },
                  { value: '4h', label: '4 Hours' },
                  { value: '1d', label: '1 Day' },
                ]}
                value={config.timeframe}
                onChange={(e) => setConfig({ ...config, timeframe: e.target.value })}
              />

              <div className="grid grid-cols-2 gap-4">
                <Input
                  label="Start Date"
                  type="date"
                  value={config.startDate}
                  onChange={(e) => setConfig({ ...config, startDate: e.target.value })}
                />
                <Input
                  label="End Date"
                  type="date"
                  value={config.endDate}
                  onChange={(e) => setConfig({ ...config, endDate: e.target.value })}
                />
              </div>

              <Input
                label="Initial Capital ($)"
                type="number"
                value={config.initialCapital}
                onChange={(e) => setConfig({ ...config, initialCapital: Number(e.target.value) })}
              />

              <Button
                className="w-full"
                size="lg"
                isLoading={isRunning}
                leftIcon={<Play className="w-4 h-4" />}
                onClick={runBacktest}
                disabled={!config.strategyId}
              >
                Run Backtest
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Results Panel */}
        <div className="lg:col-span-2 space-y-4">
          {result ? (
            <>
              {/* Key Metrics */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <MetricCard 
                  label="Total Return" 
                  value={result.totalReturn >= 0 ? '+' : ''}
                  suffix={`${result.totalReturn.toFixed(2)}%`}
                  positive={result.totalReturn > 0}
                  negative={result.totalReturn < 0}
                />
                <MetricCard 
                  label="Sharpe Ratio" 
                  value={result.sharpeRatio.toFixed(2)} 
                />
                <MetricCard 
                  label="Max Drawdown" 
                  value="-"
                  suffix={`${result.maxDrawdown.toFixed(2)}%`}
                  negative
                />
                <MetricCard 
                  label="Win Rate" 
                  value={result.winRate.toFixed(1)}
                  suffix="%"
                />
              </div>

              {/* Equity Curve */}
              <Card>
                <CardHeader title="Equity Curve" />
                <CardContent>
                  <EquityCurve 
                    data={result.equityCurve} 
                    height={250}
                    initialCapital={result.initialCapital}
                  />
                </CardContent>
              </Card>

              {/* Detailed Metrics */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Card>
                  <CardHeader title="Trade Statistics" />
                  <CardContent>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-slate-500">Total Trades</span>
                        <span className="font-mono text-slate-100">{result.totalTrades}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-500">Profit Factor</span>
                        <span className="font-mono text-slate-100">{result.profitFactor.toFixed(2)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-500">Average Trade</span>
                        <span className={`font-mono ${result.metrics.avgTrade >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                          {result.metrics.avgTrade >= 0 ? '+' : ''}{result.metrics.avgTrade.toFixed(2)}%
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-500">Average Win</span>
                        <span className="font-mono text-emerald-400">+{result.metrics.avgWin.toFixed(2)}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-500">Average Loss</span>
                        <span className="font-mono text-red-400">{result.metrics.avgLoss.toFixed(2)}%</span>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader title="Performance Extremes" />
                  <CardContent>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-slate-500">Largest Win</span>
                        <span className="font-mono text-emerald-400">+{result.metrics.largestWin.toFixed(2)}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-500">Largest Loss</span>
                        <span className="font-mono text-red-400">{result.metrics.largestLoss.toFixed(2)}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-500">Max Consecutive Wins</span>
                        <span className="font-mono text-slate-100">{result.metrics.maxConsecutiveWins}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-500">Max Consecutive Losses</span>
                        <span className="font-mono text-slate-100">{result.metrics.maxConsecutiveLosses}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-500">Avg Trade Duration</span>
                        <span className="font-mono text-slate-100">{result.metrics.avgTradeDuration.toFixed(1)}h</span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </>
          ) : (
            <Card className="h-96 flex items-center justify-center">
              <div className="text-center">
                <BarChart3 className="w-16 h-16 text-slate-700 mx-auto mb-4" />
                <p className="text-slate-500">Run a backtest to see results</p>
                <p className="text-sm text-slate-600 mt-1">Configure parameters and click Run Backtest</p>
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
};
