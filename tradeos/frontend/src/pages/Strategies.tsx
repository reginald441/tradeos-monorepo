import React, { useState } from 'react';
import {
  Plus,
  Play,
  Pause,
  Settings,
  Trash2,
  TrendingUp,
  Activity,
  BarChart3,
  Zap,
  ChevronDown,
  ChevronUp,
  Edit3,
} from 'lucide-react';
import { useTradingStore } from '@/store/tradingStore';
import { Card, CardHeader, CardContent } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Input, Select } from '@/components/ui/Input';
import type { Strategy } from '@/types';

const StrategyCard: React.FC<{
  strategy: Strategy;
  onToggle: () => void;
  onEdit: () => void;
  onDelete: () => void;
  isExpanded: boolean;
  onToggleExpand: () => void;
}> = ({ strategy, onToggle, onEdit, onDelete, isExpanded, onToggleExpand }) => {
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
        return 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30';
      case 'paused':
        return 'bg-amber-500/20 text-amber-400 border-amber-500/30';
      default:
        return 'bg-slate-700/50 text-slate-400 border-slate-600/30';
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'trend':
        return <TrendingUp className="w-4 h-4" />;
      case 'mean_reversion':
        return <Activity className="w-4 h-4" />;
      case 'arbitrage':
        return <BarChart3 className="w-4 h-4" />;
      case 'market_making':
        return <Zap className="w-4 h-4" />;
      default:
        return <Activity className="w-4 h-4" />;
    }
  };

  return (
    <div className="bg-slate-900/60 border border-slate-800/60 rounded-xl overflow-hidden">
      {/* Header */}
      <div className="p-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-slate-800 flex items-center justify-center text-slate-400">
            {getTypeIcon(strategy.type)}
          </div>
          <div>
            <h3 className="font-semibold text-slate-100">{strategy.name}</h3>
            <p className="text-xs text-slate-500">{strategy.config.symbol} • {strategy.config.timeframe}</p>
          </div>
        </div>
        
        <div className="flex items-center gap-2">
          <span className={`status-badge ${getStatusColor(strategy.status)}`}>
            {strategy.status}
          </span>
          <button
            onClick={onToggle}
            className={`p-2 rounded-lg transition-colors ${
              strategy.status === 'active'
                ? 'bg-emerald-500/20 text-emerald-400 hover:bg-emerald-500/30'
                : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
            }`}
          >
            {strategy.status === 'active' ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
          </button>
          <button
            onClick={onEdit}
            className="p-2 rounded-lg bg-slate-800 text-slate-400 hover:bg-slate-700 transition-colors"
          >
            <Edit3 className="w-4 h-4" />
          </button>
          <button
            onClick={onDelete}
            className="p-2 rounded-lg bg-slate-800 text-red-400 hover:bg-red-500/20 transition-colors"
          >
            <Trash2 className="w-4 h-4" />
          </button>
          <button
            onClick={onToggleExpand}
            className="p-2 rounded-lg bg-slate-800 text-slate-400 hover:bg-slate-700 transition-colors"
          >
            {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {/* Quick Stats */}
      <div className="px-4 pb-4 grid grid-cols-4 gap-4">
        <div>
          <p className="text-xs text-slate-500">Total Return</p>
          <p className={`font-mono font-medium ${
            strategy.performance.totalReturn >= 0 ? 'text-emerald-400' : 'text-red-400'
          }`}>
            {strategy.performance.totalReturn >= 0 ? '+' : ''}
            {strategy.performance.totalReturn.toFixed(2)}%
          </p>
        </div>
        <div>
          <p className="text-xs text-slate-500">Sharpe Ratio</p>
          <p className="font-mono font-medium text-slate-100">
            {strategy.performance.sharpeRatio.toFixed(2)}
          </p>
        </div>
        <div>
          <p className="text-xs text-slate-500">Max Drawdown</p>
          <p className="font-mono font-medium text-red-400">
            {strategy.performance.maxDrawdown.toFixed(2)}%
          </p>
        </div>
        <div>
          <p className="text-xs text-slate-500">Win Rate</p>
          <p className="font-mono font-medium text-slate-100">
            {strategy.performance.winRate.toFixed(1)}%
          </p>
        </div>
      </div>

      {/* Expanded Details */}
      {isExpanded && (
        <div className="border-t border-slate-800/50 p-4">
          <div className="grid grid-cols-2 gap-6">
            <div>
              <h4 className="text-sm font-medium text-slate-300 mb-3">Configuration</h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-slate-500">Position Size</span>
                  <span className="font-mono text-slate-300">${strategy.config.positionSize}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Max Positions</span>
                  <span className="font-mono text-slate-300">{strategy.config.maxPositions}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Leverage</span>
                  <span className="font-mono text-slate-300">{strategy.config.leverage}x</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Stop Loss</span>
                  <span className="font-mono text-slate-300">{strategy.config.stopLoss}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Take Profit</span>
                  <span className="font-mono text-slate-300">{strategy.config.takeProfit}%</span>
                </div>
              </div>
            </div>
            <div>
              <h4 className="text-sm font-medium text-slate-300 mb-3">Performance Metrics</h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-slate-500">Total Trades</span>
                  <span className="font-mono text-slate-300">{strategy.performance.totalTrades}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Profit Factor</span>
                  <span className="font-mono text-slate-300">{strategy.performance.profitFactor.toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Avg Trade</span>
                  <span className="font-mono text-slate-300">{strategy.performance.avgTrade.toFixed(2)}%</span>
                </div>
              </div>
            </div>
          </div>
          <p className="text-xs text-slate-500 mt-4">{strategy.description}</p>
        </div>
      )}
    </div>
  );
};

export const Strategies: React.FC = () => {
  const strategies = useTradingStore((state) => state.strategies);
  const toggleStrategy = useTradingStore((state) => state.toggleStrategy);
  const [expandedStrategy, setExpandedStrategy] = useState<string | null>(null);
  const [showCreateModal, setShowCreateModal] = useState(false);

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-100">Strategies</h1>
          <p className="text-slate-400 mt-1">Manage your trading strategies</p>
        </div>
        <Button 
          leftIcon={<Plus className="w-4 h-4" />}
          onClick={() => setShowCreateModal(true)}
        >
          New Strategy
        </Button>
      </div>

      {/* Stats Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <p className="text-xs text-slate-500 uppercase">Total Strategies</p>
            <p className="text-2xl font-bold text-slate-100">{strategies.length}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <p className="text-xs text-slate-500 uppercase">Active</p>
            <p className="text-2xl font-bold text-emerald-400">
              {strategies.filter(s => s.status === 'active').length}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <p className="text-xs text-slate-500 uppercase">Paused</p>
            <p className="text-2xl font-bold text-amber-400">
              {strategies.filter(s => s.status === 'paused').length}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <p className="text-xs text-slate-500 uppercase">Avg Return</p>
            <p className="text-2xl font-bold text-blue-400">
              {(strategies.reduce((acc, s) => acc + s.performance.totalReturn, 0) / strategies.length).toFixed(1)}%
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Strategy List */}
      <div className="space-y-4">
        {strategies.map((strategy) => (
          <StrategyCard
            key={strategy.id}
            strategy={strategy}
            onToggle={() => toggleStrategy(strategy.id)}
            onEdit={() => {}}
            onDelete={() => {}}
            isExpanded={expandedStrategy === strategy.id}
            onToggleExpand={() => setExpandedStrategy(
              expandedStrategy === strategy.id ? null : strategy.id
            )}
          />
        ))}
      </div>

      {/* Create Strategy Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 w-full max-w-lg max-h-[90vh] overflow-y-auto">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-bold text-slate-100">Create New Strategy</h2>
              <button
                onClick={() => setShowCreateModal(false)}
                className="text-slate-400 hover:text-slate-200"
              >
                ✕
              </button>
            </div>
            
            <form className="space-y-4">
              <Input label="Strategy Name" placeholder="e.g., BTC Trend Follower" />
              
              <div className="grid grid-cols-2 gap-4">
                <Select
                  label="Strategy Type"
                  options={[
                    { value: 'trend', label: 'Trend Following' },
                    { value: 'mean_reversion', label: 'Mean Reversion' },
                    { value: 'arbitrage', label: 'Arbitrage' },
                    { value: 'market_making', label: 'Market Making' },
                  ]}
                />
                <Select
                  label="Symbol"
                  options={[
                    { value: 'BTC-USD', label: 'BTC-USD' },
                    { value: 'ETH-USD', label: 'ETH-USD' },
                    { value: 'SOL-USD', label: 'SOL-USD' },
                  ]}
                />
              </div>
              
              <div className="grid grid-cols-2 gap-4">
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
                />
                <Input 
                  label="Position Size ($)" 
                  type="number" 
                  placeholder="1000" 
                />
              </div>
              
              <div className="grid grid-cols-3 gap-4">
                <Input label="Leverage" type="number" placeholder="2" />
                <Input label="Stop Loss (%)" type="number" placeholder="2" />
                <Input label="Take Profit (%)" type="number" placeholder="6" />
              </div>
              
              <div className="flex justify-end gap-3 pt-4">
                <Button variant="secondary" onClick={() => setShowCreateModal(false)}>
                  Cancel
                </Button>
                <Button leftIcon={<Play className="w-4 h-4" />}>
                  Create & Start
                </Button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};
