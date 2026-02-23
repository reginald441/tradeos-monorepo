import React, { useState } from 'react';
import {
  ArrowUp,
  ArrowDown,
  Buy,
  Sell,
  Wallet,
  TrendingUp,
  TrendingDown,
  Clock,
  BarChart3,
} from 'lucide-react';
import { useTradingStore } from '@/store/tradingStore';
import { Card, CardHeader, CardContent } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Input, Select } from '@/components/ui/Input';
import { PriceChart } from '@/components/charts/PriceChart';

const OrderBook: React.FC = () => {
  const generateOrders = (side: 'bid' | 'ask', count: number) => {
    const basePrice = 45120;
    return Array.from({ length: count }, (_, i) => ({
      price: side === 'bid' ? basePrice - i * 10 : basePrice + (i + 1) * 10,
      size: Math.random() * 2 + 0.1,
      total: 0,
    })).map((order, i, arr) => ({
      ...order,
      total: arr.slice(0, i + 1).reduce((sum, o) => sum + o.size, 0),
    }));
  };

  const bids = generateOrders('bid', 8);
  const asks = generateOrders('ask', 8);

  return (
    <div className="space-y-2">
      {/* Asks */}
      <div className="space-y-0.5">
        {asks.reverse().map((ask, i) => (
          <div key={i} className="flex items-center text-xs">
            <span className="w-20 text-red-400 font-mono">{ask.price.toFixed(2)}</span>
            <span className="w-16 text-slate-400 font-mono text-right">{ask.size.toFixed(4)}</span>
            <div className="flex-1 ml-2">
              <div 
                className="h-1.5 bg-red-500/20 rounded"
                style={{ width: `${(ask.total / 10) * 100}%` }}
              />
            </div>
          </div>
        ))}
      </div>

      {/* Spread */}
      <div className="py-2 border-y border-slate-800 text-center">
        <span className="text-sm font-mono text-slate-300">45120.50</span>
        <span className="text-xs text-slate-500 ml-2">Spread: 0.05</span>
      </div>

      {/* Bids */}
      <div className="space-y-0.5">
        {bids.map((bid, i) => (
          <div key={i} className="flex items-center text-xs">
            <span className="w-20 text-emerald-400 font-mono">{bid.price.toFixed(2)}</span>
            <span className="w-16 text-slate-400 font-mono text-right">{bid.size.toFixed(4)}</span>
            <div className="flex-1 ml-2">
              <div 
                className="h-1.5 bg-emerald-500/20 rounded"
                style={{ width: `${(bid.total / 10) * 100}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export const Trading: React.FC = () => {
  const candles = useTradingStore((state) => state.candles);
  const selectedSymbol = useTradingStore((state) => state.selectedSymbol);
  const positions = useTradingStore((state) => state.positions);
  const setSelectedSymbol = useTradingStore((state) => state.setSelectedSymbol);
  
  const [orderSide, setOrderSide] = useState<'buy' | 'sell'>('buy');
  const [orderType, setOrderType] = useState<'market' | 'limit'>('market');
  const [orderSize, setOrderSize] = useState('');
  const [orderPrice, setOrderPrice] = useState('');

  const symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'AVAX-USD', 'ARB-USD'];

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
    }).format(value);
  };

  const currentPosition = positions.find(p => p.symbol === selectedSymbol);

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-100">Trading</h1>
          <p className="text-slate-400 mt-1">Execute trades and monitor positions</p>
        </div>
        <div className="flex items-center gap-2">
          {symbols.map((symbol) => (
            <button
              key={symbol}
              onClick={() => setSelectedSymbol(symbol)}
              className={`px-3 py-1.5 text-sm font-medium rounded-lg transition-all ${
                selectedSymbol === symbol
                  ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30'
                  : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
              }`}
            >
              {symbol.split('-')[0]}
            </button>
          ))}
        </div>
      </div>

      {/* Main Trading Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Chart */}
        <div className="lg:col-span-3">
          <Card className="h-full">
            <CardContent className="p-4">
              <PriceChart 
                data={candles[selectedSymbol] || []} 
                height={450}
                symbol={selectedSymbol}
              />
            </CardContent>
          </Card>
        </div>

        {/* Order Panel */}
        <div className="space-y-4">
          {/* Order Form */}
          <Card>
            <CardHeader title="Place Order" />
            <CardContent>
              {/* Buy/Sell Toggle */}
              <div className="flex gap-2 mb-4">
                <button
                  onClick={() => setOrderSide('buy')}
                  className={`flex-1 py-2 rounded-lg font-medium text-sm transition-all ${
                    orderSide === 'buy'
                      ? 'bg-emerald-500 text-white'
                      : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
                  }`}
                >
                  Buy / Long
                </button>
                <button
                  onClick={() => setOrderSide('sell')}
                  className={`flex-1 py-2 rounded-lg font-medium text-sm transition-all ${
                    orderSide === 'sell'
                      ? 'bg-red-500 text-white'
                      : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
                  }`}
                >
                  Sell / Short
                </button>
              </div>

              {/* Order Type */}
              <div className="flex gap-2 mb-4">
                <button
                  onClick={() => setOrderType('market')}
                  className={`flex-1 py-1.5 rounded text-xs font-medium transition-all ${
                    orderType === 'market'
                      ? 'bg-blue-500/20 text-blue-400'
                      : 'bg-slate-800 text-slate-400'
                  }`}
                >
                  Market
                </button>
                <button
                  onClick={() => setOrderType('limit')}
                  className={`flex-1 py-1.5 rounded text-xs font-medium transition-all ${
                    orderType === 'limit'
                      ? 'bg-blue-500/20 text-blue-400'
                      : 'bg-slate-800 text-slate-400'
                  }`}
                >
                  Limit
                </button>
              </div>

              <div className="space-y-3">
                <Input
                  label="Size"
                  type="number"
                  value={orderSize}
                  onChange={(e) => setOrderSize(e.target.value)}
                  placeholder="0.00"
                  rightIcon={<span className="text-xs text-slate-500">{selectedSymbol.split('-')[0]}</span>}
                />
                
                {orderType === 'limit' && (
                  <Input
                    label="Price"
                    type="number"
                    value={orderPrice}
                    onChange={(e) => setOrderPrice(e.target.value)}
                    placeholder="0.00"
                    rightIcon={<span className="text-xs text-slate-500">USD</span>}
                  />
                )}

                <Select
                  label="Leverage"
                  options={[
                    { value: '1', label: '1x' },
                    { value: '2', label: '2x' },
                    { value: '3', label: '3x' },
                    { value: '5', label: '5x' },
                    { value: '10', label: '10x' },
                  ]}
                />

                {/* Order Summary */}
                <div className="bg-slate-900/50 rounded-lg p-3 space-y-1 text-sm">
                  <div className="flex justify-between">
                    <span className="text-slate-500">Margin Required</span>
                    <span className="font-mono text-slate-300">$0.00</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Fee (0.05%)</span>
                    <span className="font-mono text-slate-300">$0.00</span>
                  </div>
                  <div className="flex justify-between pt-1 border-t border-slate-800">
                    <span className="text-slate-500">Total</span>
                    <span className="font-mono text-slate-100">$0.00</span>
                  </div>
                </div>

                <Button
                  className="w-full"
                  variant={orderSide === 'buy' ? 'success' : 'danger'}
                  size="lg"
                  leftIcon={orderSide === 'buy' ? <ArrowUp className="w-4 h-4" /> : <ArrowDown className="w-4 h-4" />}
                >
                  {orderSide === 'buy' ? 'Buy / Long' : 'Sell / Short'}
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* Order Book */}
          <Card>
            <CardHeader title="Order Book" />
            <CardContent>
              <OrderBook />
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Position & Market Info */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Current Position */}
        <Card>
          <CardHeader title="Current Position" />
          <CardContent>
            {currentPosition ? (
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-slate-500">Side</span>
                  <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                    currentPosition.side === 'long' 
                      ? 'bg-emerald-500/20 text-emerald-400' 
                      : 'bg-red-500/20 text-red-400'
                  }`}>
                    {currentPosition.side.toUpperCase()}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-slate-500">Size</span>
                  <span className="font-mono text-slate-100">{currentPosition.size}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-slate-500">Entry Price</span>
                  <span className="font-mono text-slate-100">${currentPosition.entryPrice.toFixed(2)}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-slate-500">Mark Price</span>
                  <span className="font-mono text-slate-100">${currentPosition.markPrice.toFixed(2)}</span>
                </div>
                <div className="flex items-center justify-between pt-2 border-t border-slate-800">
                  <span className="text-slate-500">Unrealized P&L</span>
                  <span className={`font-mono font-medium ${
                    currentPosition.unrealizedPnl >= 0 ? 'text-emerald-400' : 'text-red-400'
                  }`}>
                    {currentPosition.unrealizedPnl >= 0 ? '+' : ''}
                    {formatCurrency(currentPosition.unrealizedPnl)}
                  </span>
                </div>
                <div className="flex gap-2 pt-2">
                  <Button variant="secondary" size="sm" className="flex-1">
                    Add Margin
                  </Button>
                  <Button variant="danger" size="sm" className="flex-1">
                    Close Position
                  </Button>
                </div>
              </div>
            ) : (
              <div className="text-center py-8">
                <p className="text-slate-500">No open position</p>
                <p className="text-xs text-slate-600 mt-1">Place an order to open a position</p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Market Info */}
        <Card>
          <CardHeader title="Market Info" />
          <CardContent>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-slate-500">24h High</span>
                <span className="font-mono text-slate-100">$46,250.00</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-slate-500">24h Low</span>
                <span className="font-mono text-slate-100">$44,180.00</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-slate-500">24h Volume</span>
                <span className="font-mono text-slate-100">$2.4B</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-slate-500">Open Interest</span>
                <span className="font-mono text-slate-100">$15.2B</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-slate-500">Funding Rate</span>
                <span className="font-mono text-emerald-400">+0.01%</span>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Recent Trades */}
        <Card>
          <CardHeader title="Recent Market Trades" />
          <CardContent>
            <div className="space-y-1">
              {Array.from({ length: 8 }, (_, i) => (
                <div key={i} className="flex items-center justify-between text-sm py-1">
                  <span className="text-slate-500 text-xs">
                    {new Date(Date.now() - i * 60000).toLocaleTimeString()}
                  </span>
                  <span className={`font-mono ${i % 3 === 0 ? 'text-red-400' : 'text-emerald-400'}`}>
                    ${(45120 + (Math.random() - 0.5) * 100).toFixed(2)}
                  </span>
                  <span className="font-mono text-slate-400">
                    {(Math.random() * 0.5).toFixed(4)}
                  </span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};
