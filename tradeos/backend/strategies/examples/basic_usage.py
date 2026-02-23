"""
TradeOS Strategy Engine - Basic Usage Examples
==============================================
Demonstrates how to use the strategy engine components.

Author: TradeOS Team
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import TradeOS strategy components
import sys
sys.path.insert(0, '/mnt/okcomputer/output/tradeos/backend')

from strategies import (
    # Base components
    EMACrossoverStrategy, RSIMeanReversionStrategy, BreakoutStrategy,
    SignalType, ATRPositionSizer, PercentOfEquitySizer,
    
    # Engine
    StrategyRunner, ExecutionMode, SignalAggregator,
    
    # Regime filter
    RegimeFilter, MarketRegime,
    
    # Optimizer
    WalkForwardOptimizer, ParameterSpace, OptimizationMethod, combined_objective,
    
    # Config
    StrategyConfig, create_ema_crossover_config, create_rsi_mean_reversion_config,
    
    # Indicators
    ema, rsi, atr, bollinger_bands
)


def generate_sample_data(n_bars: int = 1000, symbol: str = 'BTCUSDT') -> pd.DataFrame:
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    
    # Generate random walk price data
    returns = np.random.normal(0.0001, 0.02, n_bars)
    prices = 50000 * np.exp(np.cumsum(returns))
    
    # Generate OHLCV
    dates = pd.date_range(start='2023-01-01', periods=n_bars, freq='1h')
    
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, n_bars)),
        'high': prices * (1 + abs(np.random.normal(0, 0.01, n_bars))),
        'low': prices * (1 - abs(np.random.normal(0, 0.01, n_bars))),
        'close': prices,
        'volume': np.random.uniform(100, 1000, n_bars)
    }, index=dates)
    
    # Ensure OHLC consistency
    data['high'] = data[['open', 'close', 'high']].max(axis=1)
    data['low'] = data[['open', 'close', 'low']].min(axis=1)
    
    return data


def example_1_basic_strategy():
    """Example 1: Create and run a basic EMA crossover strategy."""
    print("=" * 60)
    print("Example 1: Basic EMA Crossover Strategy")
    print("=" * 60)
    
    # Generate sample data
    data = generate_sample_data(n_bars=500)
    
    # Create strategy with custom parameters
    strategy = EMACrossoverStrategy(
        name="EMA_12_26",
        params={
            'fast_period': 12,
            'slow_period': 26,
            'use_volume': True,
            'volume_threshold': 1.2,
            'stop_loss_atr': 2.0,
            'take_profit_atr': 3.0
        },
        position_sizer=ATRPositionSizer(risk_per_trade=0.02, atr_multiple=2.0)
    )
    
    # Run backtest
    result = strategy.backtest(
        data,
        initial_capital=100000.0,
        commission=0.001,
        slippage=0.0005
    )
    
    # Print results
    print(f"\nBacktest Results:")
    print(f"  Total Return: {result.total_return:.2%}")
    print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"  Max Drawdown: {result.max_drawdown:.2%}")
    print(f"  Win Rate: {result.win_rate:.2%}")
    print(f"  Total Trades: {result.total_trades}")
    print(f"  Profit Factor: {result.profit_factor:.2f}")
    
    return result


def example_2_multiple_strategies():
    """Example 2: Run multiple strategies with the StrategyRunner."""
    print("\n" + "=" * 60)
    print("Example 2: Multiple Strategies with StrategyRunner")
    print("=" * 60)
    
    # Generate sample data
    data = generate_sample_data(n_bars=500)
    
    # Create strategy runner
    runner = StrategyRunner(
        initial_capital=100000.0,
        mode=ExecutionMode.BACKTEST,
        signal_aggregator=SignalAggregator.WEIGHTED_CONFIDENCE,
        use_regime_filter=True
    )
    
    # Create and register strategies
    ema_strategy = EMACrossoverStrategy(
        name="EMA_Crossover",
        params={'fast_period': 12, 'slow_period': 26}
    )
    
    rsi_strategy = RSIMeanReversionStrategy(
        name="RSI_MeanReversion",
        params={'rsi_period': 14, 'oversold': 30, 'overbought': 70}
    )
    
    breakout_strategy = BreakoutStrategy(
        name="Breakout",
        params={'lookback_period': 20, 'volume_confirm': True}
    )
    
    # Register strategies with weights
    runner.register_strategy("ema", ema_strategy, weight=0.4)
    runner.register_strategy("rsi", rsi_strategy, weight=0.3)
    runner.register_strategy("breakout", breakout_strategy, weight=0.3)
    
    # Run backtest for all strategies
    results = runner.run_backtest(data, symbol="BTCUSDT")
    
    # Print results
    print(f"\nIndividual Strategy Results:")
    for name, result in results.items():
        print(f"\n  {name}:")
        print(f"    Return: {result.total_return:.2%}")
        print(f"    Sharpe: {result.sharpe_ratio:.2f}")
        print(f"    Trades: {result.total_trades}")
    
    return results


def example_3_regime_filter():
    """Example 3: Using the regime filter."""
    print("\n" + "=" * 60)
    print("Example 3: Market Regime Filter")
    print("=" * 60)
    
    # Generate sample data
    data = generate_sample_data(n_bars=500)
    
    # Create regime filter
    regime_filter = RegimeFilter(
        adx_period=14,
        adx_threshold=25.0,
        ema_period=50
    )
    
    # Update filter with data
    for i in range(50, len(data)):
        window = data.iloc[:i+1]
        state = regime_filter.update(window)
    
    # Print current regime
    print(f"\nCurrent Market Regime:")
    print(f"  Regime: {state.regime.value}")
    print(f"  ADX: {state.adx:.2f}")
    print(f"  Trend Direction: {'Up' if state.trend_direction == 1 else 'Down' if state.trend_direction == -1 else 'Neutral'}")
    print(f"  Trend Strength: {state.trend_strength:.2%}")
    print(f"  Volatility: {state.volatility:.2f}%")
    print(f"  Alignment Score: {state.alignment_score:.2f}")
    
    # Get regime recommendations
    print(f"\nStrategy Recommendations:")
    print(f"  Trend Following: {'Yes' if regime_filter.should_trade_trend_following() else 'No'}")
    print(f"  Mean Reversion: {'Yes' if regime_filter.should_trade_mean_reversion() else 'No'}")
    print(f"  Breakout: {'Yes' if regime_filter.should_trade_breakout() else 'No'}")
    print(f"  Recommended: {', '.join(regime_filter.get_recommended_strategies())}")
    
    return regime_filter


def example_4_walk_forward_optimization():
    """Example 4: Walk-forward optimization."""
    print("\n" + "=" * 60)
    print("Example 4: Walk-Forward Optimization")
    print("=" * 60)
    
    # Generate sample data
    data = generate_sample_data(n_bars=800)
    
    # Define parameter spaces
    param_spaces = [
        ParameterSpace(
            name='fast_period',
            param_type='int',
            min_value=5,
            max_value=20,
            step=5
        ),
        ParameterSpace(
            name='slow_period',
            param_type='int',
            min_value=20,
            max_value=50,
            step=10
        )
    ]
    
    # Create optimizer
    optimizer = WalkForwardOptimizer(
        strategy_class=EMACrossoverStrategy,
        param_spaces=param_spaces,
        objective_func=combined_objective,
        n_windows=3,
        train_pct=0.7,
        optimization_method=OptimizationMethod.GRID_SEARCH,
        n_jobs=1
    )
    
    # Run optimization
    print("\nRunning walk-forward optimization...")
    result = optimizer.run(data)
    
    # Print results
    metrics = result.calculate_metrics()
    print(f"\nOptimization Results:")
    print(f"  In-Sample Sharpe: {metrics['is_sharpe']:.3f}")
    print(f"  Out-of-Sample Sharpe: {metrics['oos_sharpe']:.3f}")
    print(f"  Sharpe Degradation: {metrics['sharpe_degradation']:.3f}")
    print(f"  In-Sample Return: {metrics['is_return']:.2%}")
    print(f"  Out-of-Sample Return: {metrics['oos_return']:.2%}")
    
    # Get robust parameters
    robust_params = optimizer.get_robust_params()
    print(f"\nRobust Parameters:")
    for key, value in robust_params.items():
        print(f"  {key}: {value}")
    
    return result


def example_5_strategy_configuration():
    """Example 5: Using strategy configuration."""
    print("\n" + "=" * 60)
    print("Example 5: Strategy Configuration")
    print("=" * 60)
    
    # Create configuration using helper
    config = create_ema_crossover_config(
        name="My_EMA_Strategy",
        fast_period=10,
        slow_period=30,
        symbols=['BTCUSDT', 'ETHUSDT']
    )
    
    print(f"\nStrategy Configuration:")
    print(f"  Name: {config.name}")
    print(f"  Type: {config.strategy_type.value}")
    print(f"  Symbols: {config.symbols}")
    print(f"  Timeframes: {config.timeframes}")
    print(f"  Parameters: {config.parameters.to_dict()}")
    
    # Export to JSON
    json_str = config.to_json()
    print(f"\nJSON Export:")
    print(json_str[:500] + "...")
    
    return config


def example_6_custom_strategy():
    """Example 6: Creating a custom strategy."""
    print("\n" + "=" * 60)
    print("Example 6: Custom Strategy Implementation")
    print("=" * 60)
    
    from strategies import BaseStrategy, Signal, SignalType
    from strategies.indicators import rsi, ema
    
    class CustomRSIEMAStrategy(BaseStrategy):
        """Custom strategy combining RSI and EMA."""
        
        def __init__(self, name, params=None, position_sizer=None):
            default_params = {
                'rsi_period': 14,
                'ema_period': 20,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'min_bars': 50
            }
            if params:
                default_params.update(params)
            
            super().__init__(name, default_params, position_sizer)
        
        def _precompute_indicators(self, data):
            """Precompute indicators."""
            self.rsi_values = rsi(data['close'], self.params['rsi_period']).values
            self.ema_values = ema(data['close'], self.params['ema_period']).values
        
        def generate_signal(self, data):
            """Generate trading signal."""
            if len(data) < self.params['min_bars']:
                return self._no_signal(data)
            
            # Calculate indicators
            rsi_vals = rsi(data['close'], self.params['rsi_period']).values
            ema_vals = ema(data['close'], self.params['ema_period']).values
            
            current_rsi = rsi_vals.iloc[-1]
            current_price = data['close'].iloc[-1]
            current_ema = ema_vals.iloc[-1]
            
            # Generate signal
            if current_rsi < self.params['rsi_oversold'] and current_price > current_ema:
                return Signal(
                    timestamp=data.index[-1],
                    symbol="BTCUSDT",
                    signal_type=SignalType.BUY,
                    price=current_price,
                    confidence=0.7,
                    strategy_name=self.name,
                    metadata={'rsi': current_rsi, 'ema': current_ema}
                )
            
            elif current_rsi > self.params['rsi_overbought'] and current_price < current_ema:
                return Signal(
                    timestamp=data.index[-1],
                    symbol="BTCUSDT",
                    signal_type=SignalType.SELL,
                    price=current_price,
                    confidence=0.7,
                    strategy_name=self.name,
                    metadata={'rsi': current_rsi, 'ema': current_ema}
                )
            
            return self._no_signal(data)
        
        def _no_signal(self, data):
            """Return no signal."""
            return Signal(
                timestamp=data.index[-1] if len(data) > 0 else datetime.now(),
                symbol="BTCUSDT",
                signal_type=SignalType.HOLD,
                price=data['close'].iloc[-1] if len(data) > 0 else 0,
                confidence=0.0,
                strategy_name=self.name
            )
    
    # Test custom strategy
    data = generate_sample_data(n_bars=500)
    
    strategy = CustomRSIEMAStrategy(
        name="Custom_RSI_EMA",
        params={'rsi_period': 14, 'ema_period': 20}
    )
    
    result = strategy.backtest(data)
    
    print(f"\nCustom Strategy Results:")
    print(f"  Total Return: {result.total_return:.2%}")
    print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"  Total Trades: {result.total_trades}")
    
    return result


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("TradeOS Strategy Engine - Usage Examples")
    print("=" * 60)
    
    # Run examples
    example_1_basic_strategy()
    example_2_multiple_strategies()
    example_3_regime_filter()
    example_4_walk_forward_optimization()
    example_5_strategy_configuration()
    example_6_custom_strategy()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
