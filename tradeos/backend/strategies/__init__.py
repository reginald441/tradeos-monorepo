"""
TradeOS Strategy Engine
=======================
Complete algorithmic trading strategy framework.

Modules:
    base_strategy: Abstract base class for all strategies
    indicators: Technical indicators library
    trend: Trend following strategies
    mean_reversion: Mean reversion strategies
    volatility: Volatility-based strategies
    liquidity: Liquidity detection strategies
    regime: Market regime classification
    optimizer: Walk-forward optimization
    engine: Strategy execution engine
    config: Configuration schemas

Example Usage:
    >>> from tradeos.backend.strategies import EMACrossoverStrategy, StrategyRunner
    >>> 
    >>> # Create strategy
    >>> strategy = EMACrossoverStrategy(
    ...     name="EMA_12_26",
    ...     params={'fast_period': 12, 'slow_period': 26}
    ... )
    >>> 
    >>> # Run backtest
    >>> result = strategy.backtest(data)
    >>> print(f"Return: {result.total_return:.2%}")
    >>> print(f"Sharpe: {result.sharpe_ratio:.2f}")
    >>> 
    >>> # Use strategy runner for multiple strategies
    >>> runner = StrategyRunner(initial_capital=100000)
    >>> runner.register_strategy("ema", strategy)
    >>> results = runner.run_backtest(data, symbol="BTCUSDT")

Author: TradeOS Team
Version: 1.0.0
"""

# Base strategy
from .base_strategy import (
    SignalType, OrderType, Signal, Position, Trade, BacktestResult,
    PositionSizer, FixedPositionSizer, PercentOfEquitySizer,
    KellySizer, ATRPositionSizer, BaseStrategy, MultiStrategy
)

# Indicators
from .indicators.technical import (
    sma, ema, wma, hull_ma, vwma, vwap,
    rsi, macd, stochastic, williams_r, cci, momentum,
    atr, bollinger_bands, keltner_channels, donchian_channels,
    historical_volatility, adx, supertrend, parabolic_sar,
    obv, volume_profile, vwap_with_std,
    fibonacci_retracement, fibonacci_extension, find_swing_points,
    detect_engulfing, detect_doji, detect_hammer,
    crossover, crossunder, rolling_zscore, calculate_slope, calculate_r2,
    IndicatorResult, TrendDirection
)

# Trend strategies
from .trend.trend_following import (
    EMACrossoverStrategy,
    ADXTrendStrategy,
    BreakoutStrategy,
    SuperTrendStrategy,
    MACDTrendStrategy
)

# Mean reversion strategies
from .mean_reversion.mr_strategies import (
    RSIMeanReversionStrategy,
    BollingerBandMeanReversionStrategy,
    StatisticalArbitrageStrategy,
    ZScoreMeanReversionStrategy,
    StochasticMeanReversionStrategy
)

# Volatility strategies
from .volatility.vol_strategies import (
    VolatilityRegime,
    VolatilityBreakoutStrategy,
    ATRChannelStrategy,
    VolatilityRegimeStrategy,
    VolatilityContractionStrategy,
    GapVolatilityStrategy
)

# Liquidity strategies
from .liquidity.liquidity_sweeps import (
    LiquidityLevel,
    OrderBlock,
    LiquiditySweepDetector,
    OrderBlockDetector,
    LiquiditySweepStrategy,
    OrderBlockStrategy
)

# Regime filter
from .regime.regime_filter import (
    MarketRegime,
    VolatilityRegime,
    RegimeState,
    RegimeFilter,
    MultiTimeframeRegimeFilter,
    RegimeBasedStrategyFilter
)

# Optimizer
from .optimizer.walk_forward import (
    OptimizationMethod,
    ParameterSpace,
    WFOResult,
    WalkForwardOptimizer,
    MonteCarloSimulator,
    StrategyOptimizer,
    sharpe_objective,
    return_objective,
    risk_adjusted_objective,
    profit_factor_objective,
    combined_objective
)

# Engine
from .engine.strategy_runner import (
    ExecutionMode,
    SignalAggregator,
    StrategyConfig,
    EngineState,
    StrategyRunner,
    EventDrivenRunner
)

# Config
from .config.strategy_config import (
    StrategyType,
    PositionSizingMethod,
    RiskManagementMethod,
    PositionSizingConfig,
    RiskManagementConfig,
    IndicatorConfig,
    FilterConfig,
    StrategyParameters,
    StrategyConfig,
    EngineConfig,
    create_ema_crossover_config,
    create_rsi_mean_reversion_config,
    create_breakout_config
)

__version__ = "1.0.0"
__author__ = "TradeOS Team"

__all__ = [
    # Version
    '__version__',
    '__author__',
    
    # Base Strategy
    'SignalType', 'OrderType', 'Signal', 'Position', 'Trade', 'BacktestResult',
    'PositionSizer', 'FixedPositionSizer', 'PercentOfEquitySizer',
    'KellySizer', 'ATRPositionSizer', 'BaseStrategy', 'MultiStrategy',
    
    # Indicators
    'sma', 'ema', 'wma', 'hull_ma', 'vwma', 'vwap',
    'rsi', 'macd', 'stochastic', 'williams_r', 'cci', 'momentum',
    'atr', 'bollinger_bands', 'keltner_channels', 'donchian_channels',
    'historical_volatility', 'adx', 'supertrend', 'parabolic_sar',
    'obv', 'volume_profile', 'vwap_with_std',
    'fibonacci_retracement', 'fibonacci_extension', 'find_swing_points',
    'detect_engulfing', 'detect_doji', 'detect_hammer',
    'crossover', 'crossunder', 'rolling_zscore', 'calculate_slope', 'calculate_r2',
    'IndicatorResult', 'TrendDirection',
    
    # Trend Strategies
    'EMACrossoverStrategy', 'ADXTrendStrategy', 'BreakoutStrategy',
    'SuperTrendStrategy', 'MACDTrendStrategy',
    
    # Mean Reversion Strategies
    'RSIMeanReversionStrategy', 'BollingerBandMeanReversionStrategy',
    'StatisticalArbitrageStrategy', 'ZScoreMeanReversionStrategy',
    'StochasticMeanReversionStrategy',
    
    # Volatility Strategies
    'VolatilityRegime', 'VolatilityBreakoutStrategy', 'ATRChannelStrategy',
    'VolatilityRegimeStrategy', 'VolatilityContractionStrategy', 'GapVolatilityStrategy',
    
    # Liquidity Strategies
    'LiquidityLevel', 'OrderBlock', 'LiquiditySweepDetector', 'OrderBlockDetector',
    'LiquiditySweepStrategy', 'OrderBlockStrategy',
    
    # Regime Filter
    'MarketRegime', 'VolatilityRegime', 'RegimeState', 'RegimeFilter',
    'MultiTimeframeRegimeFilter', 'RegimeBasedStrategyFilter',
    
    # Optimizer
    'OptimizationMethod', 'ParameterSpace', 'WFOResult', 'WalkForwardOptimizer',
    'MonteCarloSimulator', 'StrategyOptimizer',
    'sharpe_objective', 'return_objective', 'risk_adjusted_objective',
    'profit_factor_objective', 'combined_objective',
    
    # Engine
    'ExecutionMode', 'SignalAggregator', 'StrategyConfig', 'EngineState',
    'StrategyRunner', 'EventDrivenRunner',
    
    # Config
    'StrategyType', 'PositionSizingMethod', 'RiskManagementMethod',
    'PositionSizingConfig', 'RiskManagementConfig', 'IndicatorConfig',
    'FilterConfig', 'StrategyParameters', 'StrategyConfig', 'EngineConfig',
    'create_ema_crossover_config', 'create_rsi_mean_reversion_config', 'create_breakout_config'
]
