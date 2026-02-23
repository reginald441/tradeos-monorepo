"""
TradeOS Strategy Engine - Main Runner
=====================================
Central engine for running and managing multiple trading strategies.
Handles signal aggregation, risk management, and execution coordination.

Author: TradeOS Team
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Callable, Tuple, Union
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
import json

from ..base_strategy import BaseStrategy, Signal, SignalType, Trade, BacktestResult, Position
from ..regime.regime_filter import RegimeFilter, RegimeBasedStrategyFilter

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Strategy execution modes."""
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


class SignalAggregator(Enum):
    """Signal aggregation methods."""
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_CONFIDENCE = "weighted_confidence"
    UNANIMOUS = "unanimous"
    ANY = "any"


@dataclass
class StrategyConfig:
    """Configuration for a running strategy."""
    strategy: BaseStrategy
    enabled: bool = True
    weight: float = 1.0
    max_position_pct: float = 1.0
    allowed_regimes: List[str] = field(default_factory=list)
    timeframes: List[str] = field(default_factory=lambda: ['1h'])
    symbols: List[str] = field(default_factory=list)


@dataclass
class EngineState:
    """Current state of the strategy engine."""
    is_running: bool = False
    mode: ExecutionMode = ExecutionMode.BACKTEST
    current_time: Optional[datetime] = None
    total_equity: float = 100000.0
    open_positions: Dict[str, Position] = field(default_factory=dict)
    daily_pnl: float = 0.0
    daily_trades: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'is_running': self.is_running,
            'mode': self.mode.value,
            'current_time': self.current_time.isoformat() if self.current_time else None,
            'total_equity': self.total_equity,
            'open_positions': len(self.open_positions),
            'daily_pnl': self.daily_pnl,
            'daily_trades': self.daily_trades
        }


class StrategyRunner:
    """
    Main strategy engine that runs all active strategies.
    
    Features:
    - Multi-strategy management
    - Signal aggregation
    - Regime-based filtering
    - Position tracking
    - Risk management hooks
    - Event-driven architecture
    """
    
    def __init__(self,
                 initial_capital: float = 100000.0,
                 mode: ExecutionMode = ExecutionMode.BACKTEST,
                 signal_aggregator: SignalAggregator = SignalAggregator.WEIGHTED_CONFIDENCE,
                 use_regime_filter: bool = True):
        """
        Initialize strategy runner.
        
        Args:
            initial_capital: Starting capital
            mode: Execution mode
            signal_aggregator: Method for aggregating signals
            use_regime_filter: Enable regime-based filtering
        """
        self.initial_capital = initial_capital
        self.total_equity = initial_capital
        self.mode = mode
        self.signal_aggregator = signal_aggregator
        self.use_regime_filter = use_regime_filter
        
        # Strategy management
        self.strategies: Dict[str, StrategyConfig] = {}
        self.strategy_signals: Dict[str, List[Signal]] = defaultdict(list)
        
        # Regime filtering
        self.regime_filter: Optional[RegimeFilter] = RegimeFilter() if use_regime_filter else None
        self.strategy_filter: Optional[RegimeBasedStrategyFilter] = None
        if use_regime_filter:
            self.strategy_filter = RegimeBasedStrategyFilter(self.regime_filter)
        
        # State tracking
        self.state = EngineState(
            is_running=False,
            mode=mode,
            total_equity=initial_capital
        )
        
        # Position tracking
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Trade] = []
        
        # Callbacks
        self.on_signal_callbacks: List[Callable[[Signal], None]] = []
        self.on_trade_callbacks: List[Callable[[Trade], None]] = []
        self.on_bar_callbacks: List[Callable[[pd.DataFrame], None]] = []
        
        # Threading
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._running = False
        
        logger.info(f"StrategyRunner initialized in {mode.value} mode")
    
    def register_strategy(self, 
                         name: str, 
                         strategy: BaseStrategy,
                         weight: float = 1.0,
                         max_position_pct: float = 1.0,
                         enabled: bool = True) -> None:
        """
        Register a strategy with the engine.
        
        Args:
            name: Strategy identifier
            strategy: Strategy instance
            weight: Signal weight for aggregation
            max_position_pct: Maximum position size as % of equity
            enabled: Whether strategy is enabled
        """
        with self._lock:
            self.strategies[name] = StrategyConfig(
                strategy=strategy,
                enabled=enabled,
                weight=weight,
                max_position_pct=max_position_pct
            )
            
            # Register callbacks
            strategy.add_signal_callback(self._on_strategy_signal)
            strategy.add_trade_callback(self._on_strategy_trade)
            
            logger.info(f"Registered strategy: {name} (weight={weight})")
    
    def unregister_strategy(self, name: str) -> None:
        """Unregister a strategy."""
        with self._lock:
            if name in self.strategies:
                del self.strategies[name]
                logger.info(f"Unregistered strategy: {name}")
    
    def enable_strategy(self, name: str) -> None:
        """Enable a strategy."""
        with self._lock:
            if name in self.strategies:
                self.strategies[name].enabled = True
                logger.info(f"Enabled strategy: {name}")
    
    def disable_strategy(self, name: str) -> None:
        """Disable a strategy."""
        with self._lock:
            if name in self.strategies:
                self.strategies[name].enabled = False
                logger.info(f"Disabled strategy: {name}")
    
    def set_strategy_weight(self, name: str, weight: float) -> None:
        """Set strategy weight."""
        with self._lock:
            if name in self.strategies:
                self.strategies[name].weight = weight
    
    def _on_strategy_signal(self, signal: Signal) -> None:
        """Handle signal from a strategy."""
        self.strategy_signals[signal.strategy_name].append(signal)
        
        for callback in self.on_signal_callbacks:
            try:
                callback(signal)
            except Exception as e:
                logger.error(f"Signal callback error: {e}")
    
    def _on_strategy_trade(self, trade: Trade) -> None:
        """Handle trade from a strategy."""
        self.trade_history.append(trade)
        
        for callback in self.on_trade_callbacks:
            try:
                callback(trade)
            except Exception as e:
                logger.error(f"Trade callback error: {e}")
    
    def aggregate_signals(self, 
                         signals: List[Signal],
                         symbol: str) -> Optional[Signal]:
        """
        Aggregate multiple signals into a single signal.
        
        Args:
            signals: List of signals to aggregate
            symbol: Trading symbol
            
        Returns:
            Aggregated signal or None
        """
        if not signals:
            return None
        
        # Filter by symbol
        symbol_signals = [s for s in signals if s.symbol == symbol]
        
        if not symbol_signals:
            return None
        
        if self.signal_aggregator == SignalAggregator.ANY:
            # Return first valid signal
            return symbol_signals[0]
        
        elif self.signal_aggregator == SignalAggregator.UNANIMOUS:
            # All signals must agree
            buy_count = sum(1 for s in symbol_signals if s.signal_type == SignalType.BUY)
            sell_count = sum(1 for s in symbol_signals if s.signal_type == SignalType.SELL)
            
            if buy_count == len(symbol_signals):
                return self._create_aggregated_signal(symbol_signals, SignalType.BUY)
            elif sell_count == len(symbol_signals):
                return self._create_aggregated_signal(symbol_signals, SignalType.SELL)
            
            return None
        
        elif self.signal_aggregator == SignalAggregator.MAJORITY_VOTE:
            # Majority wins
            buy_count = sum(1 for s in symbol_signals if s.signal_type == SignalType.BUY)
            sell_count = sum(1 for s in symbol_signals if s.signal_type == SignalType.SELL)
            
            if buy_count > sell_count:
                buy_signals = [s for s in symbol_signals if s.signal_type == SignalType.BUY]
                return self._create_aggregated_signal(buy_signals, SignalType.BUY)
            elif sell_count > buy_count:
                sell_signals = [s for s in symbol_signals if s.signal_type == SignalType.SELL]
                return self._create_aggregated_signal(sell_signals, SignalType.SELL)
            
            return None
        
        elif self.signal_aggregator == SignalAggregator.WEIGHTED_CONFIDENCE:
            # Weight by confidence and strategy weight
            buy_score = 0.0
            sell_score = 0.0
            
            for signal in symbol_signals:
                strategy_weight = 1.0
                for name, config in self.strategies.items():
                    if config.strategy.name == signal.strategy_name:
                        strategy_weight = config.weight
                        break
                
                weighted_confidence = signal.confidence * strategy_weight
                
                if signal.signal_type == SignalType.BUY:
                    buy_score += weighted_confidence
                elif signal.signal_type == SignalType.SELL:
                    sell_score += weighted_confidence
            
            if buy_score > sell_score and buy_score > 0.5:
                buy_signals = [s for s in symbol_signals if s.signal_type == SignalType.BUY]
                return self._create_aggregated_signal(buy_signals, SignalType.BUY, buy_score)
            elif sell_score > buy_score and sell_score > 0.5:
                sell_signals = [s for s in symbol_signals if s.signal_type == SignalType.SELL]
                return self._create_aggregated_signal(sell_signals, SignalType.SELL, sell_score)
            
            return None
        
        return None
    
    def _create_aggregated_signal(self,
                                  signals: List[Signal],
                                  signal_type: SignalType,
                                  confidence: Optional[float] = None) -> Signal:
        """Create aggregated signal from multiple signals."""
        if not signals:
            return None
        
        # Use most recent signal as base
        base_signal = max(signals, key=lambda s: s.timestamp)
        
        # Average confidence if not provided
        if confidence is None:
            confidence = np.mean([s.confidence for s in signals])
        
        # Merge metadata
        merged_metadata = {}
        for s in signals:
            merged_metadata.update(s.metadata)
        
        return Signal(
            timestamp=base_signal.timestamp,
            symbol=base_signal.symbol,
            signal_type=signal_type,
            price=base_signal.price,
            confidence=min(confidence, 1.0),
            metadata=merged_metadata,
            strategy_name="AGGREGATED",
            timeframe=base_signal.timeframe
        )
    
    def process_bar(self, 
                   data: pd.DataFrame,
                   symbol: str) -> List[Signal]:
        """
        Process a new bar of data through all strategies.
        
        Args:
            data: OHLCV DataFrame
            symbol: Trading symbol
            
        Returns:
            List of generated signals
        """
        with self._lock:
            signals = []
            
            # Update regime filter
            if self.regime_filter:
                self.regime_filter.update(data)
            
            # Process through each strategy
            for name, config in self.strategies.items():
                if not config.enabled:
                    continue
                
                # Check regime filter
                if self.strategy_filter:
                    strategy_type = getattr(config.strategy, 'strategy_type', 'unknown')
                    if not self.strategy_filter.should_run_strategy(strategy_type):
                        continue
                
                try:
                    signal = config.strategy.on_bar(data.iloc[-1])
                    
                    if signal and signal.is_valid():
                        signal.symbol = symbol
                        signals.append(signal)
                
                except Exception as e:
                    logger.error(f"Error processing bar for strategy {name}: {e}")
            
            # Aggregate signals
            aggregated = self.aggregate_signals(signals, symbol)
            if aggregated:
                signals.append(aggregated)
            
            # Update state
            self.state.current_time = data.index[-1]
            
            # Call callbacks
            for callback in self.on_bar_callbacks:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Bar callback error: {e}")
            
            return signals
    
    def run_backtest(self, 
                    data: pd.DataFrame,
                    symbol: str,
                    commission: float = 0.001,
                    slippage: float = 0.0) -> Dict[str, BacktestResult]:
        """
        Run backtest for all strategies.
        
        Args:
            data: Historical OHLCV data
            symbol: Trading symbol
            commission: Commission rate
            slippage: Slippage percentage
            
        Returns:
            Dictionary of strategy results
        """
        results = {}
        
        for name, config in self.strategies.items():
            if not config.enabled:
                continue
            
            logger.info(f"Running backtest for {name}...")
            
            try:
                result = config.strategy.backtest(
                    data,
                    initial_capital=self.initial_capital,
                    commission=commission,
                    slippage=slippage
                )
                
                results[name] = result
                
                logger.info(f"{name} backtest complete: "
                          f"Return={result.total_return:.2%}, "
                          f"Sharpe={result.sharpe_ratio:.2f}")
            
            except Exception as e:
                logger.error(f"Backtest error for {name}: {e}")
        
        return results
    
    def get_portfolio_backtest(self,
                               data: pd.DataFrame,
                               symbol: str,
                               commission: float = 0.001,
                               slippage: float = 0.0,
                               rebalance_freq: str = 'D') -> BacktestResult:
        """
        Run portfolio-level backtest with strategy combination.
        
        Args:
            data: Historical OHLCV data
            symbol: Trading symbol
            commission: Commission rate
            slippage: Slippage percentage
            rebalance_freq: Rebalancing frequency
            
        Returns:
            Combined portfolio backtest result
        """
        # Run individual backtests
        individual_results = self.run_backtest(data, symbol, commission, slippage)
        
        if not individual_results:
            return BacktestResult(strategy_name="Portfolio")
        
        # Combine equity curves
        equity_curves = []
        weights = []
        
        for name, result in individual_results.items():
            if len(result.equity_curve) > 0:
                equity_curves.append(result.equity_curve)
                config = self.strategies[name]
                weights.append(config.weight)
        
        if not equity_curves:
            return BacktestResult(strategy_name="Portfolio")
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Create combined equity curve
        combined_equity = pd.DataFrame(equity_curves).T
        combined_equity = combined_equity.fillna(method='ffill')
        
        # Weighted average
        portfolio_equity = (combined_equity * weights).sum(axis=1)
        
        # Create portfolio result
        portfolio_result = BacktestResult(
            strategy_name="Portfolio",
            equity_curve=portfolio_equity
        )
        
        portfolio_result.calculate_metrics()
        
        return portfolio_result
    
    def get_state(self) -> EngineState:
        """Get current engine state."""
        with self._lock:
            return self.state
    
    def get_strategy_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all strategies."""
        with self._lock:
            return {
                name: config.strategy.get_stats()
                for name, config in self.strategies.items()
            }
    
    def reset(self) -> None:
        """Reset engine state."""
        with self._lock:
            self.total_equity = self.initial_capital
            self.positions = {}
            self.trade_history = []
            self.strategy_signals = defaultdict(list)
            
            for config in self.strategies.values():
                config.strategy.reset()
            
            self.state = EngineState(
                is_running=False,
                mode=self.mode,
                total_equity=self.initial_capital
            )
            
            logger.info("StrategyRunner reset")
    
    def add_signal_callback(self, callback: Callable[[Signal], None]) -> None:
        """Add signal callback."""
        self.on_signal_callbacks.append(callback)
    
    def add_trade_callback(self, callback: Callable[[Trade], None]) -> None:
        """Add trade callback."""
        self.on_trade_callbacks.append(callback)
    
    def add_bar_callback(self, callback: Callable[[pd.DataFrame], None]) -> None:
        """Add bar callback."""
        self.on_bar_callbacks.append(callback)
    
    def export_results(self, filepath: str) -> None:
        """Export backtest results to file."""
        stats = self.get_strategy_stats()
        
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        logger.info(f"Results exported to {filepath}")
    
    def shutdown(self) -> None:
        """Shutdown the engine."""
        with self._lock:
            self._running = False
            self._executor.shutdown(wait=True)
            logger.info("StrategyRunner shutdown complete")


class EventDrivenRunner(StrategyRunner):
    """
    Event-driven strategy runner for live/paper trading.
    
    Processes market data events as they arrive.
    """
    
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 mode: ExecutionMode = ExecutionMode.PAPER,
                 signal_aggregator: SignalAggregator = SignalAggregator.WEIGHTED_CONFIDENCE,
                 use_regime_filter: bool = True):
        super().__init__(initial_capital, mode, signal_aggregator, use_regime_filter)
        
        self._event_queue = asyncio.Queue()
        self._data_buffer: Dict[str, pd.DataFrame] = {}
        self._running_task = None
    
    async def start(self) -> None:
        """Start the event-driven runner."""
        self._running = True
        self.state.is_running = True
        self._running_task = asyncio.create_task(self._event_loop())
        logger.info("EventDrivenRunner started")
    
    async def stop(self) -> None:
        """Stop the event-driven runner."""
        self._running = False
        self.state.is_running = False
        
        if self._running_task:
            self._running_task.cancel()
            try:
                await self._running_task
            except asyncio.CancelledError:
                pass
        
        logger.info("EventDrivenRunner stopped")
    
    async def _event_loop(self) -> None:
        """Main event processing loop."""
        while self._running:
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(), 
                    timeout=1.0
                )
                
                await self._process_event(event)
            
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Event loop error: {e}")
    
    async def _process_event(self, event: Dict[str, Any]) -> None:
        """Process a market event."""
        event_type = event.get('type')
        
        if event_type == 'bar':
            symbol = event.get('symbol')
            bar = event.get('data')
            
            # Update buffer
            if symbol not in self._data_buffer:
                self._data_buffer[symbol] = pd.DataFrame()
            
            self._data_buffer[symbol] = pd.concat([
                self._data_buffer[symbol],
                pd.DataFrame([bar])
            ]).tail(1000)  # Keep last 1000 bars
            
            # Process through strategies
            self.process_bar(self._data_buffer[symbol], symbol)
        
        elif event_type == 'tick':
            # Process tick data
            symbol = event.get('symbol')
            tick = event.get('data')
            
            for config in self.strategies.values():
                if config.enabled:
                    config.strategy.on_tick(tick)
    
    async def submit_event(self, event: Dict[str, Any]) -> None:
        """Submit an event to the processing queue."""
        await self._event_queue.put(event)
    
    async def submit_bar(self, 
                        symbol: str, 
                        bar: Dict[str, Any]) -> None:
        """Submit a bar event."""
        await self.submit_event({
            'type': 'bar',
            'symbol': symbol,
            'data': bar
        })
    
    async def submit_tick(self,
                         symbol: str,
                         tick: Dict[str, Any]) -> None:
        """Submit a tick event."""
        await self.submit_event({
            'type': 'tick',
            'symbol': symbol,
            'data': tick
        })


# Export all classes
__all__ = [
    'ExecutionMode',
    'SignalAggregator',
    'StrategyConfig',
    'EngineState',
    'StrategyRunner',
    'EventDrivenRunner'
]
