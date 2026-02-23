"""
TradeOS Strategy Engine - Base Strategy Module
==============================================
Abstract base class for all trading strategies.
Provides unified interface for signal generation, position sizing, and backtesting.

Author: TradeOS Team
Version: 1.0.0
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Callable, Any, Tuple, Union
from datetime import datetime
import pandas as pd
import numpy as np
from collections import deque
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Trade signal types."""
    BUY = 1
    SELL = -1
    HOLD = 0
    CLOSE = 2


class OrderType(Enum):
    """Order execution types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


@dataclass
class Signal:
    """Trade signal data structure."""
    timestamp: datetime
    symbol: str
    signal_type: SignalType
    price: float
    confidence: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    strategy_name: str = ""
    timeframe: str = "1h"
    
    def __post_init__(self):
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
    
    def is_valid(self) -> bool:
        """Check if signal is valid."""
        return self.signal_type != SignalType.HOLD and self.confidence > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary."""
        return {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'signal_type': self.signal_type.name,
            'price': self.price,
            'confidence': self.confidence,
            'metadata': self.metadata,
            'strategy_name': self.strategy_name,
            'timeframe': self.timeframe
        }


@dataclass
class Position:
    """Position tracking data structure."""
    symbol: str
    side: SignalType  # BUY (long) or SELL (short)
    entry_price: float
    size: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_long(self) -> bool:
        return self.side == SignalType.BUY
    
    @property
    def is_short(self) -> bool:
        return self.side == SignalType.SELL
    
    def update_unrealized_pnl(self, current_price: float) -> float:
        """Update and return unrealized PnL."""
        if self.is_long:
            self.unrealized_pnl = (current_price - self.entry_price) * self.size
        else:
            self.unrealized_pnl = (self.entry_price - current_price) * self.size
        return self.unrealized_pnl
    
    def close_position(self, exit_price: float, exit_time: datetime) -> float:
        """Close position and return realized PnL."""
        pnl = self.update_unrealized_pnl(exit_price)
        self.realized_pnl = pnl
        return pnl


@dataclass
class Trade:
    """Completed trade record."""
    entry_signal: Signal
    exit_signal: Optional[Signal]
    position: Position
    pnl: float
    pnl_pct: float
    duration: float  # in hours
    exit_reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'entry_time': self.entry_signal.timestamp,
            'exit_time': self.exit_signal.timestamp if self.exit_signal else None,
            'symbol': self.position.symbol,
            'side': self.position.side.name,
            'entry_price': self.position.entry_price,
            'exit_price': self.exit_signal.price if self.exit_signal else None,
            'size': self.position.size,
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct,
            'duration_hours': self.duration,
            'exit_reason': self.exit_reason
        }


@dataclass
class BacktestResult:
    """Backtest results container."""
    strategy_name: str
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series())
    signals: List[Signal] = field(default_factory=list)
    
    # Performance metrics
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate all performance metrics."""
        if not self.trades:
            return {}
        
        self.total_trades = len(self.trades)
        pnls = [t.pnl for t in self.trades]
        
        self.winning_trades = sum(1 for p in pnls if p > 0)
        self.losing_trades = sum(1 for p in pnls if p < 0)
        self.win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        self.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        self.avg_trade = np.mean(pnls) if pnls else 0
        
        if len(self.equity_curve) > 1:
            returns = self.equity_curve.pct_change().dropna()
            self.sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
            
            # Max drawdown
            cummax = self.equity_curve.cummax()
            drawdown = (self.equity_curve - cummax) / cummax
            self.max_drawdown = drawdown.min()
            
            self.total_return = (self.equity_curve.iloc[-1] / self.equity_curve.iloc[0] - 1) if self.equity_curve.iloc[0] > 0 else 0
        
        return {
            'total_return': self.total_return,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'avg_trade': self.avg_trade,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert trades to DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([t.to_dict() for t in self.trades])


class PositionSizer(ABC):
    """Abstract base class for position sizing algorithms."""
    
    @abstractmethod
    def calculate_size(self, 
                       capital: float,
                       price: float,
                       signal: Signal,
                       volatility: Optional[float] = None) -> float:
        """Calculate position size."""
        pass


class FixedPositionSizer(PositionSizer):
    """Fixed dollar amount position sizing."""
    
    def __init__(self, fixed_amount: float = 10000.0):
        self.fixed_amount = fixed_amount
    
    def calculate_size(self, 
                       capital: float,
                       price: float,
                       signal: Signal,
                       volatility: Optional[float] = None) -> float:
        """Calculate position size based on fixed amount."""
        return self.fixed_amount / price if price > 0 else 0


class PercentOfEquitySizer(PositionSizer):
    """Percentage of equity position sizing."""
    
    def __init__(self, percent: float = 0.1):
        self.percent = percent
    
    def calculate_size(self, 
                       capital: float,
                       price: float,
                       signal: Signal,
                       volatility: Optional[float] = None) -> float:
        """Calculate position size as percentage of equity."""
        position_value = capital * self.percent
        return position_value / price if price > 0 else 0


class KellySizer(PositionSizer):
    """Kelly Criterion position sizing."""
    
    def __init__(self, win_rate: float = 0.5, avg_win: float = 1.0, avg_loss: float = 1.0):
        self.win_rate = win_rate
        self.avg_win = avg_win
        self.avg_loss = avg_loss
    
    def calculate_size(self, 
                       capital: float,
                       price: float,
                       signal: Signal,
                       volatility: Optional[float] = None) -> float:
        """Calculate position size using Kelly Criterion."""
        b = self.avg_win / self.avg_loss if self.avg_loss > 0 else 1
        q = 1 - self.win_rate
        kelly = (self.win_rate * b - q) / b if b > 0 else 0
        kelly = max(0, min(kelly, 0.25))  # Cap at 25%
        
        position_value = capital * kelly
        return position_value / price if price > 0 else 0


class ATRPositionSizer(PositionSizer):
    """ATR-based volatility-adjusted position sizing."""
    
    def __init__(self, risk_per_trade: float = 0.02, atr_multiple: float = 2.0):
        self.risk_per_trade = risk_per_trade
        self.atr_multiple = atr_multiple
    
    def calculate_size(self, 
                       capital: float,
                       price: float,
                       signal: Signal,
                       volatility: Optional[float] = None) -> float:
        """Calculate position size based on ATR."""
        if volatility is None or volatility <= 0:
            # Fallback to percent of equity
            position_value = capital * self.risk_per_trade
            return position_value / price if price > 0 else 0
        
        risk_amount = capital * self.risk_per_trade
        stop_distance = volatility * self.atr_multiple
        
        if stop_distance <= 0:
            return 0
        
        return risk_amount / stop_distance


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    Provides a unified interface for:
    - Signal generation
    - Position sizing integration
    - Backtesting capabilities
    - Risk management hooks
    """
    
    def __init__(self, 
                 name: str,
                 params: Optional[Dict[str, Any]] = None,
                 position_sizer: Optional[PositionSizer] = None):
        """
        Initialize strategy.
        
        Args:
            name: Strategy identifier
            params: Strategy parameters dictionary
            position_sizer: Position sizing algorithm
        """
        self.name = name
        self.params = params or {}
        self.position_sizer = position_sizer or PercentOfEquitySizer(0.1)
        
        # State tracking
        self.is_initialized = False
        self.current_position: Optional[Position] = None
        self.signals_history: List[Signal] = []
        self.trades_history: List[Trade] = []
        self.data_buffer: deque = deque(maxlen=1000)
        
        # Performance tracking
        self.equity: float = 100000.0  # Default starting capital
        self.initial_capital: float = 100000.0
        
        # Callbacks
        self.on_signal_callbacks: List[Callable[[Signal], None]] = []
        self.on_trade_callbacks: List[Callable[[Trade], None]] = []
        
        logger.info(f"Strategy '{name}' initialized")
    
    def initialize(self, data: pd.DataFrame) -> None:
        """Initialize strategy with historical data."""
        self._validate_data(data)
        self._precompute_indicators(data)
        self.is_initialized = True
        logger.info(f"Strategy '{self.name}' initialized with {len(data)} data points")
    
    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data format."""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_columns if col not in data.columns.str.lower()]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    @abstractmethod
    def _precompute_indicators(self, data: pd.DataFrame) -> None:
        """Precompute technical indicators. Override in subclass."""
        pass
    
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """
        Generate trading signal from data.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            Signal object
        """
        pass
    
    def on_bar(self, bar: pd.Series) -> Optional[Signal]:
        """
        Process new bar data.
        
        Args:
            bar: Single bar of OHLCV data
            
        Returns:
            Signal if generated, None otherwise
        """
        self.data_buffer.append(bar)
        
        # Convert buffer to DataFrame for indicator calculation
        if len(self.data_buffer) >= self.params.get('min_bars', 50):
            df = pd.DataFrame(list(self.data_buffer))
            signal = self.generate_signal(df)
            
            if signal.is_valid():
                self.signals_history.append(signal)
                self._notify_signal(signal)
                return signal
        
        return None
    
    def on_tick(self, tick: Dict[str, Any]) -> Optional[Signal]:
        """
        Process tick data (for high-frequency strategies).
        
        Args:
            tick: Tick data dictionary with price, volume, timestamp
            
        Returns:
            Signal if generated, None otherwise
        """
        # Default implementation - override for tick-based strategies
        return None
    
    def calculate_position_size(self, 
                                signal: Signal, 
                                volatility: Optional[float] = None) -> float:
        """
        Calculate position size for a signal.
        
        Args:
            signal: Trading signal
            volatility: Optional volatility measure (e.g., ATR)
            
        Returns:
            Position size in units
        """
        return self.position_sizer.calculate_size(
            self.equity, signal.price, signal, volatility
        )
    
    def set_position_sizer(self, sizer: PositionSizer) -> None:
        """Set position sizing algorithm."""
        self.position_sizer = sizer
    
    def add_signal_callback(self, callback: Callable[[Signal], None]) -> None:
        """Add callback for signal events."""
        self.on_signal_callbacks.append(callback)
    
    def add_trade_callback(self, callback: Callable[[Trade], None]) -> None:
        """Add callback for trade events."""
        self.on_trade_callbacks.append(callback)
    
    def _notify_signal(self, signal: Signal) -> None:
        """Notify signal callbacks."""
        for callback in self.on_signal_callbacks:
            try:
                callback(signal)
            except Exception as e:
                logger.error(f"Signal callback error: {e}")
    
    def _notify_trade(self, trade: Trade) -> None:
        """Notify trade callbacks."""
        for callback in self.on_trade_callbacks:
            try:
                callback(trade)
            except Exception as e:
                logger.error(f"Trade callback error: {e}")
    
    def enter_position(self, signal: Signal, size: Optional[float] = None) -> Position:
        """
        Enter a new position.
        
        Args:
            signal: Entry signal
            size: Position size (calculated if None)
            
        Returns:
            Position object
        """
        if size is None:
            volatility = signal.metadata.get('atr', None)
            size = self.calculate_position_size(signal, volatility)
        
        position = Position(
            symbol=signal.symbol,
            side=signal.signal_type,
            entry_price=signal.price,
            size=size,
            entry_time=signal.timestamp,
            stop_loss=signal.metadata.get('stop_loss'),
            take_profit=signal.metadata.get('take_profit'),
            metadata=signal.metadata
        )
        
        self.current_position = position
        logger.info(f"Entered {signal.signal_type.name} position: {size} @ {signal.price}")
        
        return position
    
    def exit_position(self, 
                      signal: Signal, 
                      reason: str = "signal") -> Optional[Trade]:
        """
        Exit current position.
        
        Args:
            signal: Exit signal
            reason: Exit reason
            
        Returns:
            Trade record if position existed, None otherwise
        """
        if self.current_position is None:
            return None
        
        position = self.current_position
        pnl = position.close_position(signal.price, signal.timestamp)
        
        # Calculate percentage return
        invested = position.entry_price * position.size
        pnl_pct = (pnl / invested) * 100 if invested > 0 else 0
        
        # Calculate duration
        duration = (signal.timestamp - position.entry_time).total_seconds() / 3600
        
        trade = Trade(
            entry_signal=self.signals_history[-1] if self.signals_history else signal,
            exit_signal=signal,
            position=position,
            pnl=pnl,
            pnl_pct=pnl_pct,
            duration=duration,
            exit_reason=reason
        )
        
        self.trades_history.append(trade)
        self.equity += pnl
        self.current_position = None
        
        self._notify_trade(trade)
        logger.info(f"Exited position: PnL={pnl:.2f} ({pnl_pct:.2f}%), Reason={reason}")
        
        return trade
    
    def check_exit_conditions(self, 
                              current_price: float, 
                              timestamp: datetime) -> Optional[Signal]:
        """
        Check if current position should be exited based on stop/take profit.
        
        Args:
            current_price: Current market price
            timestamp: Current timestamp
            
        Returns:
            Exit signal if conditions met, None otherwise
        """
        if self.current_position is None:
            return None
        
        position = self.current_position
        exit_reason = None
        
        # Check stop loss
        if position.stop_loss is not None:
            if position.is_long and current_price <= position.stop_loss:
                exit_reason = "stop_loss"
            elif position.is_short and current_price >= position.stop_loss:
                exit_reason = "stop_loss"
        
        # Check take profit
        if position.take_profit is not None and exit_reason is None:
            if position.is_long and current_price >= position.take_profit:
                exit_reason = "take_profit"
            elif position.is_short and current_price <= position.take_profit:
                exit_reason = "take_profit"
        
        if exit_reason:
            signal = Signal(
                timestamp=timestamp,
                symbol=position.symbol,
                signal_type=SignalType.CLOSE,
                price=current_price,
                confidence=1.0,
                strategy_name=self.name
            )
            self.exit_position(signal, exit_reason)
            return signal
        
        return None
    
    def backtest(self, 
                 data: pd.DataFrame,
                 initial_capital: float = 100000.0,
                 commission: float = 0.001,
                 slippage: float = 0.0) -> BacktestResult:
        """
        Run backtest on historical data.
        
        Args:
            data: OHLCV DataFrame with datetime index
            initial_capital: Starting capital
            commission: Commission rate per trade
            slippage: Slippage as percentage
            
        Returns:
            BacktestResult with performance metrics
        """
        self.initial_capital = initial_capital
        self.equity = initial_capital
        
        result = BacktestResult(strategy_name=self.name)
        equity_curve = [initial_capital]
        
        # Initialize strategy
        self.initialize(data)
        
        for i in range(self.params.get('min_bars', 50), len(data)):
            current_data = data.iloc[:i+1]
            current_bar = data.iloc[i]
            timestamp = data.index[i]
            
            # Check exit conditions first
            if self.current_position is not None:
                self.check_exit_conditions(current_bar['close'], timestamp)
            
            # Generate signal
            signal = self.generate_signal(current_data)
            signal.timestamp = timestamp
            signal.strategy_name = self.name
            
            # Process signal
            if signal.is_valid():
                result.signals.append(signal)
                
                # Apply slippage
                if signal.signal_type == SignalType.BUY:
                    execution_price = current_bar['close'] * (1 + slippage)
                elif signal.signal_type == SignalType.SELL:
                    execution_price = current_bar['close'] * (1 - slippage)
                else:
                    execution_price = current_bar['close']
                
                signal.price = execution_price
                
                # Execute signal
                if self.current_position is None:
                    # Enter new position
                    if signal.signal_type in [SignalType.BUY, SignalType.SELL]:
                        self.enter_position(signal)
                        
                else:
                    # Exit or reverse position
                    if (self.current_position.is_long and signal.signal_type == SignalType.SELL) or \
                       (self.current_position.is_short and signal.signal_type == SignalType.BUY):
                        self.exit_position(signal, "reverse")
                        self.enter_position(signal)
                    elif signal.signal_type == SignalType.CLOSE:
                        self.exit_position(signal, "signal")
            
            # Update equity curve
            current_equity = self.equity
            if self.current_position is not None:
                current_equity += self.current_position.update_unrealized_pnl(current_bar['close'])
            equity_curve.append(current_equity)
        
        # Close any open position at end
        if self.current_position is not None:
            final_signal = Signal(
                timestamp=data.index[-1],
                symbol=self.current_position.symbol,
                signal_type=SignalType.CLOSE,
                price=data['close'].iloc[-1],
                confidence=1.0,
                strategy_name=self.name
            )
            self.exit_position(final_signal, "end_of_data")
        
        # Build results
        result.trades = self.trades_history
        result.equity_curve = pd.Series(equity_curve, index=[data.index[0]] + list(data.index[self.params.get('min_bars', 50)-1:]))
        
        # Apply commission
        for trade in result.trades:
            commission_cost = trade.position.entry_price * trade.position.size * commission * 2  # Entry + exit
            trade.pnl -= commission_cost
        
        result.calculate_metrics()
        
        logger.info(f"Backtest complete: {result.total_trades} trades, Return: {result.total_return:.2%}")
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get strategy statistics."""
        return {
            'name': self.name,
            'total_signals': len(self.signals_history),
            'total_trades': len(self.trades_history),
            'current_position': self.current_position is not None,
            'equity': self.equity,
            'is_initialized': self.is_initialized
        }
    
    def reset(self) -> None:
        """Reset strategy state."""
        self.is_initialized = False
        self.current_position = None
        self.signals_history = []
        self.trades_history = []
        self.data_buffer.clear()
        self.equity = self.initial_capital
        logger.info(f"Strategy '{self.name}' reset")
    
    def update_params(self, params: Dict[str, Any]) -> None:
        """Update strategy parameters."""
        self.params.update(params)
        logger.info(f"Strategy '{self.name}' parameters updated: {params}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize strategy to dictionary."""
        return {
            'name': self.name,
            'params': self.params,
            'stats': self.get_stats()
        }


class MultiStrategy(BaseStrategy):
    """
    Strategy that combines multiple sub-strategies.
    Uses voting or consensus mechanism for signal generation.
    """
    
    def __init__(self, 
                 name: str,
                 strategies: List[BaseStrategy],
                 consensus_threshold: float = 0.5,
                 params: Optional[Dict[str, Any]] = None):
        """
        Initialize multi-strategy.
        
        Args:
            name: Strategy name
            strategies: List of sub-strategies
            consensus_threshold: Minimum consensus for signal (0-1)
            params: Additional parameters
        """
        super().__init__(name, params)
        self.strategies = strategies
        self.consensus_threshold = consensus_threshold
    
    def _precompute_indicators(self, data: pd.DataFrame) -> None:
        """Initialize all sub-strategies."""
        for strategy in self.strategies:
            strategy.initialize(data)
    
    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """Generate consensus signal from all sub-strategies."""
        signals = []
        
        for strategy in self.strategies:
            signal = strategy.generate_signal(data)
            if signal.is_valid():
                signals.append(signal)
        
        if not signals:
            return Signal(
                timestamp=data.index[-1],
                symbol="",
                signal_type=SignalType.HOLD,
                price=data['close'].iloc[-1],
                confidence=0.0,
                strategy_name=self.name
            )
        
        # Calculate consensus
        buy_votes = sum(1 for s in signals if s.signal_type == SignalType.BUY)
        sell_votes = sum(1 for s in signals if s.signal_type == SignalType.SELL)
        total_votes = len(signals)
        
        buy_consensus = buy_votes / total_votes
        sell_consensus = sell_votes / total_votes
        
        current_price = data['close'].iloc[-1]
        timestamp = data.index[-1]
        
        if buy_consensus >= self.consensus_threshold:
            avg_confidence = np.mean([s.confidence for s in signals if s.signal_type == SignalType.BUY])
            return Signal(
                timestamp=timestamp,
                symbol=signals[0].symbol,
                signal_type=SignalType.BUY,
                price=current_price,
                confidence=avg_confidence * buy_consensus,
                strategy_name=self.name,
                metadata={'consensus': buy_consensus, 'votes': buy_votes}
            )
        elif sell_consensus >= self.consensus_threshold:
            avg_confidence = np.mean([s.confidence for s in signals if s.signal_type == SignalType.SELL])
            return Signal(
                timestamp=timestamp,
                symbol=signals[0].symbol,
                signal_type=SignalType.SELL,
                price=current_price,
                confidence=avg_confidence * sell_consensus,
                strategy_name=self.name,
                metadata={'consensus': sell_consensus, 'votes': sell_votes}
            )
        
        return Signal(
            timestamp=timestamp,
            symbol=signals[0].symbol if signals else "",
            signal_type=SignalType.HOLD,
            price=current_price,
            confidence=0.0,
            strategy_name=self.name
        )


# Export all classes
__all__ = [
    'SignalType', 'OrderType', 'Signal', 'Position', 'Trade', 'BacktestResult',
    'PositionSizer', 'FixedPositionSizer', 'PercentOfEquitySizer', 
    'KellySizer', 'ATRPositionSizer',
    'BaseStrategy', 'MultiStrategy'
]
