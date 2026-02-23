"""
TradeOS Liquidity Sweep Detection Module
========================================
Detects liquidity sweeps, stop hunts, and order blocks.
Used primarily in price action trading.

Author: TradeOS Team
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from collections import deque
import logging

from ..base_strategy import BaseStrategy, Signal, SignalType
from ..indicators.technical import atr, ema, find_swing_points

logger = logging.getLogger(__name__)


@dataclass
class LiquidityLevel:
    """Represents a liquidity level (support/resistance with stops)."""
    price: float
    level_type: str  # 'support' or 'resistance'
    strength: int = 1  # Number of touches
    first_touch: datetime = field(default_factory=datetime.now)
    last_touch: datetime = field(default_factory=datetime.now)
    is_swept: bool = False
    sweep_price: Optional[float] = None
    sweep_time: Optional[datetime] = None
    volume_at_sweep: Optional[float] = None
    
    def touch(self, timestamp: datetime) -> None:
        """Record a touch at this level."""
        self.strength += 1
        self.last_touch = timestamp
    
    def mark_swept(self, price: float, timestamp: datetime, volume: float) -> None:
        """Mark level as swept."""
        self.is_swept = True
        self.sweep_price = price
        self.sweep_time = timestamp
        self.volume_at_sweep = volume


@dataclass
class OrderBlock:
    """Represents an order block (institutional order zone)."""
    high: float
    low: float
    open_price: float
    close_price: float
    timestamp: datetime
    block_type: str  # 'bullish' or 'bearish'
    volume: float = 0
    is_mitigated: bool = False
    mitigation_price: Optional[float] = None
    
    @property
    def mid_price(self) -> float:
        return (self.high + self.low) / 2
    
    @property
    def height(self) -> float:
        return self.high - self.low
    
    def contains_price(self, price: float) -> bool:
        """Check if price is within the order block."""
        return self.low <= price <= self.high
    
    def mitigate(self, price: float) -> None:
        """Mark order block as mitigated."""
        self.is_mitigated = True
        self.mitigation_price = price


class LiquiditySweepDetector:
    """
    Detects liquidity sweeps and stop hunts in price action.
    
    Identifies:
    - Equal highs/lows (liquidity pools)
    - Sweep patterns (false breakouts)
    - Stop hunt detection
    """
    
    def __init__(self,
                 lookback_period: int = 50,
                 sweep_threshold_pct: float = 0.001,
                 equal_levels_tolerance: float = 0.002,
                 min_touches: int = 2):
        """
        Initialize liquidity detector.
        
        Args:
            lookback_period: Bars to look back for levels
            sweep_threshold_pct: Minimum % beyond level to count as sweep
            equal_levels_tolerance: Tolerance for equal highs/lows
            min_touches: Minimum touches to consider a level valid
        """
        self.lookback_period = lookback_period
        self.sweep_threshold_pct = sweep_threshold_pct
        self.equal_levels_tolerance = equal_levels_tolerance
        self.min_touches = min_touches
        
        self.liquidity_levels: List[LiquidityLevel] = []
        self.sweeps_history: List[Dict] = []
        self.recent_swing_highs: deque = deque(maxlen=20)
        self.recent_swing_lows: deque = deque(maxlen=20)
    
    def find_equal_levels(self, data: pd.DataFrame) -> Tuple[List[float], List[float]]:
        """
        Find equal highs and lows (liquidity pools).
        
        Returns:
            Tuple of (equal_highs, equal_lows) lists
        """
        highs = data['high'].tail(self.lookback_period)
        lows = data['low'].tail(self.lookback_period)
        
        equal_highs = []
        equal_lows = []
        
        # Find clusters of similar highs
        for i, high in enumerate(highs):
            similar_count = 0
            for j, other_high in enumerate(highs):
                if i != j:
                    diff_pct = abs(high - other_high) / high
                    if diff_pct < self.equal_levels_tolerance:
                        similar_count += 1
            
            if similar_count >= self.min_touches - 1:
                if not any(abs(high - eh) / high < self.equal_levels_tolerance for eh in equal_highs):
                    equal_highs.append(high)
        
        # Find clusters of similar lows
        for i, low in enumerate(lows):
            similar_count = 0
            for j, other_low in enumerate(lows):
                if i != j:
                    diff_pct = abs(low - other_low) / low
                    if diff_pct < self.equal_levels_tolerance:
                        similar_count += 1
            
            if similar_count >= self.min_touches - 1:
                if not any(abs(low - el) / low < self.equal_levels_tolerance for el in equal_lows):
                    equal_lows.append(low)
        
        return equal_highs, equal_lows
    
    def detect_sweep(self, 
                     data: pd.DataFrame,
                     level: float,
                     level_type: str) -> Optional[Dict[str, Any]]:
        """
        Detect if a liquidity sweep occurred at a level.
        
        Args:
            data: OHLCV DataFrame
            level: Price level to check
            level_type: 'support' or 'resistance'
            
        Returns:
            Sweep information if detected, None otherwise
        """
        if len(data) < 3:
            return None
        
        current_bar = data.iloc[-1]
        prev_bar = data.iloc[-2]
        prev2_bar = data.iloc[-3]
        
        current_price = current_bar['close']
        
        sweep_detected = False
        sweep_info = None
        
        if level_type == 'resistance':
            # Bullish sweep: price briefly breaks above, then falls back
            broke_above = prev_bar['high'] > level * (1 + self.sweep_threshold_pct)
            fell_back = current_price < level
            
            if broke_above and fell_back:
                # Check for wick rejection
                upper_wick = prev_bar['high'] - max(prev_bar['open'], prev_bar['close'])
                body = abs(prev_bar['close'] - prev_bar['open'])
                
                if upper_wick > body * 0.5:  # Significant upper wick
                    sweep_detected = True
                    sweep_info = {
                        'type': 'bullish_sweep',
                        'level': level,
                        'sweep_price': prev_bar['high'],
                        'close_price': current_price,
                        'wick_ratio': upper_wick / body if body > 0 else float('inf'),
                        'volume': prev_bar.get('volume', 0),
                        'timestamp': data.index[-1]
                    }
        
        else:  # support
            # Bearish sweep: price briefly breaks below, then bounces back
            broke_below = prev_bar['low'] < level * (1 - self.sweep_threshold_pct)
            bounced_back = current_price > level
            
            if broke_below and bounced_back:
                # Check for wick rejection
                lower_wick = min(prev_bar['open'], prev_bar['close']) - prev_bar['low']
                body = abs(prev_bar['close'] - prev_bar['open'])
                
                if lower_wick > body * 0.5:  # Significant lower wick
                    sweep_detected = True
                    sweep_info = {
                        'type': 'bearish_sweep',
                        'level': level,
                        'sweep_price': prev_bar['low'],
                        'close_price': current_price,
                        'wick_ratio': lower_wick / body if body > 0 else float('inf'),
                        'volume': prev_bar.get('volume', 0),
                        'timestamp': data.index[-1]
                    }
        
        if sweep_detected and sweep_info:
            self.sweeps_history.append(sweep_info)
        
        return sweep_info if sweep_detected else None
    
    def detect_stop_hunt(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Detect stop hunt patterns.
        
        A stop hunt is a quick move beyond a level followed by rapid reversal.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            Stop hunt information if detected
        """
        if len(data) < 5:
            return None
        
        recent = data.tail(5)
        
        # Find recent range
        range_high = recent['high'].max()
        range_low = recent['low'].min()
        range_size = range_high - range_low
        
        if range_size == 0:
            return None
        
        # Check for spike and reversal
        first_bar = recent.iloc[0]
        last_bar = recent.iloc[-1]
        middle_bars = recent.iloc[1:-1]
        
        # Bullish stop hunt: spike below support, then close near high
        spike_low = middle_bars['low'].min()
        spike_from_low = (spike_low - range_low) / range_size
        
        if spike_from_low < 0.2:  # Spike below 20% of range
            close_position = (last_bar['close'] - range_low) / range_size
            if close_position > 0.6:  # Close in upper 40%
                return {
                    'type': 'bullish_stop_hunt',
                    'spike_price': spike_low,
                    'recovery_close': last_bar['close'],
                    'range_low': range_low,
                    'range_high': range_high,
                    'timestamp': data.index[-1]
                }
        
        # Bearish stop hunt: spike above resistance, then close near low
        spike_high = middle_bars['high'].max()
        spike_from_high = (range_high - spike_high) / range_size
        
        if spike_from_high < 0.2:  # Spike above 80% of range
            close_position = (last_bar['close'] - range_low) / range_size
            if close_position < 0.4:  # Close in lower 40%
                return {
                    'type': 'bearish_stop_hunt',
                    'spike_price': spike_high,
                    'recovery_close': last_bar['close'],
                    'range_low': range_low,
                    'range_high': range_high,
                    'timestamp': data.index[-1]
                }
        
        return None
    
    def update(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Update detector with new data and return any detected sweeps.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            List of detected sweep events
        """
        sweeps = []
        
        # Find liquidity levels
        equal_highs, equal_lows = self.find_equal_levels(data)
        
        # Check for sweeps at equal highs (resistance)
        for high in equal_highs:
            sweep = self.detect_sweep(data, high, 'resistance')
            if sweep:
                sweeps.append(sweep)
        
        # Check for sweeps at equal lows (support)
        for low in equal_lows:
            sweep = self.detect_sweep(data, low, 'support')
            if sweep:
                sweeps.append(sweep)
        
        # Check for stop hunts
        stop_hunt = self.detect_stop_hunt(data)
        if stop_hunt:
            sweeps.append(stop_hunt)
        
        return sweeps


class OrderBlockDetector:
    """
    Detects order blocks - areas where institutional orders were placed.
    
    Identifies:
    - Bullish order blocks (before up moves)
    - Bearish order blocks (before down moves)
    - Mitigated vs active blocks
    """
    
    def __init__(self,
                 lookback: int = 20,
                 min_impulse_pct: float = 0.5,
                 max_blocks: int = 10):
        """
        Initialize order block detector.
        
        Args:
            lookback: Bars to look back for blocks
            min_impulse_pct: Minimum % move to qualify as impulse
            max_blocks: Maximum blocks to track
        """
        self.lookback = lookback
        self.min_impulse_pct = min_impulse_pct
        self.max_blocks = max_blocks
        
        self.bullish_blocks: deque = deque(maxlen=max_blocks)
        self.bearish_blocks: deque = deque(maxlen=max_blocks)
    
    def is_bullish_candle(self, bar: pd.Series) -> bool:
        """Check if candle is bullish."""
        return bar['close'] > bar['open']
    
    def is_bearish_candle(self, bar: pd.Series) -> bool:
        """Check if candle is bearish."""
        return bar['close'] < bar['open']
    
    def detect_bullish_block(self, data: pd.DataFrame) -> Optional[OrderBlock]:
        """
        Detect bullish order block.
        
        Pattern: Bearish candle followed by strong bullish impulse.
        """
        if len(data) < 3:
            return None
        
        # Look for bearish candle before bullish impulse
        for i in range(-3, -min(len(data), self.lookback), -1):
            candle = data.iloc[i]
            next_candle = data.iloc[i + 1] if i + 1 < 0 else data.iloc[-1]
            
            # Bearish candle followed by bullish move
            if self.is_bearish_candle(candle):
                move_pct = (next_candle['close'] - candle['close']) / candle['close'] * 100
                
                if move_pct > self.min_impulse_pct and self.is_bullish_candle(next_candle):
                    return OrderBlock(
                        high=candle['high'],
                        low=candle['low'],
                        open_price=candle['open'],
                        close_price=candle['close'],
                        timestamp=data.index[i],
                        block_type='bullish',
                        volume=candle.get('volume', 0)
                    )
        
        return None
    
    def detect_bearish_block(self, data: pd.DataFrame) -> Optional[OrderBlock]:
        """
        Detect bearish order block.
        
        Pattern: Bullish candle followed by strong bearish impulse.
        """
        if len(data) < 3:
            return None
        
        # Look for bullish candle before bearish impulse
        for i in range(-3, -min(len(data), self.lookback), -1):
            candle = data.iloc[i]
            next_candle = data.iloc[i + 1] if i + 1 < 0 else data.iloc[-1]
            
            # Bullish candle followed by bearish move
            if self.is_bullish_candle(candle):
                move_pct = (candle['close'] - next_candle['close']) / candle['close'] * 100
                
                if move_pct > self.min_impulse_pct and self.is_bearish_candle(next_candle):
                    return OrderBlock(
                        high=candle['high'],
                        low=candle['low'],
                        open_price=candle['open'],
                        close_price=candle['close'],
                        timestamp=data.index[i],
                        block_type='bearish',
                        volume=candle.get('volume', 0)
                    )
        
        return None
    
    def check_mitigation(self, data: pd.DataFrame) -> None:
        """Check if any order blocks have been mitigated."""
        current_price = data['close'].iloc[-1]
        
        for block in list(self.bullish_blocks):
            if not block.is_mitigated:
                # Bullish block is mitigated when price trades below its low
                if current_price < block.low:
                    block.mitigate(current_price)
        
        for block in list(self.bearish_blocks):
            if not block.is_mitigated:
                # Bearish block is mitigated when price trades above its high
                if current_price > block.high:
                    block.mitigate(current_price)
    
    def get_active_blocks(self, block_type: Optional[str] = None) -> List[OrderBlock]:
        """Get active (non-mitigated) order blocks."""
        blocks = []
        
        if block_type is None or block_type == 'bullish':
            blocks.extend([b for b in self.bullish_blocks if not b.is_mitigated])
        
        if block_type is None or block_type == 'bearish':
            blocks.extend([b for b in self.bearish_blocks if not b.is_mitigated])
        
        return blocks
    
    def update(self, data: pd.DataFrame) -> Dict[str, List[OrderBlock]]:
        """
        Update detector with new data.
        
        Returns:
            Dictionary with new bullish and bearish blocks
        """
        # Check mitigation first
        self.check_mitigation(data)
        
        # Detect new blocks
        new_bullish = self.detect_bullish_block(data)
        new_bearish = self.detect_bearish_block(data)
        
        result = {'bullish': [], 'bearish': []}
        
        if new_bullish:
            self.bullish_blocks.append(new_bullish)
            result['bullish'].append(new_bullish)
        
        if new_bearish:
            self.bearish_blocks.append(new_bearish)
            result['bearish'].append(new_bearish)
        
        return result


class LiquiditySweepStrategy(BaseStrategy):
    """
    Trading strategy based on liquidity sweeps.
    
    Enters after liquidity sweep with confirmation.
    
    Parameters:
        sweep_lookback: Lookback for liquidity levels (default: 50)
        sweep_threshold: Threshold for sweep detection (default: 0.001)
        confirmation_candles: Candles to wait for confirmation (default: 1)
        use_order_blocks: Use order blocks for entry (default: True)
        min_sweep_wick: Minimum wick ratio for valid sweep (default: 1.5)
    """
    
    def __init__(self,
                 name: str = "Liquidity_Sweep",
                 params: Optional[Dict[str, Any]] = None,
                 position_sizer=None):
        
        default_params = {
            'sweep_lookback': 50,
            'sweep_threshold': 0.001,
            'confirmation_candles': 1,
            'use_order_blocks': True,
            'min_sweep_wick': 1.5,
            'min_bars': 30,
            'stop_loss_atr': 1.5,
            'take_profit_atr': 2.5
        }
        
        if params:
            default_params.update(params)
        
        super().__init__(name, default_params, position_sizer)
        
        self.sweep_detector = LiquiditySweepDetector(
            lookback_period=self.params['sweep_lookback'],
            sweep_threshold_pct=self.params['sweep_threshold']
        )
        
        self.ob_detector = OrderBlockDetector() if self.params['use_order_blocks'] else None
        
        self.pending_sweep: Optional[Dict] = None
        self.confirmation_count = 0
    
    def _precompute_indicators(self, data: pd.DataFrame) -> None:
        """Precompute indicators."""
        self.atr_values = atr(data, 14).values
        logger.info(f"Precomputed liquidity sweep indicators for {self.name}")
    
    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """Generate signals based on liquidity sweeps."""
        if len(data) < self.params['min_bars']:
            return self._no_signal(data)
        
        current_price = data['close'].iloc[-1]
        timestamp = data.index[-1]
        symbol = data.get('symbol', ['UNKNOWN'])[-1] if 'symbol' in data.columns else 'UNKNOWN'
        
        # Update detectors
        sweeps = self.sweep_detector.update(data)
        
        if self.ob_detector:
            self.ob_detector.update(data)
        
        atr_val = atr(data, 14).values.iloc[-1]
        
        # Process new sweeps
        for sweep in sweeps:
            wick_ratio = sweep.get('wick_ratio', 0)
            
            if wick_ratio >= self.params['min_sweep_wick']:
                self.pending_sweep = sweep
                self.confirmation_count = 0
        
        # Check for confirmation
        if self.pending_sweep:
            self.confirmation_count += 1
            
            if self.confirmation_count >= self.params['confirmation_candles']:
                sweep = self.pending_sweep
                sweep_type = sweep['type']
                
                # Check for order block confirmation
                ob_confirmed = True
                if self.ob_detector:
                    active_blocks = self.ob_detector.get_active_blocks()
                    ob_confirmed = any(
                        b.contains_price(current_price) for b in active_blocks
                    )
                
                if ob_confirmed:
                    self.pending_sweep = None
                    self.confirmation_count = 0
                    
                    if sweep_type == 'bullish_sweep' or sweep_type == 'bullish_stop_hunt':
                        return Signal(
                            timestamp=timestamp,
                            symbol=symbol,
                            signal_type=SignalType.BUY,
                            price=current_price,
                            confidence=0.75,
                            strategy_name=self.name,
                            metadata={
                                'sweep_type': sweep_type,
                                'sweep_level': sweep.get('level', sweep.get('range_low')),
                                'sweep_price': sweep.get('sweep_price', sweep.get('spike_price')),
                                'atr': atr_val,
                                'stop_loss': sweep.get('sweep_price', current_price - atr_val),
                                'take_profit': current_price + atr_val * self.params['take_profit_atr']
                            }
                        )
                    
                    elif sweep_type == 'bearish_sweep' or sweep_type == 'bearish_stop_hunt':
                        return Signal(
                            timestamp=timestamp,
                            symbol=symbol,
                            signal_type=SignalType.SELL,
                            price=current_price,
                            confidence=0.75,
                            strategy_name=self.name,
                            metadata={
                                'sweep_type': sweep_type,
                                'sweep_level': sweep.get('level', sweep.get('range_high')),
                                'sweep_price': sweep.get('sweep_price', sweep.get('spike_price')),
                                'atr': atr_val,
                                'stop_loss': sweep.get('sweep_price', current_price + atr_val),
                                'take_profit': current_price - atr_val * self.params['take_profit_atr']
                            }
                        )
        
        return self._no_signal(data)
    
    def _no_signal(self, data: pd.DataFrame) -> Signal:
        """Return no-signal response."""
        return Signal(
            timestamp=data.index[-1] if len(data) > 0 else datetime.now(),
            symbol=data.get('symbol', ['UNKNOWN'])[-1] if 'symbol' in data.columns else 'UNKNOWN',
            signal_type=SignalType.HOLD,
            price=data['close'].iloc[-1] if len(data) > 0 else 0,
            confidence=0.0,
            strategy_name=self.name
        )


class OrderBlockStrategy(BaseStrategy):
    """
    Trading strategy based on order blocks.
    
    Enters when price returns to an active order block.
    
    Parameters:
        min_impulse_pct: Minimum impulse for block detection (default: 0.5)
        max_blocks: Maximum blocks to track (default: 10)
        entry_at: Entry point within block ('mid', 'high', 'low') (default: 'mid')
        require_fvg: Require fair value gap for entry (default: False)
    """
    
    def __init__(self,
                 name: str = "Order_Block",
                 params: Optional[Dict[str, Any]] = None,
                 position_sizer=None):
        
        default_params = {
            'min_impulse_pct': 0.5,
            'max_blocks': 10,
            'entry_at': 'mid',
            'require_fvg': False,
            'min_bars': 20,
            'stop_loss_atr': 1.5,
            'take_profit_atr': 3.0
        }
        
        if params:
            default_params.update(params)
        
        super().__init__(name, default_params, position_sizer)
        
        self.ob_detector = OrderBlockDetector(
            min_impulse_pct=self.params['min_impulse_pct'],
            max_blocks=self.params['max_blocks']
        )
    
    def _precompute_indicators(self, data: pd.DataFrame) -> None:
        """Precompute indicators."""
        self.atr_values = atr(data, 14).values
        logger.info(f"Precomputed order block indicators for {self.name}")
    
    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """Generate signals based on order block interaction."""
        if len(data) < self.params['min_bars']:
            return self._no_signal(data)
        
        current_price = data['close'].iloc[-1]
        current_low = data['low'].iloc[-1]
        current_high = data['high'].iloc[-1]
        timestamp = data.index[-1]
        symbol = data.get('symbol', ['UNKNOWN'])[-1] if 'symbol' in data.columns else 'UNKNOWN'
        
        # Update order block detector
        self.ob_detector.update(data)
        
        atr_val = atr(data, 14).values.iloc[-1]
        
        # Check active bullish blocks
        bullish_blocks = self.ob_detector.get_active_blocks('bullish')
        for block in bullish_blocks:
            # Price entered bullish block
            if block.low <= current_price <= block.high:
                entry_price = getattr(block, self.params['entry_at'] + '_price', block.mid_price)
                
                return Signal(
                    timestamp=timestamp,
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    price=current_price,
                    confidence=0.7,
                    strategy_name=self.name,
                    metadata={
                        'block_type': 'bullish',
                        'block_high': block.high,
                        'block_low': block.low,
                        'block_mid': block.mid_price,
                        'atr': atr_val,
                        'stop_loss': block.low - atr_val * 0.5,
                        'take_profit': current_price + atr_val * self.params['take_profit_atr']
                    }
                )
        
        # Check active bearish blocks
        bearish_blocks = self.ob_detector.get_active_blocks('bearish')
        for block in bearish_blocks:
            # Price entered bearish block
            if block.low <= current_price <= block.high:
                entry_price = getattr(block, self.params['entry_at'] + '_price', block.mid_price)
                
                return Signal(
                    timestamp=timestamp,
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    price=current_price,
                    confidence=0.7,
                    strategy_name=self.name,
                    metadata={
                        'block_type': 'bearish',
                        'block_high': block.high,
                        'block_low': block.low,
                        'block_mid': block.mid_price,
                        'atr': atr_val,
                        'stop_loss': block.high + atr_val * 0.5,
                        'take_profit': current_price - atr_val * self.params['take_profit_atr']
                    }
                )
        
        return self._no_signal(data)
    
    def _no_signal(self, data: pd.DataFrame) -> Signal:
        """Return no-signal response."""
        return Signal(
            timestamp=data.index[-1] if len(data) > 0 else datetime.now(),
            symbol=data.get('symbol', ['UNKNOWN'])[-1] if 'symbol' in data.columns else 'UNKNOWN',
            signal_type=SignalType.HOLD,
            price=data['close'].iloc[-1] if len(data) > 0 else 0,
            confidence=0.0,
            strategy_name=self.name
        )


# Export all classes
__all__ = [
    'LiquidityLevel',
    'OrderBlock',
    'LiquiditySweepDetector',
    'OrderBlockDetector',
    'LiquiditySweepStrategy',
    'OrderBlockStrategy'
]
