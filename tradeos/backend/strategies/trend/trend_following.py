"""
TradeOS Trend Following Strategies
==================================
Collection of trend-following trading strategies.

Author: TradeOS Team
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

from ..base_strategy import BaseStrategy, Signal, SignalType, Position, Trade, BacktestResult
from ..indicators.technical import (
    ema, sma, adx, atr, bollinger_bands, volume_profile, 
    crossover, crossunder, vwap, supertrend, macd
)

logger = logging.getLogger(__name__)


class EMACrossoverStrategy(BaseStrategy):
    """
    EMA Crossover Strategy.
    
    Generates buy signals when fast EMA crosses above slow EMA.
    Generates sell signals when fast EMA crosses below slow EMA.
    
    Parameters:
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        use_volume: Confirm with volume increase (default: True)
        volume_threshold: Minimum volume increase ratio (default: 1.2)
    """
    
    def __init__(self, 
                 name: str = "EMA_Crossover",
                 params: Optional[Dict[str, Any]] = None,
                 position_sizer=None):
        
        default_params = {
            'fast_period': 12,
            'slow_period': 26,
            'use_volume': True,
            'volume_threshold': 1.2,
            'min_bars': 50,
            'stop_loss_atr': 2.0,
            'take_profit_atr': 3.0
        }
        
        if params:
            default_params.update(params)
        
        super().__init__(name, default_params, position_sizer)
        
        self.fast_ema = None
        self.slow_ema = None
        self.atr_values = None
    
    def _precompute_indicators(self, data: pd.DataFrame) -> None:
        """Precompute EMA indicators."""
        close = data['close']
        
        self.fast_ema = ema(close, self.params['fast_period']).values
        self.slow_ema = ema(close, self.params['slow_period']).values
        self.atr_values = atr(data, 14).values
        
        logger.info(f"Precomputed EMA indicators for {self.name}")
    
    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """Generate signal based on EMA crossover."""
        if len(data) < self.params['min_bars']:
            return self._no_signal(data)
        
        current_price = data['close'].iloc[-1]
        timestamp = data.index[-1]
        symbol = data.get('symbol', ['UNKNOWN'])[-1] if 'symbol' in data.columns else 'UNKNOWN'
        
        # Recalculate EMAs for current window
        close = data['close']
        fast_ema = ema(close, self.params['fast_period']).values
        slow_ema = ema(close, self.params['slow_period']).values
        
        # Check for crossover
        fast_curr = fast_ema.iloc[-1]
        fast_prev = fast_ema.iloc[-2]
        slow_curr = slow_ema.iloc[-1]
        slow_prev = slow_ema.iloc[-2]
        
        is_crossover = (fast_prev <= slow_prev) and (fast_curr > slow_curr)
        is_crossunder = (fast_prev >= slow_prev) and (fast_curr < slow_curr)
        
        # Volume confirmation
        volume_confirmed = True
        if self.params['use_volume'] and 'volume' in data.columns:
            vol_sma = data['volume'].rolling(20).mean().iloc[-1]
            current_vol = data['volume'].iloc[-1]
            volume_confirmed = current_vol > vol_sma * (self.params['volume_threshold'] - 1)
        
        # Calculate ATR for stop loss
        atr_val = atr(data, 14).values.iloc[-1]
        
        if is_crossover and volume_confirmed:
            confidence = min(0.5 + abs(fast_curr - slow_curr) / slow_curr * 10, 0.95)
            return Signal(
                timestamp=timestamp,
                symbol=symbol,
                signal_type=SignalType.BUY,
                price=current_price,
                confidence=confidence,
                strategy_name=self.name,
                metadata={
                    'fast_ema': fast_curr,
                    'slow_ema': slow_curr,
                    'atr': atr_val,
                    'stop_loss': current_price - atr_val * self.params['stop_loss_atr'],
                    'take_profit': current_price + atr_val * self.params['take_profit_atr']
                }
            )
        
        elif is_crossunder and volume_confirmed:
            confidence = min(0.5 + abs(fast_curr - slow_curr) / slow_curr * 10, 0.95)
            return Signal(
                timestamp=timestamp,
                symbol=symbol,
                signal_type=SignalType.SELL,
                price=current_price,
                confidence=confidence,
                strategy_name=self.name,
                metadata={
                    'fast_ema': fast_curr,
                    'slow_ema': slow_curr,
                    'atr': atr_val,
                    'stop_loss': current_price + atr_val * self.params['stop_loss_atr'],
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


class ADXTrendStrategy(BaseStrategy):
    """
    ADX + EMA Trend Strategy.
    
    Uses ADX to identify strong trends and EMA for direction.
    Only trades when ADX indicates a strong trend (above threshold).
    
    Parameters:
        adx_period: ADX calculation period (default: 14)
        adx_threshold: Minimum ADX for trend strength (default: 25)
        ema_period: EMA period for trend direction (default: 20)
        use_di_confirmation: Use DI+/- crossover confirmation (default: True)
    """
    
    def __init__(self,
                 name: str = "ADX_Trend",
                 params: Optional[Dict[str, Any]] = None,
                 position_sizer=None):
        
        default_params = {
            'adx_period': 14,
            'adx_threshold': 25,
            'ema_period': 20,
            'use_di_confirmation': True,
            'min_bars': 50,
            'stop_loss_atr': 1.5,
            'take_profit_atr': 2.5
        }
        
        if params:
            default_params.update(params)
        
        super().__init__(name, default_params, position_sizer)
    
    def _precompute_indicators(self, data: pd.DataFrame) -> None:
        """Precompute ADX and EMA indicators."""
        self.adx_data = adx(data, self.params['adx_period'])
        self.ema_line = ema(data['close'], self.params['ema_period']).values
        self.atr_values = atr(data, 14).values
        
        logger.info(f"Precomputed ADX indicators for {self.name}")
    
    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """Generate signal based on ADX trend strength and EMA direction."""
        if len(data) < self.params['min_bars']:
            return self._no_signal(data)
        
        current_price = data['close'].iloc[-1]
        timestamp = data.index[-1]
        symbol = data.get('symbol', ['UNKNOWN'])[-1] if 'symbol' in data.columns else 'UNKNOWN'
        
        # Calculate indicators
        adx_data = adx(data, self.params['adx_period'])
        ema_line = ema(data['close'], self.params['ema_period']).values
        atr_val = atr(data, 14).values.iloc[-1]
        
        adx_val = adx_data['adx'].values.iloc[-1]
        plus_di = adx_data['plus_di'].values.iloc[-1]
        minus_di = adx_data['minus_di'].values.iloc[-1]
        ema_val = ema_line.iloc[-1]
        
        # Check trend strength
        strong_trend = adx_val > self.params['adx_threshold']
        
        if not strong_trend:
            return self._no_signal(data)
        
        # Determine trend direction
        above_ema = current_price > ema_val
        di_bullish = plus_di > minus_di if self.params['use_di_confirmation'] else True
        di_bearish = minus_di > plus_di if self.params['use_di_confirmation'] else True
        
        # Generate signals
        if above_ema and di_bullish:
            confidence = min(adx_val / 50, 0.95)
            return Signal(
                timestamp=timestamp,
                symbol=symbol,
                signal_type=SignalType.BUY,
                price=current_price,
                confidence=confidence,
                strategy_name=self.name,
                metadata={
                    'adx': adx_val,
                    'plus_di': plus_di,
                    'minus_di': minus_di,
                    'ema': ema_val,
                    'atr': atr_val,
                    'stop_loss': current_price - atr_val * self.params['stop_loss_atr'],
                    'take_profit': current_price + atr_val * self.params['take_profit_atr']
                }
            )
        
        elif not above_ema and di_bearish:
            confidence = min(adx_val / 50, 0.95)
            return Signal(
                timestamp=timestamp,
                symbol=symbol,
                signal_type=SignalType.SELL,
                price=current_price,
                confidence=confidence,
                strategy_name=self.name,
                metadata={
                    'adx': adx_val,
                    'plus_di': plus_di,
                    'minus_di': minus_di,
                    'ema': ema_val,
                    'atr': atr_val,
                    'stop_loss': current_price + atr_val * self.params['stop_loss_atr'],
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


class BreakoutStrategy(BaseStrategy):
    """
    Breakout Strategy with Volume Confirmation.
    
    Enters long when price breaks above resistance with volume.
    Enters short when price breaks below support with volume.
    
    Parameters:
        lookback_period: Period for finding support/resistance (default: 20)
        volume_confirm: Require volume confirmation (default: True)
        volume_mult: Volume multiplier threshold (default: 1.5)
        breakout_threshold: Minimum breakout percentage (default: 0.01)
        use_atr_filter: Filter breakouts by ATR (default: True)
    """
    
    def __init__(self,
                 name: str = "Breakout",
                 params: Optional[Dict[str, Any]] = None,
                 position_sizer=None):
        
        default_params = {
            'lookback_period': 20,
            'volume_confirm': True,
            'volume_mult': 1.5,
            'breakout_threshold': 0.01,
            'use_atr_filter': True,
            'min_bars': 30,
            'stop_loss_atr': 2.0,
            'take_profit_atr': 3.0
        }
        
        if params:
            default_params.update(params)
        
        super().__init__(name, default_params, position_sizer)
    
    def _precompute_indicators(self, data: pd.DataFrame) -> None:
        """Precompute support/resistance levels."""
        self.atr_values = atr(data, 14).values
        logger.info(f"Precomputed breakout indicators for {self.name}")
    
    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """Generate breakout signals."""
        if len(data) < self.params['min_bars']:
            return self._no_signal(data)
        
        current_price = data['close'].iloc[-1]
        timestamp = data.index[-1]
        symbol = data.get('symbol', ['UNKNOWN'])[-1] if 'symbol' in data.columns else 'UNKNOWN'
        
        period = self.params['lookback_period']
        
        # Find support and resistance
        resistance = data['high'].tail(period).max()
        support = data['low'].tail(period).min()
        
        # Calculate ATR
        atr_val = atr(data, 14).values.iloc[-1]
        
        # Volume check
        volume_confirmed = True
        if self.params['volume_confirm'] and 'volume' in data.columns:
            avg_volume = data['volume'].tail(period).mean()
            current_volume = data['volume'].iloc[-1]
            volume_confirmed = current_volume > avg_volume * self.params['volume_mult']
        
        # ATR filter - avoid breakouts in low volatility
        if self.params['use_atr_filter']:
            atr_pct = atr_val / current_price
            if atr_pct < 0.005:  # Less than 0.5% ATR
                return self._no_signal(data)
        
        # Check for breakout
        breakout_pct = self.params['breakout_threshold']
        
        long_breakout = (current_price > resistance * (1 + breakout_pct)) and volume_confirmed
        short_breakout = (current_price < support * (1 - breakout_pct)) and volume_confirmed
        
        # Avoid false breakouts - check if price is holding
        if long_breakout or short_breakout:
            # Require close near the high for long, near low for short
            if long_breakout:
                bar_strength = (current_price - data['low'].iloc[-1]) / (data['high'].iloc[-1] - data['low'].iloc[-1] + 1e-10)
                if bar_strength < 0.6:  # Close not in upper 40% of range
                    long_breakout = False
            else:
                bar_strength = (data['high'].iloc[-1] - current_price) / (data['high'].iloc[-1] - data['low'].iloc[-1] + 1e-10)
                if bar_strength < 0.6:  # Close not in lower 40% of range
                    short_breakout = False
        
        if long_breakout:
            confidence = min(0.5 + (current_price - resistance) / resistance * 10, 0.95)
            return Signal(
                timestamp=timestamp,
                symbol=symbol,
                signal_type=SignalType.BUY,
                price=current_price,
                confidence=confidence,
                strategy_name=self.name,
                metadata={
                    'resistance': resistance,
                    'support': support,
                    'atr': atr_val,
                    'breakout_type': 'resistance',
                    'stop_loss': current_price - atr_val * self.params['stop_loss_atr'],
                    'take_profit': current_price + atr_val * self.params['take_profit_atr']
                }
            )
        
        elif short_breakout:
            confidence = min(0.5 + (support - current_price) / support * 10, 0.95)
            return Signal(
                timestamp=timestamp,
                symbol=symbol,
                signal_type=SignalType.SELL,
                price=current_price,
                confidence=confidence,
                strategy_name=self.name,
                metadata={
                    'resistance': resistance,
                    'support': support,
                    'atr': atr_val,
                    'breakout_type': 'support',
                    'stop_loss': current_price + atr_val * self.params['stop_loss_atr'],
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


class SuperTrendStrategy(BaseStrategy):
    """
    SuperTrend Strategy.
    
    Uses SuperTrend indicator for trend following.
    Buy when price closes above SuperTrend, sell when below.
    
    Parameters:
        atr_period: ATR period for SuperTrend (default: 10)
        factor: ATR multiplier (default: 3.0)
        use_confirmation: Wait for confirmation candle (default: True)
    """
    
    def __init__(self,
                 name: str = "SuperTrend",
                 params: Optional[Dict[str, Any]] = None,
                 position_sizer=None):
        
        default_params = {
            'atr_period': 10,
            'factor': 3.0,
            'use_confirmation': True,
            'min_bars': 30,
            'stop_loss_atr': 1.5,
            'take_profit_atr': 2.5
        }
        
        if params:
            default_params.update(params)
        
        super().__init__(name, default_params, position_sizer)
        self.prev_direction = 0
    
    def _precompute_indicators(self, data: pd.DataFrame) -> None:
        """Precompute SuperTrend indicator."""
        self.supertrend_data = supertrend(data, self.params['atr_period'], self.params['factor'])
        logger.info(f"Precomputed SuperTrend indicators for {self.name}")
    
    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """Generate signals based on SuperTrend direction changes."""
        if len(data) < self.params['min_bars']:
            return self._no_signal(data)
        
        current_price = data['close'].iloc[-1]
        timestamp = data.index[-1]
        symbol = data.get('symbol', ['UNKNOWN'])[-1] if 'symbol' in data.columns else 'UNKNOWN'
        
        # Calculate SuperTrend
        st_data = supertrend(data, self.params['atr_period'], self.params['factor'])
        direction = st_data['direction'].values
        supertrend_val = st_data['supertrend'].values
        
        curr_dir = direction.iloc[-1]
        prev_dir = direction.iloc[-2] if len(direction) > 1 else 0
        
        atr_val = atr(data, 14).values.iloc[-1]
        
        # Direction change detection
        bullish_change = (curr_dir == 1) and (prev_dir == -1)
        bearish_change = (curr_dir == -1) and (prev_dir == 1)
        
        # Confirmation - price should be on correct side
        if self.params['use_confirmation']:
            if bullish_change and current_price < supertrend_val.iloc[-1]:
                bullish_change = False
            if bearish_change and current_price > supertrend_val.iloc[-1]:
                bearish_change = False
        
        if bullish_change:
            return Signal(
                timestamp=timestamp,
                symbol=symbol,
                signal_type=SignalType.BUY,
                price=current_price,
                confidence=0.75,
                strategy_name=self.name,
                metadata={
                    'supertrend': supertrend_val.iloc[-1],
                    'direction': curr_dir,
                    'atr': atr_val,
                    'stop_loss': supertrend_val.iloc[-1] - atr_val * 0.5,
                    'take_profit': current_price + atr_val * self.params['take_profit_atr']
                }
            )
        
        elif bearish_change:
            return Signal(
                timestamp=timestamp,
                symbol=symbol,
                signal_type=SignalType.SELL,
                price=current_price,
                confidence=0.75,
                strategy_name=self.name,
                metadata={
                    'supertrend': supertrend_val.iloc[-1],
                    'direction': curr_dir,
                    'atr': atr_val,
                    'stop_loss': supertrend_val.iloc[-1] + atr_val * 0.5,
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


class MACDTrendStrategy(BaseStrategy):
    """
    MACD Trend Strategy.
    
    Uses MACD for trend identification and signal generation.
    Buy when MACD crosses above signal line, sell when below.
    
    Parameters:
        fast: Fast EMA period (default: 12)
        slow: Slow EMA period (default: 26)
        signal: Signal line period (default: 9)
        use_histogram: Use histogram for confirmation (default: True)
        trend_filter_ema: EMA period for trend filter (default: 200, 0 to disable)
    """
    
    def __init__(self,
                 name: str = "MACD_Trend",
                 params: Optional[Dict[str, Any]] = None,
                 position_sizer=None):
        
        default_params = {
            'fast': 12,
            'slow': 26,
            'signal': 9,
            'use_histogram': True,
            'trend_filter_ema': 200,
            'min_bars': 50,
            'stop_loss_atr': 2.0,
            'take_profit_atr': 3.0
        }
        
        if params:
            default_params.update(params)
        
        super().__init__(name, default_params, position_sizer)
    
    def _precompute_indicators(self, data: pd.DataFrame) -> None:
        """Precompute MACD indicators."""
        self.macd_data = macd(data['close'], self.params['fast'], self.params['slow'], self.params['signal'])
        if self.params['trend_filter_ema'] > 0:
            self.trend_ema = ema(data['close'], self.params['trend_filter_ema']).values
        self.atr_values = atr(data, 14).values
        logger.info(f"Precomputed MACD indicators for {self.name}")
    
    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """Generate signals based on MACD crossovers."""
        if len(data) < self.params['min_bars']:
            return self._no_signal(data)
        
        current_price = data['close'].iloc[-1]
        timestamp = data.index[-1]
        symbol = data.get('symbol', ['UNKNOWN'])[-1] if 'symbol' in data.columns else 'UNKNOWN'
        
        # Calculate MACD
        macd_data = macd(data['close'], self.params['fast'], self.params['slow'], self.params['signal'])
        macd_line = macd_data['macd'].values
        signal_line = macd_data['signal'].values
        histogram = macd_data['histogram'].values
        
        # Trend filter
        trend_aligned = True
        if self.params['trend_filter_ema'] > 0:
            trend_ema = ema(data['close'], self.params['trend_filter_ema']).values.iloc[-1]
            trend_aligned = current_price > trend_ema
        
        atr_val = atr(data, 14).values.iloc[-1]
        
        # Detect crossovers
        macd_curr = macd_line.iloc[-1]
        macd_prev = macd_line.iloc[-2]
        signal_curr = signal_line.iloc[-1]
        signal_prev = signal_line.iloc[-2]
        
        bullish_cross = (macd_prev <= signal_prev) and (macd_curr > signal_curr)
        bearish_cross = (macd_prev >= signal_prev) and (macd_curr < signal_curr)
        
        # Histogram confirmation
        hist_confirmed = True
        if self.params['use_histogram']:
            hist_curr = histogram.iloc[-1]
            hist_prev = histogram.iloc[-2]
            if bullish_cross:
                hist_confirmed = hist_curr > hist_prev  # Histogram increasing
            elif bearish_cross:
                hist_confirmed = hist_curr < hist_prev  # Histogram decreasing
        
        if bullish_cross and hist_confirmed and trend_aligned:
            confidence = min(0.5 + abs(macd_curr) / abs(signal_curr) * 0.3, 0.95)
            return Signal(
                timestamp=timestamp,
                symbol=symbol,
                signal_type=SignalType.BUY,
                price=current_price,
                confidence=confidence,
                strategy_name=self.name,
                metadata={
                    'macd': macd_curr,
                    'signal': signal_curr,
                    'histogram': histogram.iloc[-1],
                    'atr': atr_val,
                    'stop_loss': current_price - atr_val * self.params['stop_loss_atr'],
                    'take_profit': current_price + atr_val * self.params['take_profit_atr']
                }
            )
        
        elif bearish_cross and hist_confirmed:
            confidence = min(0.5 + abs(macd_curr) / abs(signal_curr) * 0.3, 0.95)
            return Signal(
                timestamp=timestamp,
                symbol=symbol,
                signal_type=SignalType.SELL,
                price=current_price,
                confidence=confidence,
                strategy_name=self.name,
                metadata={
                    'macd': macd_curr,
                    'signal': signal_curr,
                    'histogram': histogram.iloc[-1],
                    'atr': atr_val,
                    'stop_loss': current_price + atr_val * self.params['stop_loss_atr'],
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


# Export all strategies
__all__ = [
    'EMACrossoverStrategy',
    'ADXTrendStrategy', 
    'BreakoutStrategy',
    'SuperTrendStrategy',
    'MACDTrendStrategy'
]
