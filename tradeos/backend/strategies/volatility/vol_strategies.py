"""
TradeOS Volatility-Based Strategies
===================================
Collection of volatility-based trading strategies.

Author: TradeOS Team
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
import logging

from ..base_strategy import BaseStrategy, Signal, SignalType, Position, Trade, BacktestResult, ATRPositionSizer
from ..indicators.technical import (
    atr, bollinger_bands, keltner_channels, historical_volatility,
    adx, ema, sma, calculate_slope, rolling_zscore
)

logger = logging.getLogger(__name__)


class VolatilityRegime(Enum):
    """Volatility regime classification."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"


class VolatilityBreakoutStrategy(BaseStrategy):
    """
    Volatility Breakout Strategy.
    
    Enters when price breaks out during high volatility expansion.
    Uses Bollinger Band squeeze and expansion detection.
    
    Parameters:
        bb_period: Bollinger Bands period (default: 20)
        bb_std: Standard deviation multiplier (default: 2.0)
        squeeze_threshold: Bandwidth threshold for squeeze detection (default: 6.0)
        volume_confirm: Require volume confirmation (default: True)
        min_expansion: Minimum volatility expansion ratio (default: 1.2)
    """
    
    def __init__(self,
                 name: str = "Volatility_Breakout",
                 params: Optional[Dict[str, Any]] = None,
                 position_sizer=None):
        
        default_params = {
            'bb_period': 20,
            'bb_std': 2.0,
            'squeeze_threshold': 6.0,
            'volume_confirm': True,
            'min_expansion': 1.2,
            'min_bars': 50,
            'stop_loss_atr': 2.0,
            'take_profit_atr': 3.0
        }
        
        if params:
            default_params.update(params)
        
        super().__init__(name, default_params, position_sizer)
        self.in_squeeze = False
        self.squeeze_low = None
    
    def _precompute_indicators(self, data: pd.DataFrame) -> None:
        """Precompute Bollinger Bands and volatility metrics."""
        self.bb_data = bollinger_bands(data['close'], self.params['bb_period'], self.params['bb_std'])
        self.atr_values = atr(data, 14).values
        self.hv_values = historical_volatility(data['close'], 20).values
        logger.info(f"Precomputed volatility indicators for {self.name}")
    
    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """Generate volatility breakout signals."""
        if len(data) < self.params['min_bars']:
            return self._no_signal(data)
        
        current_price = data['close'].iloc[-1]
        current_high = data['high'].iloc[-1]
        current_low = data['low'].iloc[-1]
        timestamp = data.index[-1]
        symbol = data.get('symbol', ['UNKNOWN'])[-1] if 'symbol' in data.columns else 'UNKNOWN'
        
        # Calculate Bollinger Bands
        bb_data = bollinger_bands(data['close'], self.params['bb_period'], self.params['bb_std'])
        bandwidth = bb_data['bandwidth'].values
        upper = bb_data['upper'].values
        lower = bb_data['lower'].values
        
        current_bandwidth = bandwidth.iloc[-1]
        prev_bandwidth = bandwidth.iloc[-2] if len(bandwidth) > 1 else current_bandwidth
        
        # Calculate ATR
        atr_val = atr(data, 14).values.iloc[-1]
        
        # Detect squeeze
        squeeze_threshold = self.params['squeeze_threshold']
        is_squeeze = current_bandwidth < squeeze_threshold
        
        if is_squeeze and not self.in_squeeze:
            self.in_squeeze = True
            self.squeeze_low = current_bandwidth
        elif not is_squeeze and self.in_squeeze:
            # Squeeze released - potential breakout
            expansion_ratio = current_bandwidth / self.squeeze_low if self.squeeze_low else 1
            
            if expansion_ratio >= self.params['min_expansion']:
                # Check for breakout direction
                prev_upper = upper.iloc[-2]
                prev_lower = lower.iloc[-2]
                
                volume_confirmed = True
                if self.params['volume_confirm'] and 'volume' in data.columns:
                    avg_vol = data['volume'].tail(20).mean()
                    current_vol = data['volume'].iloc[-1]
                    volume_confirmed = current_vol > avg_vol * 1.3
                
                # Bullish breakout
                if current_high > prev_upper and volume_confirmed:
                    self.in_squeeze = False
                    confidence = min(0.5 + (expansion_ratio - 1) * 0.3, 0.9)
                    return Signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        signal_type=SignalType.BUY,
                        price=current_price,
                        confidence=confidence,
                        strategy_name=self.name,
                        metadata={
                            'bandwidth': current_bandwidth,
                            'expansion_ratio': expansion_ratio,
                            'atr': atr_val,
                            'breakout_type': 'squeeze_release_bullish',
                            'stop_loss': current_price - atr_val * self.params['stop_loss_atr'],
                            'take_profit': current_price + atr_val * self.params['take_profit_atr']
                        }
                    )
                
                # Bearish breakout
                elif current_low < prev_lower and volume_confirmed:
                    self.in_squeeze = False
                    confidence = min(0.5 + (expansion_ratio - 1) * 0.3, 0.9)
                    return Signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        signal_type=SignalType.SELL,
                        price=current_price,
                        confidence=confidence,
                        strategy_name=self.name,
                        metadata={
                            'bandwidth': current_bandwidth,
                            'expansion_ratio': expansion_ratio,
                            'atr': atr_val,
                            'breakout_type': 'squeeze_release_bearish',
                            'stop_loss': current_price + atr_val * self.params['stop_loss_atr'],
                            'take_profit': current_price - atr_val * self.params['take_profit_atr']
                        }
                    )
        
        # Also check for volatility expansion without squeeze
        if not is_squeeze and not self.in_squeeze:
            bandwidth_change = (current_bandwidth - prev_bandwidth) / prev_bandwidth if prev_bandwidth > 0 else 0
            
            if bandwidth_change > 0.3:  # 30% bandwidth expansion
                # Check for directional move
                price_change = (current_price - data['close'].iloc[-5]) / data['close'].iloc[-5]
                
                if price_change > 0.02:  # 2% up move
                    return Signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        signal_type=SignalType.BUY,
                        price=current_price,
                        confidence=0.6,
                        strategy_name=self.name,
                        metadata={
                            'bandwidth_change': bandwidth_change,
                            'price_change': price_change,
                            'atr': atr_val,
                            'breakout_type': 'vol_expansion'
                        }
                    )
                elif price_change < -0.02:  # 2% down move
                    return Signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        signal_type=SignalType.SELL,
                        price=current_price,
                        confidence=0.6,
                        strategy_name=self.name,
                        metadata={
                            'bandwidth_change': bandwidth_change,
                            'price_change': price_change,
                            'atr': atr_val,
                            'breakout_type': 'vol_expansion'
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


class ATRChannelStrategy(BaseStrategy):
    """
    ATR Channel Breakout Strategy.
    
    Uses ATR-based channels for breakout detection.
    
    Parameters:
        atr_period: ATR calculation period (default: 14)
        atr_mult: ATR multiplier for channel width (default: 2.0)
        channel_period: Period for channel baseline (default: 20)
        use_keltner: Use Keltner Channels instead (default: False)
    """
    
    def __init__(self,
                 name: str = "ATR_Channel",
                 params: Optional[Dict[str, Any]] = None,
                 position_sizer=None):
        
        default_params = {
            'atr_period': 14,
            'atr_mult': 2.0,
            'channel_period': 20,
            'use_keltner': False,
            'min_bars': 30,
            'stop_loss_atr': 1.5,
            'take_profit_atr': 2.5
        }
        
        if params:
            default_params.update(params)
        
        super().__init__(name, default_params, position_sizer)
    
    def _precompute_indicators(self, data: pd.DataFrame) -> None:
        """Precompute ATR channels."""
        if self.params['use_keltner']:
            self.kc_data = keltner_channels(
                data, 
                self.params['channel_period'],
                self.params['atr_period'],
                self.params['atr_mult']
            )
        else:
            self.atr_values = atr(data, self.params['atr_period']).values
            self.channel_baseline = ema(data['close'], self.params['channel_period']).values
        
        logger.info(f"Precomputed ATR channels for {self.name}")
    
    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """Generate signals based on ATR channel breakouts."""
        if len(data) < self.params['min_bars']:
            return self._no_signal(data)
        
        current_price = data['close'].iloc[-1]
        current_high = data['high'].iloc[-1]
        current_low = data['low'].iloc[-1]
        timestamp = data.index[-1]
        symbol = data.get('symbol', ['UNKNOWN'])[-1] if 'symbol' in data.columns else 'UNKNOWN'
        
        if self.params['use_keltner']:
            kc_data = keltner_channels(
                data,
                self.params['channel_period'],
                self.params['atr_period'],
                self.params['atr_mult']
            )
            upper = kc_data['upper'].values.iloc[-1]
            lower = kc_data['lower'].values.iloc[-1]
            middle = kc_data['middle'].values.iloc[-1]
        else:
            atr_val = atr(data, self.params['atr_period']).values.iloc[-1]
            middle = ema(data['close'], self.params['channel_period']).values.iloc[-1]
            upper = middle + atr_val * self.params['atr_mult']
            lower = middle - atr_val * self.params['atr_mult']
        
        atr_val = atr(data, 14).values.iloc[-1]
        
        # Check for channel breakouts
        prev_high = data['high'].iloc[-2]
        prev_low = data['low'].iloc[-2]
        
        breakout_up = (prev_high <= upper) and (current_high > upper)
        breakout_down = (prev_low >= lower) and (current_low < lower)
        
        if breakout_up:
            confidence = min(0.5 + (current_high - upper) / atr_val * 0.3, 0.9)
            return Signal(
                timestamp=timestamp,
                symbol=symbol,
                signal_type=SignalType.BUY,
                price=current_price,
                confidence=confidence,
                strategy_name=self.name,
                metadata={
                    'channel_upper': upper,
                    'channel_lower': lower,
                    'channel_middle': middle,
                    'atr': atr_val,
                    'breakout_type': 'atr_channel_up',
                    'stop_loss': middle - atr_val * 0.5,
                    'take_profit': current_price + atr_val * self.params['take_profit_atr']
                }
            )
        
        elif breakout_down:
            confidence = min(0.5 + (lower - current_low) / atr_val * 0.3, 0.9)
            return Signal(
                timestamp=timestamp,
                symbol=symbol,
                signal_type=SignalType.SELL,
                price=current_price,
                confidence=confidence,
                strategy_name=self.name,
                metadata={
                    'channel_upper': upper,
                    'channel_lower': lower,
                    'channel_middle': middle,
                    'atr': atr_val,
                    'breakout_type': 'atr_channel_down',
                    'stop_loss': middle + atr_val * 0.5,
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


class VolatilityRegimeStrategy(BaseStrategy):
    """
    Volatility Regime-Based Strategy.
    
    Adjusts trading behavior based on volatility regime.
    Can be used as a filter for other strategies.
    
    Parameters:
        hv_period: Historical volatility period (default: 20)
        low_vol_threshold: Threshold for low volatility (default: 15)
        high_vol_threshold: Threshold for high volatility (default: 40)
        extreme_vol_threshold: Threshold for extreme volatility (default: 60)
        regime_lookback: Lookback for regime persistence (default: 5)
    """
    
    def __init__(self,
                 name: str = "Volatility_Regime",
                 params: Optional[Dict[str, Any]] = None,
                 position_sizer=None):
        
        default_params = {
            'hv_period': 20,
            'low_vol_threshold': 15,
            'high_vol_threshold': 40,
            'extreme_vol_threshold': 60,
            'regime_lookback': 5,
            'min_bars': 50,
            'trade_in_low_vol': True,
            'trade_in_high_vol': False,
            'reduce_size_high_vol': True,
            'size_reduction': 0.5
        }
        
        if params:
            default_params.update(params)
        
        super().__init__(name, default_params, position_sizer)
        self.current_regime = VolatilityRegime.NORMAL
        self.regime_history: List[VolatilityRegime] = []
    
    def _precompute_indicators(self, data: pd.DataFrame) -> None:
        """Precompute volatility metrics."""
        self.hv_values = historical_volatility(data['close'], self.params['hv_period']).values
        self.atr_values = atr(data, 14).values
        logger.info(f"Precomputed volatility regime indicators for {self.name}")
    
    def classify_regime(self, volatility: float) -> VolatilityRegime:
        """Classify volatility regime."""
        if volatility < self.params['low_vol_threshold']:
            return VolatilityRegime.LOW
        elif volatility > self.params['extreme_vol_threshold']:
            return VolatilityRegime.EXTREME
        elif volatility > self.params['high_vol_threshold']:
            return VolatilityRegime.HIGH
        else:
            return VolatilityRegime.NORMAL
    
    def get_regime(self) -> VolatilityRegime:
        """Get current volatility regime."""
        return self.current_regime
    
    def should_trade(self) -> bool:
        """Check if trading should be allowed in current regime."""
        if self.current_regime == VolatilityRegime.LOW:
            return self.params['trade_in_low_vol']
        elif self.current_regime == VolatilityRegime.HIGH:
            return self.params['trade_in_high_vol']
        elif self.current_regime == VolatilityRegime.EXTREME:
            return False
        else:
            return True
    
    def get_position_multiplier(self) -> float:
        """Get position size multiplier based on regime."""
        if self.current_regime == VolatilityRegime.HIGH and self.params['reduce_size_high_vol']:
            return self.params['size_reduction']
        elif self.current_regime == VolatilityRegime.EXTREME:
            return 0.0
        else:
            return 1.0
    
    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """Generate regime classification signal (not a trade signal)."""
        if len(data) < self.params['min_bars']:
            return self._no_signal(data)
        
        current_price = data['close'].iloc[-1]
        timestamp = data.index[-1]
        symbol = data.get('symbol', ['UNKNOWN'])[-1] if 'symbol' in data.columns else 'UNKNOWN'
        
        # Calculate historical volatility
        hv_values = historical_volatility(data['close'], self.params['hv_period']).values
        current_hv = hv_values.iloc[-1]
        
        # Classify regime
        new_regime = self.classify_regime(current_hv)
        self.regime_history.append(new_regime)
        
        # Require persistence
        if len(self.regime_history) >= self.params['regime_lookback']:
            recent_regimes = self.regime_history[-self.params['regime_lookback']:]
            # Check if all recent regimes are the same
            if all(r == new_regime for r in recent_regimes):
                if new_regime != self.current_regime:
                    self.current_regime = new_regime
                    logger.info(f"Volatility regime changed to {new_regime.value} (HV: {current_hv:.2f}%)")
        
        # Return informational signal about regime
        return Signal(
            timestamp=timestamp,
            symbol=symbol,
            signal_type=SignalType.HOLD,
            price=current_price,
            confidence=0.0,
            strategy_name=self.name,
            metadata={
                'regime': self.current_regime.value,
                'historical_volatility': current_hv,
                'should_trade': self.should_trade(),
                'position_multiplier': self.get_position_multiplier()
            }
        )
    
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


class VolatilityContractionStrategy(BaseStrategy):
    """
    Volatility Contraction Strategy.
    
    Identifies periods of low volatility that often precede large moves.
    
    Parameters:
        short_atr_period: Short-term ATR period (default: 7)
        long_atr_period: Long-term ATR period (default: 14)
        contraction_threshold: ATR ratio threshold for contraction (default: 0.8)
        min_contraction_bars: Minimum bars in contraction (default: 3)
        volume_filter: Require decreasing volume during contraction (default: True)
    """
    
    def __init__(self,
                 name: str = "Volatility_Contraction",
                 params: Optional[Dict[str, Any]] = None,
                 position_sizer=None):
        
        default_params = {
            'short_atr_period': 7,
            'long_atr_period': 14,
            'contraction_threshold': 0.8,
            'min_contraction_bars': 3,
            'volume_filter': True,
            'min_bars': 30,
            'stop_loss_atr': 2.0,
            'take_profit_atr': 4.0
        }
        
        if params:
            default_params.update(params)
        
        super().__init__(name, default_params, position_sizer)
        self.contraction_count = 0
        self.contraction_active = False
    
    def _precompute_indicators(self, data: pd.DataFrame) -> None:
        """Precompute ATR indicators."""
        self.short_atr = atr(data, self.params['short_atr_period']).values
        self.long_atr = atr(data, self.params['long_atr_period']).values
        logger.info(f"Precomputed volatility contraction indicators for {self.name}")
    
    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """Generate signals based on volatility contraction."""
        if len(data) < self.params['min_bars']:
            return self._no_signal(data)
        
        current_price = data['close'].iloc[-1]
        current_high = data['high'].iloc[-1]
        current_low = data['low'].iloc[-1]
        timestamp = data.index[-1]
        symbol = data.get('symbol', ['UNKNOWN'])[-1] if 'symbol' in data.columns else 'UNKNOWN'
        
        # Calculate ATRs
        short_atr = atr(data, self.params['short_atr_period']).values.iloc[-1]
        long_atr = atr(data, self.params['long_atr_period']).values.iloc[-1]
        
        # Check for contraction
        atr_ratio = short_atr / long_atr if long_atr > 0 else 1
        is_contracting = atr_ratio < self.params['contraction_threshold']
        
        # Volume filter
        volume_ok = True
        if self.params['volume_filter'] and 'volume' in data.columns:
            vol_trend = calculate_slope(data['volume'], 10).values.iloc[-1]
            volume_ok = vol_trend < 0  # Decreasing volume
        
        # Track contraction
        if is_contracting and volume_ok:
            self.contraction_count += 1
            if self.contraction_count >= self.params['min_contraction_bars']:
                self.contraction_active = True
        else:
            # Contraction ended - potential breakout
            if self.contraction_active and self.contraction_count >= self.params['min_contraction_bars']:
                self.contraction_active = False
                
                # Determine direction based on price action
                price_range = data['high'].tail(self.contraction_count).max() - data['low'].tail(self.contraction_count).min()
                price_mid = (data['high'].tail(self.contraction_count).max() + data['low'].tail(self.contraction_count).min()) / 2
                
                # Check which side was breached
                broke_high = current_high > data['high'].tail(self.contraction_count).max() * 0.999
                broke_low = current_low < data['low'].tail(self.contraction_count).min() * 1.001
                
                atr_val = atr(data, 14).values.iloc[-1]
                
                if broke_high and not broke_low:
                    self.contraction_count = 0
                    return Signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        signal_type=SignalType.BUY,
                        price=current_price,
                        confidence=0.7,
                        strategy_name=self.name,
                        metadata={
                            'contraction_bars': self.contraction_count,
                            'atr_ratio': atr_ratio,
                            'atr': atr_val,
                            'breakout_type': 'contraction_high',
                            'stop_loss': current_price - atr_val * self.params['stop_loss_atr'],
                            'take_profit': current_price + atr_val * self.params['take_profit_atr']
                        }
                    )
                
                elif broke_low and not broke_high:
                    self.contraction_count = 0
                    return Signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        signal_type=SignalType.SELL,
                        price=current_price,
                        confidence=0.7,
                        strategy_name=self.name,
                        metadata={
                            'contraction_bars': self.contraction_count,
                            'atr_ratio': atr_ratio,
                            'atr': atr_val,
                            'breakout_type': 'contraction_low',
                            'stop_loss': current_price + atr_val * self.params['stop_loss_atr'],
                            'take_profit': current_price - atr_val * self.params['take_profit_atr']
                        }
                    )
            
            self.contraction_count = 0
        
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


class GapVolatilityStrategy(BaseStrategy):
    """
    Gap and Volatility Strategy.
    
    Trades gaps with volatility-based position sizing and targets.
    
    Parameters:
        min_gap_pct: Minimum gap percentage (default: 1.0)
        max_gap_pct: Maximum gap percentage to trade (default: 5.0)
        fill_threshold: Gap fill percentage for exit (default: 0.5)
        use_atr_filter: Only trade if ATR is reasonable (default: True)
    """
    
    def __init__(self,
                 name: str = "Gap_Volatility",
                 params: Optional[Dict[str, Any]] = None,
                 position_sizer=None):
        
        default_params = {
            'min_gap_pct': 1.0,
            'max_gap_pct': 5.0,
            'fill_threshold': 0.5,
            'use_atr_filter': True,
            'min_bars': 10,
            'stop_loss_atr': 1.5,
            'take_profit_atr': 2.0
        }
        
        if params:
            default_params.update(params)
        
        super().__init__(name, default_params, position_sizer)
        self.pending_gap = None
    
    def _precompute_indicators(self, data: pd.DataFrame) -> None:
        """Precompute volatility metrics."""
        self.atr_values = atr(data, 14).values
        logger.info(f"Precomputed gap volatility indicators for {self.name}")
    
    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """Generate signals based on gaps."""
        if len(data) < self.params['min_bars']:
            return self._no_signal(data)
        
        current_price = data['close'].iloc[-1]
        current_open = data['open'].iloc[-1]
        prev_close = data['close'].iloc[-2]
        timestamp = data.index[-1]
        symbol = data.get('symbol', ['UNKNOWN'])[-1] if 'symbol' in data.columns else 'UNKNOWN'
        
        # Calculate gap
        gap_pct = (current_open - prev_close) / prev_close * 100
        gap_size = abs(gap_pct)
        
        atr_val = atr(data, 14).values.iloc[-1]
        
        # ATR filter
        if self.params['use_atr_filter']:
            atr_pct = atr_val / prev_close * 100
            if atr_pct < 0.5:  # Too low volatility
                return self._no_signal(data)
        
        # Check if gap is within tradeable range
        min_gap = self.params['min_gap_pct']
        max_gap = self.params['max_gap_pct']
        
        if gap_size < min_gap or gap_size > max_gap:
            return self._no_signal(data)
        
        # Gap up - potential fade (mean reversion)
        if gap_pct > 0:
            # Look for signs of exhaustion
            if current_price < current_open:  # Price already fading
                fill_target = prev_close + (gap_pct * self.params['fill_threshold'] / 100 * prev_close)
                return Signal(
                    timestamp=timestamp,
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    price=current_price,
                    confidence=min(0.5 + gap_pct / 10, 0.85),
                    strategy_name=self.name,
                    metadata={
                        'gap_pct': gap_pct,
                        'gap_type': 'up_gap_fade',
                        'fill_target': fill_target,
                        'atr': atr_val,
                        'stop_loss': current_open + atr_val,
                        'take_profit': fill_target
                    }
                )
        
        # Gap down - potential fade
        else:
            if current_price > current_open:  # Price already bouncing
                fill_target = prev_close - (abs(gap_pct) * self.params['fill_threshold'] / 100 * prev_close)
                return Signal(
                    timestamp=timestamp,
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    price=current_price,
                    confidence=min(0.5 + abs(gap_pct) / 10, 0.85),
                    strategy_name=self.name,
                    metadata={
                        'gap_pct': gap_pct,
                        'gap_type': 'down_gap_fade',
                        'fill_target': fill_target,
                        'atr': atr_val,
                        'stop_loss': current_open - atr_val,
                        'take_profit': fill_target
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
    'VolatilityRegime',
    'VolatilityBreakoutStrategy',
    'ATRChannelStrategy',
    'VolatilityRegimeStrategy',
    'VolatilityContractionStrategy',
    'GapVolatilityStrategy'
]
