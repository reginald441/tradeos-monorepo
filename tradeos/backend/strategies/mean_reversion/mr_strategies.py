"""
TradeOS Mean Reversion Strategies
=================================
Collection of mean reversion trading strategies.

Author: TradeOS Team
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from scipy import stats
import logging

from ..base_strategy import BaseStrategy, Signal, SignalType, Position, Trade, BacktestResult
from ..indicators.technical import (
    rsi, bollinger_bands, ema, sma, atr, rolling_zscore,
    stochastic, williams_r, calculate_slope, calculate_r2
)

logger = logging.getLogger(__name__)


class RSIMeanReversionStrategy(BaseStrategy):
    """
    RSI Mean Reversion Strategy.
    
    Buys when RSI is oversold (below threshold).
    Sells when RSI is overbought (above threshold).
    
    Parameters:
        rsi_period: RSI calculation period (default: 14)
        oversold: Oversold threshold (default: 30)
        overbought: Overbought threshold (default: 70)
        exit_at_middle: Exit when RSI returns to 50 (default: True)
        use_divergence: Look for RSI divergence (default: False)
        trend_filter_ema: EMA for trend alignment (default: 0 for disabled)
    """
    
    def __init__(self,
                 name: str = "RSI_MeanReversion",
                 params: Optional[Dict[str, Any]] = None,
                 position_sizer=None):
        
        default_params = {
            'rsi_period': 14,
            'oversold': 30,
            'overbought': 70,
            'exit_at_middle': True,
            'use_divergence': False,
            'trend_filter_ema': 0,
            'min_bars': 30,
            'stop_loss_atr': 2.0,
            'take_profit_atr': 2.0
        }
        
        if params:
            default_params.update(params)
        
        super().__init__(name, default_params, position_sizer)
        self.prev_rsi = None
    
    def _precompute_indicators(self, data: pd.DataFrame) -> None:
        """Precompute RSI indicator."""
        self.rsi_values = rsi(data['close'], self.params['rsi_period']).values
        self.atr_values = atr(data, 14).values
        logger.info(f"Precomputed RSI indicators for {self.name}")
    
    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """Generate mean reversion signals based on RSI."""
        if len(data) < self.params['min_bars']:
            return self._no_signal(data)
        
        current_price = data['close'].iloc[-1]
        timestamp = data.index[-1]
        symbol = data.get('symbol', ['UNKNOWN'])[-1] if 'symbol' in data.columns else 'UNKNOWN'
        
        # Calculate RSI
        rsi_values = rsi(data['close'], self.params['rsi_period']).values
        current_rsi = rsi_values.iloc[-1]
        prev_rsi = rsi_values.iloc[-2] if len(rsi_values) > 1 else 50
        
        atr_val = atr(data, 14).values.iloc[-1]
        
        # Trend filter
        trend_aligned = True
        if self.params['trend_filter_ema'] > 0:
            trend_ema = ema(data['close'], self.params['trend_filter_ema']).values.iloc[-1]
            trend_aligned = current_price > trend_ema  # Only trade long in uptrend
        
        # Check for oversold condition (buy signal)
        oversold_threshold = self.params['oversold']
        overbought_threshold = self.params['overbought']
        
        # RSI leaving oversold zone
        bullish_reversal = (prev_rsi <= oversold_threshold) and (current_rsi > oversold_threshold)
        
        # RSI leaving overbought zone
        bearish_reversal = (prev_rsi >= overbought_threshold) and (current_rsi < overbought_threshold)
        
        # Check for divergence if enabled
        divergence_confirmed = True
        if self.params['use_divergence']:
            if len(data) >= 20:
                price_low_10 = data['low'].tail(10).min()
                price_low_20 = data['low'].tail(20).min()
                rsi_low_10 = rsi_values.tail(10).min()
                rsi_low_20 = rsi_values.tail(20).min()
                
                # Bullish divergence: price making lower lows, RSI making higher lows
                if price_low_20 < price_low_10 and rsi_low_20 > rsi_low_10:
                    divergence_confirmed = True
                else:
                    divergence_confirmed = False
        
        if bullish_reversal and trend_aligned and divergence_confirmed:
            confidence = min(0.5 + (oversold_threshold - current_rsi) / 20, 0.9)
            return Signal(
                timestamp=timestamp,
                symbol=symbol,
                signal_type=SignalType.BUY,
                price=current_price,
                confidence=confidence,
                strategy_name=self.name,
                metadata={
                    'rsi': current_rsi,
                    'prev_rsi': prev_rsi,
                    'oversold': oversold_threshold,
                    'atr': atr_val,
                    'stop_loss': current_price - atr_val * self.params['stop_loss_atr'],
                    'take_profit': current_price + atr_val * self.params['take_profit_atr']
                }
            )
        
        elif bearish_reversal:
            confidence = min(0.5 + (current_rsi - overbought_threshold) / 20, 0.9)
            return Signal(
                timestamp=timestamp,
                symbol=symbol,
                signal_type=SignalType.SELL,
                price=current_price,
                confidence=confidence,
                strategy_name=self.name,
                metadata={
                    'rsi': current_rsi,
                    'prev_rsi': prev_rsi,
                    'overbought': overbought_threshold,
                    'atr': atr_val,
                    'stop_loss': current_price + atr_val * self.params['stop_loss_atr'],
                    'take_profit': current_price - atr_val * self.params['take_profit_atr']
                }
            )
        
        # Exit at middle RSI if enabled
        if self.params['exit_at_middle'] and self.current_position is not None:
            middle_zone = 45 <= current_rsi <= 55
            if middle_zone:
                return Signal(
                    timestamp=timestamp,
                    symbol=symbol,
                    signal_type=SignalType.CLOSE,
                    price=current_price,
                    confidence=0.8,
                    strategy_name=self.name,
                    metadata={'rsi': current_rsi, 'exit_reason': 'middle_rsi'}
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


class BollingerBandMeanReversionStrategy(BaseStrategy):
    """
    Bollinger Bands Mean Reversion Strategy.
    
    Buys when price touches lower band.
    Sells when price touches upper band.
    
    Parameters:
        bb_period: Bollinger Bands period (default: 20)
        bb_std: Standard deviation multiplier (default: 2.0)
        use_band_width: Only trade when bands are wide enough (default: True)
        min_band_width: Minimum bandwidth percentage (default: 5.0)
        confirmation_candles: Wait for N candles (default: 1)
    """
    
    def __init__(self,
                 name: str = "BB_MeanReversion",
                 params: Optional[Dict[str, Any]] = None,
                 position_sizer=None):
        
        default_params = {
            'bb_period': 20,
            'bb_std': 2.0,
            'use_band_width': True,
            'min_band_width': 5.0,
            'confirmation_candles': 1,
            'min_bars': 30,
            'stop_loss_atr': 2.0,
            'take_profit_atr': 2.0
        }
        
        if params:
            default_params.update(params)
        
        super().__init__(name, default_params, position_sizer)
        self.touch_count = 0
        self.touch_side = None
    
    def _precompute_indicators(self, data: pd.DataFrame) -> None:
        """Precompute Bollinger Bands."""
        self.bb_data = bollinger_bands(data['close'], self.params['bb_period'], self.params['bb_std'])
        self.atr_values = atr(data, 14).values
        logger.info(f"Precomputed Bollinger Bands for {self.name}")
    
    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """Generate mean reversion signals based on Bollinger Bands."""
        if len(data) < self.params['min_bars']:
            return self._no_signal(data)
        
        current_price = data['close'].iloc[-1]
        current_low = data['low'].iloc[-1]
        current_high = data['high'].iloc[-1]
        timestamp = data.index[-1]
        symbol = data.get('symbol', ['UNKNOWN'])[-1] if 'symbol' in data.columns else 'UNKNOWN'
        
        # Calculate Bollinger Bands
        bb_data = bollinger_bands(data['close'], self.params['bb_period'], self.params['bb_std'])
        upper = bb_data['upper'].values.iloc[-1]
        lower = bb_data['lower'].values.iloc[-1]
        middle = bb_data['middle'].values.iloc[-1]
        bandwidth = bb_data['bandwidth'].values.iloc[-1]
        percent_b = bb_data['percent_b'].values.iloc[-1]
        
        atr_val = atr(data, 14).values.iloc[-1]
        
        # Check band width
        if self.params['use_band_width'] and bandwidth < self.params['min_band_width']:
            return self._no_signal(data)
        
        # Check for touches
        touch_lower = current_low <= lower
        touch_upper = current_high >= upper
        
        # Count consecutive touches
        if touch_lower:
            if self.touch_side == 'lower':
                self.touch_count += 1
            else:
                self.touch_side = 'lower'
                self.touch_count = 1
        elif touch_upper:
            if self.touch_side == 'upper':
                self.touch_count += 1
            else:
                self.touch_side = 'upper'
                self.touch_count = 1
        else:
            self.touch_count = 0
            self.touch_side = None
        
        confirmation_met = self.touch_count >= self.params['confirmation_candles']
        
        # Bounce from lower band (buy signal)
        if touch_lower and confirmation_met and current_price > lower:
            confidence = min(0.5 + (lower - current_low) / atr_val * 0.2, 0.9)
            self.touch_count = 0
            return Signal(
                timestamp=timestamp,
                symbol=symbol,
                signal_type=SignalType.BUY,
                price=current_price,
                confidence=confidence,
                strategy_name=self.name,
                metadata={
                    'bb_upper': upper,
                    'bb_lower': lower,
                    'bb_middle': middle,
                    'percent_b': percent_b,
                    'bandwidth': bandwidth,
                    'atr': atr_val,
                    'stop_loss': lower - atr_val * 0.5,
                    'take_profit': middle
                }
            )
        
        # Bounce from upper band (sell signal)
        elif touch_upper and confirmation_met and current_price < upper:
            confidence = min(0.5 + (current_high - upper) / atr_val * 0.2, 0.9)
            self.touch_count = 0
            return Signal(
                timestamp=timestamp,
                symbol=symbol,
                signal_type=SignalType.SELL,
                price=current_price,
                confidence=confidence,
                strategy_name=self.name,
                metadata={
                    'bb_upper': upper,
                    'bb_lower': lower,
                    'bb_middle': middle,
                    'percent_b': percent_b,
                    'bandwidth': bandwidth,
                    'atr': atr_val,
                    'stop_loss': upper + atr_val * 0.5,
                    'take_profit': middle
                }
            )
        
        # Exit at middle band if in position
        if self.current_position is not None:
            near_middle = abs(current_price - middle) / atr_val < 0.5
            if near_middle:
                return Signal(
                    timestamp=timestamp,
                    symbol=symbol,
                    signal_type=SignalType.CLOSE,
                    price=current_price,
                    confidence=0.7,
                    strategy_name=self.name,
                    metadata={'exit_reason': 'middle_band', 'bb_middle': middle}
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


class StatisticalArbitrageStrategy(BaseStrategy):
    """
    Statistical Arbitrage Strategy Framework.
    
    Base class for pairs trading and statistical arbitrage.
    Requires two correlated assets.
    
    Parameters:
        lookback_period: Period for calculating statistics (default: 60)
        entry_zscore: Z-score threshold for entry (default: 2.0)
        exit_zscore: Z-score threshold for exit (default: 0.5)
        min_correlation: Minimum correlation required (default: 0.7)
    """
    
    def __init__(self,
                 name: str = "StatArb",
                 params: Optional[Dict[str, Any]] = None,
                 position_sizer=None):
        
        default_params = {
            'lookback_period': 60,
            'entry_zscore': 2.0,
            'exit_zscore': 0.5,
            'min_correlation': 0.7,
            'min_bars': 100,
            'hedge_ratio_method': 'ols'  # 'ols' or 'price_ratio'
        }
        
        if params:
            default_params.update(params)
        
        super().__init__(name, default_params, position_sizer)
        self.hedge_ratio = 1.0
        self.spread_mean = 0.0
        self.spread_std = 1.0
    
    def _precompute_indicators(self, data: pd.DataFrame) -> None:
        """Precompute spread statistics."""
        # Expects data with two price columns: 'close' and 'close_2'
        if 'close_2' not in data.columns:
            logger.warning(f"StatArb requires 'close_2' column for pair asset")
            return
        
        self._calculate_spread_stats(data)
        logger.info(f"Precomputed spread statistics for {self.name}")
    
    def _calculate_spread_stats(self, data: pd.DataFrame) -> Tuple[float, float, float]:
        """Calculate hedge ratio and spread statistics."""
        asset1 = data['close']
        asset2 = data['close_2']
        
        period = self.params['lookback_period']
        
        if self.params['hedge_ratio_method'] == 'ols':
            # OLS regression for hedge ratio
            x = asset2.tail(period).values
            y = asset1.tail(period).values
            slope, intercept, r_value, _, _ = stats.linregress(x, y)
            self.hedge_ratio = slope
            correlation = r_value
        else:
            # Simple price ratio
            self.hedge_ratio = asset1.iloc[-1] / asset2.iloc[-1]
            correlation = asset1.tail(period).corr(asset2.tail(period))
        
        # Calculate spread
        spread = asset1 - self.hedge_ratio * asset2
        self.spread_mean = spread.tail(period).mean()
        self.spread_std = spread.tail(period).std()
        
        return self.hedge_ratio, correlation, self.spread_std
    
    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """Generate statistical arbitrage signals."""
        if len(data) < self.params['min_bars']:
            return self._no_signal(data)
        
        if 'close_2' not in data.columns:
            return self._no_signal(data)
        
        current_price = data['close'].iloc[-1]
        timestamp = data.index[-1]
        symbol = data.get('symbol', ['UNKNOWN'])[-1] if 'symbol' in data.columns else 'UNKNOWN'
        
        # Calculate spread statistics
        hedge_ratio, correlation, spread_std = self._calculate_spread_stats(data)
        
        # Check correlation
        if abs(correlation) < self.params['min_correlation']:
            return self._no_signal(data)
        
        # Calculate current spread and z-score
        asset2_price = data['close_2'].iloc[-1]
        current_spread = current_price - hedge_ratio * asset2_price
        zscore = (current_spread - self.spread_mean) / spread_std if spread_std > 0 else 0
        
        entry_threshold = self.params['entry_zscore']
        exit_threshold = self.params['exit_zscore']
        
        # Long asset1, short asset2 when spread is below -entry
        if zscore < -entry_threshold:
            confidence = min(0.5 + abs(zscore) / 4, 0.9)
            return Signal(
                timestamp=timestamp,
                symbol=symbol,
                signal_type=SignalType.BUY,
                price=current_price,
                confidence=confidence,
                strategy_name=self.name,
                metadata={
                    'zscore': zscore,
                    'spread': current_spread,
                    'hedge_ratio': hedge_ratio,
                    'correlation': correlation,
                    'pair_price': asset2_price,
                    'spread_mean': self.spread_mean,
                    'spread_std': spread_std
                }
            )
        
        # Short asset1, long asset2 when spread is above entry
        elif zscore > entry_threshold:
            confidence = min(0.5 + abs(zscore) / 4, 0.9)
            return Signal(
                timestamp=timestamp,
                symbol=symbol,
                signal_type=SignalType.SELL,
                price=current_price,
                confidence=confidence,
                strategy_name=self.name,
                metadata={
                    'zscore': zscore,
                    'spread': current_spread,
                    'hedge_ratio': hedge_ratio,
                    'correlation': correlation,
                    'pair_price': asset2_price,
                    'spread_mean': self.spread_mean,
                    'spread_std': spread_std
                }
            )
        
        # Exit when spread returns to mean
        if self.current_position is not None and abs(zscore) < exit_threshold:
            return Signal(
                timestamp=timestamp,
                symbol=symbol,
                signal_type=SignalType.CLOSE,
                price=current_price,
                confidence=0.8,
                strategy_name=self.name,
                metadata={
                    'zscore': zscore,
                    'exit_reason': 'mean_reversion',
                    'spread': current_spread
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


class ZScoreMeanReversionStrategy(BaseStrategy):
    """
    Z-Score Mean Reversion Strategy.
    
    Uses rolling Z-score to identify extreme price deviations.
    
    Parameters:
        lookback_period: Period for calculating mean and std (default: 20)
        entry_zscore: Z-score for entry (default: 2.0)
        exit_zscore: Z-score for exit (default: 0.5)
        use_mean_reversion: True for mean reversion, False for momentum (default: True)
    """
    
    def __init__(self,
                 name: str = "ZScore_MeanReversion",
                 params: Optional[Dict[str, Any]] = None,
                 position_sizer=None):
        
        default_params = {
            'lookback_period': 20,
            'entry_zscore': 2.0,
            'exit_zscore': 0.5,
            'use_mean_reversion': True,
            'min_bars': 30,
            'stop_loss_atr': 2.0,
            'take_profit_atr': 2.0
        }
        
        if params:
            default_params.update(params)
        
        super().__init__(name, default_params, position_sizer)
    
    def _precompute_indicators(self, data: pd.DataFrame) -> None:
        """Precompute Z-score."""
        self.zscore_values = rolling_zscore(data['close'], self.params['lookback_period']).values
        self.atr_values = atr(data, 14).values
        logger.info(f"Precomputed Z-score for {self.name}")
    
    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """Generate signals based on Z-score."""
        if len(data) < self.params['min_bars']:
            return self._no_signal(data)
        
        current_price = data['close'].iloc[-1]
        timestamp = data.index[-1]
        symbol = data.get('symbol', ['UNKNOWN'])[-1] if 'symbol' in data.columns else 'UNKNOWN'
        
        # Calculate Z-score
        zscore_data = rolling_zscore(data['close'], self.params['lookback_period'])
        current_zscore = zscore_data.values.iloc[-1]
        
        atr_val = atr(data, 14).values.iloc[-1]
        
        entry_threshold = self.params['entry_zscore']
        exit_threshold = self.params['exit_zscore']
        mean_reversion = self.params['use_mean_reversion']
        
        if mean_reversion:
            # Mean reversion: buy when Z-score is very negative, sell when very positive
            if current_zscore < -entry_threshold:
                confidence = min(0.5 + abs(current_zscore) / 4, 0.9)
                return Signal(
                    timestamp=timestamp,
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    price=current_price,
                    confidence=confidence,
                    strategy_name=self.name,
                    metadata={
                        'zscore': current_zscore,
                        'entry_threshold': -entry_threshold,
                        'atr': atr_val,
                        'stop_loss': current_price - atr_val * self.params['stop_loss_atr'],
                        'take_profit': current_price + atr_val * self.params['take_profit_atr']
                    }
                )
            
            elif current_zscore > entry_threshold:
                confidence = min(0.5 + abs(current_zscore) / 4, 0.9)
                return Signal(
                    timestamp=timestamp,
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    price=current_price,
                    confidence=confidence,
                    strategy_name=self.name,
                    metadata={
                        'zscore': current_zscore,
                        'entry_threshold': entry_threshold,
                        'atr': atr_val,
                        'stop_loss': current_price + atr_val * self.params['stop_loss_atr'],
                        'take_profit': current_price - atr_val * self.params['take_profit_atr']
                    }
                )
            
            # Exit when Z-score returns to near zero
            if self.current_position is not None and abs(current_zscore) < exit_threshold:
                return Signal(
                    timestamp=timestamp,
                    symbol=symbol,
                    signal_type=SignalType.CLOSE,
                    price=current_price,
                    confidence=0.7,
                    strategy_name=self.name,
                    metadata={'zscore': current_zscore, 'exit_reason': 'zscore_mean'}
                )
        
        else:
            # Momentum mode: buy when Z-score crosses above threshold
            prev_zscore = zscore_data.values.iloc[-2] if len(zscore_data.values) > 1 else 0
            
            if prev_zscore < entry_threshold and current_zscore > entry_threshold:
                return Signal(
                    timestamp=timestamp,
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    price=current_price,
                    confidence=0.7,
                    strategy_name=self.name,
                    metadata={'zscore': current_zscore, 'mode': 'momentum'}
                )
            
            elif prev_zscore > -entry_threshold and current_zscore < -entry_threshold:
                return Signal(
                    timestamp=timestamp,
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    price=current_price,
                    confidence=0.7,
                    strategy_name=self.name,
                    metadata={'zscore': current_zscore, 'mode': 'momentum'}
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


class StochasticMeanReversionStrategy(BaseStrategy):
    """
    Stochastic Oscillator Mean Reversion Strategy.
    
    Uses Stochastic %K and %D for mean reversion signals.
    
    Parameters:
        k_period: %K period (default: 14)
        d_period: %D period (default: 3)
        smooth_k: %K smoothing (default: 3)
        oversold: Oversold level (default: 20)
        overbought: Overbought level (default: 80)
    """
    
    def __init__(self,
                 name: str = "Stoch_MeanReversion",
                 params: Optional[Dict[str, Any]] = None,
                 position_sizer=None):
        
        default_params = {
            'k_period': 14,
            'd_period': 3,
            'smooth_k': 3,
            'oversold': 20,
            'overbought': 80,
            'min_bars': 30,
            'stop_loss_atr': 2.0,
            'take_profit_atr': 2.0
        }
        
        if params:
            default_params.update(params)
        
        super().__init__(name, default_params, position_sizer)
    
    def _precompute_indicators(self, data: pd.DataFrame) -> None:
        """Precompute Stochastic indicator."""
        self.stoch_data = stochastic(
            data, 
            self.params['k_period'],
            self.params['d_period'],
            self.params['smooth_k']
        )
        self.atr_values = atr(data, 14).values
        logger.info(f"Precomputed Stochastic for {self.name}")
    
    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """Generate signals based on Stochastic oscillator."""
        if len(data) < self.params['min_bars']:
            return self._no_signal(data)
        
        current_price = data['close'].iloc[-1]
        timestamp = data.index[-1]
        symbol = data.get('symbol', ['UNKNOWN'])[-1] if 'symbol' in data.columns else 'UNKNOWN'
        
        # Calculate Stochastic
        stoch_data = stochastic(
            data,
            self.params['k_period'],
            self.params['d_period'],
            self.params['smooth_k']
        )
        
        k_values = stoch_data['k'].values
        d_values = stoch_data['d'].values
        
        k_curr = k_values.iloc[-1]
        k_prev = k_values.iloc[-2] if len(k_values) > 1 else 50
        d_curr = d_values.iloc[-1]
        
        atr_val = atr(data, 14).values.iloc[-1]
        
        oversold = self.params['oversold']
        overbought = self.params['overbought']
        
        # Buy when %K crosses above %D in oversold zone
        bullish_cross = (k_prev <= d_curr) and (k_curr > d_curr) and (k_prev < oversold)
        
        # Sell when %K crosses below %D in overbought zone
        bearish_cross = (k_prev >= d_curr) and (k_curr < d_curr) and (k_prev > overbought)
        
        if bullish_cross:
            confidence = min(0.5 + (oversold - k_curr) / 20, 0.9)
            return Signal(
                timestamp=timestamp,
                symbol=symbol,
                signal_type=SignalType.BUY,
                price=current_price,
                confidence=confidence,
                strategy_name=self.name,
                metadata={
                    'stoch_k': k_curr,
                    'stoch_d': d_curr,
                    'oversold': oversold,
                    'atr': atr_val,
                    'stop_loss': current_price - atr_val * self.params['stop_loss_atr'],
                    'take_profit': current_price + atr_val * self.params['take_profit_atr']
                }
            )
        
        elif bearish_cross:
            confidence = min(0.5 + (k_curr - overbought) / 20, 0.9)
            return Signal(
                timestamp=timestamp,
                symbol=symbol,
                signal_type=SignalType.SELL,
                price=current_price,
                confidence=confidence,
                strategy_name=self.name,
                metadata={
                    'stoch_k': k_curr,
                    'stoch_d': d_curr,
                    'overbought': overbought,
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
    'RSIMeanReversionStrategy',
    'BollingerBandMeanReversionStrategy',
    'StatisticalArbitrageStrategy',
    'ZScoreMeanReversionStrategy',
    'StochasticMeanReversionStrategy'
]
