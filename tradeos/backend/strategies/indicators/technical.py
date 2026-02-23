"""
TradeOS Technical Indicators Module
===================================
Comprehensive technical analysis indicators library.
Implements common indicators using pandas and numpy.

Author: TradeOS Team
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Dict, Union
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TrendDirection(Enum):
    """Trend direction enumeration."""
    UP = 1
    DOWN = -1
    NEUTRAL = 0


@dataclass
class IndicatorResult:
    """Container for indicator results with metadata."""
    values: pd.Series
    name: str
    params: Dict[str, any]
    timestamp: pd.DatetimeIndex
    
    def __post_init__(self):
        if len(self.values) != len(self.timestamp):
            raise ValueError("Values and timestamp must have same length")
    
    def latest(self) -> float:
        """Get latest indicator value."""
        return self.values.iloc[-1] if len(self.values) > 0 else np.nan
    
    def crossover(self, other: 'IndicatorResult') -> pd.Series:
        """Detect crossovers with another indicator."""
        return (self.values > other.values) & (self.values.shift(1) <= other.values.shift(1))
    
    def crossunder(self, other: 'IndicatorResult') -> pd.Series:
        """Detect crossunders with another indicator."""
        return (self.values < other.values) & (self.values.shift(1) >= other.values.shift(1))


# ============================================================================
# MOVING AVERAGES
# ============================================================================

def sma(data: pd.Series, period: int = 20) -> IndicatorResult:
    """
    Simple Moving Average (SMA).
    
    Args:
        data: Price series
        period: Lookback period
        
    Returns:
        IndicatorResult with SMA values
    """
    values = data.rolling(window=period, min_periods=period).mean()
    return IndicatorResult(
        values=values,
        name=f"SMA_{period}",
        params={'period': period},
        timestamp=data.index
    )


def ema(data: pd.Series, period: int = 20) -> IndicatorResult:
    """
    Exponential Moving Average (EMA).
    
    Args:
        data: Price series
        period: Lookback period
        
    Returns:
        IndicatorResult with EMA values
    """
    values = data.ewm(span=period, adjust=False, min_periods=period).mean()
    return IndicatorResult(
        values=values,
        name=f"EMA_{period}",
        params={'period': period},
        timestamp=data.index
    )


def wma(data: pd.Series, period: int = 20) -> IndicatorResult:
    """
    Weighted Moving Average (WMA).
    
    Args:
        data: Price series
        period: Lookback period
        
    Returns:
        IndicatorResult with WMA values
    """
    weights = np.arange(1, period + 1)
    values = data.rolling(window=period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )
    return IndicatorResult(
        values=values,
        name=f"WMA_{period}",
        params={'period': period},
        timestamp=data.index
    )


def hull_ma(data: pd.Series, period: int = 16) -> IndicatorResult:
    """
    Hull Moving Average - reduces lag while maintaining smoothness.
    
    Args:
        data: Price series
        period: Lookback period
        
    Returns:
        IndicatorResult with HMA values
    """
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = wma(data, half_period).values
    wma_full = wma(data, period).values
    
    raw_hma = 2 * wma_half - wma_full
    values = wma(raw_hma.dropna(), sqrt_period).values
    
    # Align indices
    values = values.reindex(data.index)
    
    return IndicatorResult(
        values=values,
        name=f"HMA_{period}",
        params={'period': period},
        timestamp=data.index
    )


def vwma(data: pd.DataFrame, period: int = 20) -> IndicatorResult:
    """
    Volume Weighted Moving Average (VWMA).
    
    Args:
        data: DataFrame with 'close' and 'volume' columns
        period: Lookback period
        
    Returns:
        IndicatorResult with VWMA values
    """
    typical_price = data['close'] * data['volume']
    values = typical_price.rolling(window=period).sum() / data['volume'].rolling(window=period).sum()
    
    return IndicatorResult(
        values=values,
        name=f"VWMA_{period}",
        params={'period': period},
        timestamp=data.index
    )


def vwap(data: pd.DataFrame, anchor: str = 'D') -> IndicatorResult:
    """
    Volume Weighted Average Price (VWAP).
    
    Args:
        data: DataFrame with 'high', 'low', 'close', 'volume' columns
        anchor: Time anchor ('D' for daily, 'W' for weekly, etc.)
        
    Returns:
        IndicatorResult with VWAP values
    """
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    
    # Group by anchor period
    groups = data.index.to_period(anchor)
    
    cum_tp_vol = (typical_price * data['volume']).groupby(groups).cumsum()
    cum_vol = data['volume'].groupby(groups).cumsum()
    
    values = cum_tp_vol / cum_vol
    
    return IndicatorResult(
        values=values,
        name=f"VWAP_{anchor}",
        params={'anchor': anchor},
        timestamp=data.index
    )


# ============================================================================
# MOMENTUM OSCILLATORS
# ============================================================================

def rsi(data: pd.Series, period: int = 14) -> IndicatorResult:
    """
    Relative Strength Index (RSI).
    
    Args:
        data: Price series
        period: Lookback period
        
    Returns:
        IndicatorResult with RSI values (0-100)
    """
    delta = data.diff()
    
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    values = 100 - (100 / (1 + rs))
    
    return IndicatorResult(
        values=values,
        name=f"RSI_{period}",
        params={'period': period},
        timestamp=data.index
    )


def macd(data: pd.Series, 
         fast: int = 12, 
         slow: int = 26, 
         signal: int = 9) -> Dict[str, IndicatorResult]:
    """
    Moving Average Convergence Divergence (MACD).
    
    Args:
        data: Price series
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line period
        
    Returns:
        Dictionary with 'macd', 'signal', and 'histogram' IndicatorResults
    """
    ema_fast = ema(data, fast).values
    ema_slow = ema(data, slow).values
    
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line.dropna(), signal).values
    
    # Align signal line
    signal_line = signal_line.reindex(macd_line.index)
    histogram = macd_line - signal_line
    
    return {
        'macd': IndicatorResult(
            values=macd_line,
            name=f"MACD_{fast}_{slow}",
            params={'fast': fast, 'slow': slow, 'signal': signal},
            timestamp=data.index
        ),
        'signal': IndicatorResult(
            values=signal_line,
            name=f"MACD_Signal_{signal}",
            params={'fast': fast, 'slow': slow, 'signal': signal},
            timestamp=data.index
        ),
        'histogram': IndicatorResult(
            values=histogram,
            name=f"MACD_Histogram",
            params={'fast': fast, 'slow': slow, 'signal': signal},
            timestamp=data.index
        )
    }


def stochastic(data: pd.DataFrame,
               k_period: int = 14,
               d_period: int = 3,
               smooth_k: int = 3) -> Dict[str, IndicatorResult]:
    """
    Stochastic Oscillator.
    
    Args:
        data: DataFrame with 'high', 'low', 'close' columns
        k_period: %K period
        d_period: %D period
        smooth_k: %K smoothing period
        
    Returns:
        Dictionary with 'k' and 'd' IndicatorResults
    """
    lowest_low = data['low'].rolling(window=k_period).min()
    highest_high = data['high'].rolling(window=k_period).max()
    
    k_raw = 100 * (data['close'] - lowest_low) / (highest_high - lowest_low)
    k = k_raw.rolling(window=smooth_k).mean()
    d = k.rolling(window=d_period).mean()
    
    return {
        'k': IndicatorResult(
            values=k,
            name=f"Stoch_K_{k_period}",
            params={'k_period': k_period, 'd_period': d_period, 'smooth_k': smooth_k},
            timestamp=data.index
        ),
        'd': IndicatorResult(
            values=d,
            name=f"Stoch_D_{d_period}",
            params={'k_period': k_period, 'd_period': d_period, 'smooth_k': smooth_k},
            timestamp=data.index
        )
    }


def williams_r(data: pd.DataFrame, period: int = 14) -> IndicatorResult:
    """
    Williams %R - momentum indicator.
    
    Args:
        data: DataFrame with 'high', 'low', 'close' columns
        period: Lookback period
        
    Returns:
        IndicatorResult with Williams %R values
    """
    highest_high = data['high'].rolling(window=period).max()
    lowest_low = data['low'].rolling(window=period).min()
    
    values = -100 * (highest_high - data['close']) / (highest_high - lowest_low)
    
    return IndicatorResult(
        values=values,
        name=f"WilliamsR_{period}",
        params={'period': period},
        timestamp=data.index
    )


def cci(data: pd.DataFrame, period: int = 20) -> IndicatorResult:
    """
    Commodity Channel Index (CCI).
    
    Args:
        data: DataFrame with 'high', 'low', 'close' columns
        period: Lookback period
        
    Returns:
        IndicatorResult with CCI values
    """
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    sma_tp = typical_price.rolling(window=period).mean()
    mean_deviation = typical_price.rolling(window=period).apply(
        lambda x: np.abs(x - x.mean()).mean()
    )
    
    values = (typical_price - sma_tp) / (0.015 * mean_deviation)
    
    return IndicatorResult(
        values=values,
        name=f"CCI_{period}",
        params={'period': period},
        timestamp=data.index
    )


def momentum(data: pd.Series, period: int = 10) -> IndicatorResult:
    """
    Momentum indicator - rate of change.
    
    Args:
        data: Price series
        period: Lookback period
        
    Returns:
        IndicatorResult with momentum values
    """
    values = data / data.shift(period) * 100
    
    return IndicatorResult(
        values=values,
        name=f"Momentum_{period}",
        params={'period': period},
        timestamp=data.index
    )


# ============================================================================
# VOLATILITY INDICATORS
# ============================================================================

def atr(data: pd.DataFrame, period: int = 14) -> IndicatorResult:
    """
    Average True Range (ATR).
    
    Args:
        data: DataFrame with 'high', 'low', 'close' columns
        period: Lookback period
        
    Returns:
        IndicatorResult with ATR values
    """
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift())
    low_close = np.abs(data['low'] - data['close'].shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    values = true_range.ewm(alpha=1/period, min_periods=period).mean()
    
    return IndicatorResult(
        values=values,
        name=f"ATR_{period}",
        params={'period': period},
        timestamp=data.index
    )


def bollinger_bands(data: pd.Series,
                    period: int = 20,
                    std_dev: float = 2.0) -> Dict[str, IndicatorResult]:
    """
    Bollinger Bands.
    
    Args:
        data: Price series
        period: SMA period
        std_dev: Standard deviation multiplier
        
    Returns:
        Dictionary with 'upper', 'middle', 'lower', and 'bandwidth' IndicatorResults
    """
    middle = sma(data, period).values
    std = data.rolling(window=period).std()
    
    upper = middle + std * std_dev
    lower = middle - std * std_dev
    bandwidth = (upper - lower) / middle * 100
    percent_b = (data - lower) / (upper - lower)
    
    return {
        'upper': IndicatorResult(
            values=upper,
            name=f"BB_Upper_{period}_{std_dev}",
            params={'period': period, 'std_dev': std_dev},
            timestamp=data.index
        ),
        'middle': IndicatorResult(
            values=middle,
            name=f"BB_Middle_{period}",
            params={'period': period, 'std_dev': std_dev},
            timestamp=data.index
        ),
        'lower': IndicatorResult(
            values=lower,
            name=f"BB_Lower_{period}_{std_dev}",
            params={'period': period, 'std_dev': std_dev},
            timestamp=data.index
        ),
        'bandwidth': IndicatorResult(
            values=bandwidth,
            name=f"BB_Bandwidth_{period}",
            params={'period': period, 'std_dev': std_dev},
            timestamp=data.index
        ),
        'percent_b': IndicatorResult(
            values=percent_b,
            name=f"BB_PercentB_{period}",
            params={'period': period, 'std_dev': std_dev},
            timestamp=data.index
        )
    }


def keltner_channels(data: pd.DataFrame,
                     ema_period: int = 20,
                     atr_period: int = 10,
                    atr_mult: float = 2.0) -> Dict[str, IndicatorResult]:
    """
    Keltner Channels.
    
    Args:
        data: DataFrame with 'high', 'low', 'close' columns
        ema_period: EMA period for middle line
        atr_period: ATR period
        atr_mult: ATR multiplier
        
    Returns:
        Dictionary with 'upper', 'middle', 'lower' IndicatorResults
    """
    middle = ema(data['close'], ema_period).values
    atr_val = atr(data, atr_period).values
    
    upper = middle + atr_mult * atr_val
    lower = middle - atr_mult * atr_val
    
    return {
        'upper': IndicatorResult(
            values=upper,
            name=f"KC_Upper_{ema_period}_{atr_mult}",
            params={'ema_period': ema_period, 'atr_period': atr_period, 'atr_mult': atr_mult},
            timestamp=data.index
        ),
        'middle': IndicatorResult(
            values=middle,
            name=f"KC_Middle_{ema_period}",
            params={'ema_period': ema_period, 'atr_period': atr_period, 'atr_mult': atr_mult},
            timestamp=data.index
        ),
        'lower': IndicatorResult(
            values=lower,
            name=f"KC_Lower_{ema_period}_{atr_mult}",
            params={'ema_period': ema_period, 'atr_period': atr_period, 'atr_mult': atr_mult},
            timestamp=data.index
        )
    }


def donchian_channels(data: pd.DataFrame, period: int = 20) -> Dict[str, IndicatorResult]:
    """
    Donchian Channels.
    
    Args:
        data: DataFrame with 'high', 'low' columns
        period: Lookback period
        
    Returns:
        Dictionary with 'upper', 'middle', 'lower' IndicatorResults
    """
    upper = data['high'].rolling(window=period).max()
    lower = data['low'].rolling(window=period).min()
    middle = (upper + lower) / 2
    
    return {
        'upper': IndicatorResult(
            values=upper,
            name=f"DC_Upper_{period}",
            params={'period': period},
            timestamp=data.index
        ),
        'middle': IndicatorResult(
            values=middle,
            name=f"DC_Middle_{period}",
            params={'period': period},
            timestamp=data.index
        ),
        'lower': IndicatorResult(
            values=lower,
            name=f"DC_Lower_{period}",
            params={'period': period},
            timestamp=data.index
        )
    }


def historical_volatility(data: pd.Series, period: int = 20) -> IndicatorResult:
    """
    Historical volatility (annualized).
    
    Args:
        data: Price series
        period: Lookback period
        
    Returns:
        IndicatorResult with volatility values
    """
    log_returns = np.log(data / data.shift(1))
    values = log_returns.rolling(window=period).std() * np.sqrt(252) * 100
    
    return IndicatorResult(
        values=values,
        name=f"HV_{period}",
        params={'period': period},
        timestamp=data.index
    )


# ============================================================================
# TREND INDICATORS
# ============================================================================

def adx(data: pd.DataFrame, period: int = 14) -> Dict[str, IndicatorResult]:
    """
    Average Directional Index (ADX).
    
    Args:
        data: DataFrame with 'high', 'low', 'close' columns
        period: Lookback period
        
    Returns:
        Dictionary with 'adx', 'plus_di', 'minus_di' IndicatorResults
    """
    # True Range
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift())
    low_close = np.abs(data['low'] - data['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # Directional Movement
    plus_dm = data['high'].diff()
    minus_dm = -data['low'].diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # Smoothed values
    atr_val = tr.ewm(alpha=1/period, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr_val
    minus_di = 100 * minus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr_val
    
    # ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_val = dx.ewm(alpha=1/period, min_periods=period).mean()
    
    return {
        'adx': IndicatorResult(
            values=adx_val,
            name=f"ADX_{period}",
            params={'period': period},
            timestamp=data.index
        ),
        'plus_di': IndicatorResult(
            values=plus_di,
            name=f"PlusDI_{period}",
            params={'period': period},
            timestamp=data.index
        ),
        'minus_di': IndicatorResult(
            values=minus_di,
            name=f"MinusDI_{period}",
            params={'period': period},
            timestamp=data.index
        )
    }


def supertrend(data: pd.DataFrame, 
               atr_period: int = 10, 
               factor: float = 3.0) -> Dict[str, IndicatorResult]:
    """
    SuperTrend indicator.
    
    Args:
        data: DataFrame with 'high', 'low', 'close' columns
        atr_period: ATR period
        factor: ATR multiplier
        
    Returns:
        Dictionary with 'supertrend', 'direction', 'upper', 'lower' IndicatorResults
    """
    atr_val = atr(data, atr_period).values
    
    hl2 = (data['high'] + data['low']) / 2
    upper_band = hl2 + factor * atr_val
    lower_band = hl2 - factor * atr_val
    
    # Initialize
    supertrend_vals = pd.Series(index=data.index, dtype=float)
    direction = pd.Series(index=data.index, dtype=int)
    
    for i in range(len(data)):
        if i == 0:
            supertrend_vals.iloc[i] = upper_band.iloc[i]
            direction.iloc[i] = 1
        else:
            close = data['close'].iloc[i]
            prev_upper = upper_band.iloc[i-1]
            prev_lower = lower_band.iloc[i-1]
            prev_supertrend = supertrend_vals.iloc[i-1]
            
            # Adjust bands
            if upper_band.iloc[i] < prev_upper or data['close'].iloc[i-1] > prev_upper:
                upper_band.iloc[i] = prev_upper
            if lower_band.iloc[i] > prev_lower or data['close'].iloc[i-1] < prev_lower:
                lower_band.iloc[i] = prev_lower
            
            # Determine trend
            if prev_supertrend == prev_upper:
                if close > upper_band.iloc[i]:
                    supertrend_vals.iloc[i] = lower_band.iloc[i]
                    direction.iloc[i] = 1  # Up
                else:
                    supertrend_vals.iloc[i] = upper_band.iloc[i]
                    direction.iloc[i] = -1  # Down
            else:
                if close < lower_band.iloc[i]:
                    supertrend_vals.iloc[i] = upper_band.iloc[i]
                    direction.iloc[i] = -1  # Down
                else:
                    supertrend_vals.iloc[i] = lower_band.iloc[i]
                    direction.iloc[i] = 1  # Up
    
    return {
        'supertrend': IndicatorResult(
            values=supertrend_vals,
            name=f"SuperTrend_{atr_period}_{factor}",
            params={'atr_period': atr_period, 'factor': factor},
            timestamp=data.index
        ),
        'direction': IndicatorResult(
            values=direction,
            name=f"SuperTrend_Dir_{atr_period}",
            params={'atr_period': atr_period, 'factor': factor},
            timestamp=data.index
        ),
        'upper': IndicatorResult(
            values=upper_band,
            name=f"SuperTrend_Upper_{atr_period}",
            params={'atr_period': atr_period, 'factor': factor},
            timestamp=data.index
        ),
        'lower': IndicatorResult(
            values=lower_band,
            name=f"SuperTrend_Lower_{atr_period}",
            params={'atr_period': atr_period, 'factor': factor},
            timestamp=data.index
        )
    }


def parabolic_sar(data: pd.DataFrame, 
                  af_start: float = 0.02, 
                  af_max: float = 0.2,
                  af_step: float = 0.02) -> IndicatorResult:
    """
    Parabolic Stop and Reverse (SAR).
    
    Args:
        data: DataFrame with 'high', 'low' columns
        af_start: Starting acceleration factor
        af_max: Maximum acceleration factor
        af_step: Acceleration factor step
        
    Returns:
        IndicatorResult with SAR values
    """
    sar = pd.Series(index=data.index, dtype=float)
    trend = pd.Series(index=data.index, dtype=int)
    
    af = af_start
    ep = data['high'].iloc[0]  # Extreme point
    sar.iloc[0] = data['low'].iloc[0]
    trend.iloc[0] = 1  # Start with uptrend
    
    for i in range(1, len(data)):
        prev_sar = sar.iloc[i-1]
        
        if trend.iloc[i-1] == 1:  # Uptrend
            sar.iloc[i] = prev_sar + af * (ep - prev_sar)
            
            # Ensure SAR is below recent lows
            sar.iloc[i] = min(sar.iloc[i], data['low'].iloc[i-1], data['low'].iloc[max(0, i-2)])
            
            if data['low'].iloc[i] < sar.iloc[i]:  # Trend reversal
                trend.iloc[i] = -1
                sar.iloc[i] = ep
                ep = data['low'].iloc[i]
                af = af_start
            else:
                trend.iloc[i] = 1
                if data['high'].iloc[i] > ep:
                    ep = data['high'].iloc[i]
                    af = min(af + af_step, af_max)
        else:  # Downtrend
            sar.iloc[i] = prev_sar - af * (prev_sar - ep)
            
            # Ensure SAR is above recent highs
            sar.iloc[i] = max(sar.iloc[i], data['high'].iloc[i-1], data['high'].iloc[max(0, i-2)])
            
            if data['high'].iloc[i] > sar.iloc[i]:  # Trend reversal
                trend.iloc[i] = 1
                sar.iloc[i] = ep
                ep = data['high'].iloc[i]
                af = af_start
            else:
                trend.iloc[i] = -1
                if data['low'].iloc[i] < ep:
                    ep = data['low'].iloc[i]
                    af = min(af + af_step, af_max)
    
    return IndicatorResult(
        values=sar,
        name=f"PSAR_{af_start}_{af_max}",
        params={'af_start': af_start, 'af_max': af_max, 'af_step': af_step},
        timestamp=data.index
    )


# ============================================================================
# VOLUME INDICATORS
# ============================================================================

def obv(data: pd.DataFrame) -> IndicatorResult:
    """
    On-Balance Volume (OBV).
    
    Args:
        data: DataFrame with 'close', 'volume' columns
        
    Returns:
        IndicatorResult with OBV values
    """
    obv_vals = pd.Series(index=data.index, dtype=float)
    obv_vals.iloc[0] = data['volume'].iloc[0]
    
    for i in range(1, len(data)):
        if data['close'].iloc[i] > data['close'].iloc[i-1]:
            obv_vals.iloc[i] = obv_vals.iloc[i-1] + data['volume'].iloc[i]
        elif data['close'].iloc[i] < data['close'].iloc[i-1]:
            obv_vals.iloc[i] = obv_vals.iloc[i-1] - data['volume'].iloc[i]
        else:
            obv_vals.iloc[i] = obv_vals.iloc[i-1]
    
    return IndicatorResult(
        values=obv_vals,
        name="OBV",
        params={},
        timestamp=data.index
    )


def volume_profile(data: pd.DataFrame, 
                   bins: int = 50,
                   lookback: Optional[int] = None) -> Dict[str, any]:
    """
    Volume Profile - distribution of volume by price level.
    
    Args:
        data: DataFrame with 'high', 'low', 'close', 'volume' columns
        bins: Number of price bins
        lookback: Number of bars to analyze (None for all)
        
    Returns:
        Dictionary with volume profile data
    """
    if lookback:
        data = data.tail(lookback)
    
    # Use typical price
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    
    # Create price bins
    price_min = typical_price.min()
    price_max = typical_price.max()
    bin_edges = np.linspace(price_min, price_max, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Assign volume to bins
    volume_by_bin = np.zeros(bins)
    for i in range(bins):
        mask = (typical_price >= bin_edges[i]) & (typical_price < bin_edges[i+1])
        volume_by_bin[i] = data.loc[mask, 'volume'].sum()
    
    # Find POC (Point of Control) - highest volume bin
    poc_idx = np.argmax(volume_by_bin)
    poc = bin_centers[poc_idx]
    
    # Calculate Value Area (70% of volume)
    total_volume = volume_by_bin.sum()
    target_volume = total_volume * 0.7
    
    sorted_indices = np.argsort(volume_by_bin)[::-1]
    cumulative_volume = 0
    value_area_indices = []
    
    for idx in sorted_indices:
        cumulative_volume += volume_by_bin[idx]
        value_area_indices.append(idx)
        if cumulative_volume >= target_volume:
            break
    
    value_area_low = bin_centers[min(value_area_indices)]
    value_area_high = bin_centers[max(value_area_indices)]
    
    return {
        'price_levels': bin_centers,
        'volume': volume_by_bin,
        'poc': poc,
        'value_area_low': value_area_low,
        'value_area_high': value_area_high,
        'value_area_volume': cumulative_volume
    }


def vwap_with_std(data: pd.DataFrame, 
                  period: int = 20,
                  std_mult: float = 1.0) -> Dict[str, IndicatorResult]:
    """
    VWAP with standard deviation bands.
    
    Args:
        data: DataFrame with 'high', 'low', 'close', 'volume' columns
        period: VWAP period
        std_mult: Standard deviation multiplier for bands
        
    Returns:
        Dictionary with 'vwap', 'upper', 'lower' IndicatorResults
    """
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    
    cum_tp_vol = (typical_price * data['volume']).rolling(window=period).sum()
    cum_vol = data['volume'].rolling(window=period).sum()
    vwap_val = cum_tp_vol / cum_vol
    
    # Calculate standard deviation
    variance = ((typical_price - vwap_val) ** 2 * data['volume']).rolling(window=period).sum() / cum_vol
    std = np.sqrt(variance)
    
    upper = vwap_val + std * std_mult
    lower = vwap_val - std * std_mult
    
    return {
        'vwap': IndicatorResult(
            values=vwap_val,
            name=f"VWAP_{period}",
            params={'period': period, 'std_mult': std_mult},
            timestamp=data.index
        ),
        'upper': IndicatorResult(
            values=upper,
            name=f"VWAP_Upper_{period}",
            params={'period': period, 'std_mult': std_mult},
            timestamp=data.index
        ),
        'lower': IndicatorResult(
            values=lower,
            name=f"VWAP_Lower_{period}",
            params={'period': period, 'std_mult': std_mult},
            timestamp=data.index
        )
    }


# ============================================================================
# FIBONACCI TOOLS
# ============================================================================

def fibonacci_retracement(high: float, 
                          low: float,
                          levels: List[float] = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]) -> Dict[str, float]:
    """
    Calculate Fibonacci retracement levels.
    
    Args:
        high: Swing high price
        low: Swing low price
        levels: Fibonacci levels to calculate
        
    Returns:
        Dictionary of level names to prices
    """
    diff = high - low
    return {f"fib_{int(level*1000) if level != 0.5 else '50'}": high - diff * level 
            for level in levels}


def fibonacci_extension(high: float,
                        low: float,
                        retracement: float,
                        levels: List[float] = [0.618, 1, 1.272, 1.618, 2, 2.618]) -> Dict[str, float]:
    """
    Calculate Fibonacci extension levels.
    
    Args:
        high: Swing high price
        low: Swing low price
        retracement: Retracement level price
        levels: Extension levels to calculate
        
    Returns:
        Dictionary of level names to prices
    """
    diff = high - low
    direction = 1 if retracement > low else -1
    
    return {f"ext_{int(level*1000)}": retracement + diff * level * direction 
            for level in levels}


def find_swing_points(data: pd.Series, 
                      window: int = 5) -> Tuple[pd.Series, pd.Series]:
    """
    Find swing highs and lows.
    
    Args:
        data: Price series
        window: Window for detecting swings
        
    Returns:
        Tuple of (swing_highs, swing_lows) series
    """
    # Swing highs
    swing_highs = (data == data.rolling(window=window*2+1, center=True).max()) & \
                  (data.diff(window) > 0) & (data.diff(-window) < 0)
    
    # Swing lows
    swing_lows = (data == data.rolling(window=window*2+1, center=True).min()) & \
                 (data.diff(window) < 0) & (data.diff(-window) > 0)
    
    return swing_highs, swing_lows


# ============================================================================
# PATTERN DETECTION
# ============================================================================

def detect_engulfing(data: pd.DataFrame) -> pd.Series:
    """
    Detect bullish and bearish engulfing patterns.
    
    Args:
        data: DataFrame with 'open', 'high', 'low', 'close' columns
        
    Returns:
        Series with pattern signals (1=bullish, -1=bearish, 0=none)
    """
    prev_open = data['open'].shift(1)
    prev_close = data['close'].shift(1)
    
    # Bullish engulfing
    bullish = (data['close'] > data['open']) & \
              (prev_close < prev_open) & \
              (data['open'] <= prev_close) & \
              (data['close'] >= prev_open)
    
    # Bearish engulfing
    bearish = (data['close'] < data['open']) & \
              (prev_close > prev_open) & \
              (data['open'] >= prev_close) & \
              (data['close'] <= prev_open)
    
    return bullish.astype(int) - bearish.astype(int)


def detect_doji(data: pd.DataFrame, threshold: float = 0.1) -> pd.Series:
    """
    Detect doji candlestick patterns.
    
    Args:
        data: DataFrame with 'open', 'high', 'low', 'close' columns
        threshold: Body/range ratio threshold
        
    Returns:
        Boolean series indicating doji patterns
    """
    body = abs(data['close'] - data['open'])
    range_val = data['high'] - data['low']
    
    return (body / range_val) < threshold


def detect_hammer(data: pd.DataFrame) -> pd.Series:
    """
    Detect hammer and inverted hammer patterns.
    
    Args:
        data: DataFrame with 'open', 'high', 'low', 'close' columns
        
    Returns:
        Series with pattern signals (1=hammer, -1=inverted hammer, 0=none)
    """
    body = abs(data['close'] - data['open'])
    upper_shadow = data['high'] - data[['open', 'close']].max(axis=1)
    lower_shadow = data[['open', 'close']].min(axis=1) - data['low']
    range_val = data['high'] - data['low']
    
    # Hammer: small body, long lower shadow, little/no upper shadow
    hammer = (body < range_val * 0.3) & \
             (lower_shadow > body * 2) & \
             (upper_shadow < body * 0.5)
    
    # Inverted hammer: small body, long upper shadow, little/no lower shadow
    inv_hammer = (body < range_val * 0.3) & \
                 (upper_shadow > body * 2) & \
                 (lower_shadow < body * 0.5)
    
    return hammer.astype(int) - inv_hammer.astype(int)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def crossover(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """Detect when series1 crosses above series2."""
    return (series1 > series2) & (series1.shift(1) <= series2.shift(1))


def crossunder(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """Detect when series1 crosses below series2."""
    return (series1 < series2) & (series1.shift(1) >= series2.shift(1))


def rolling_zscore(data: pd.Series, period: int = 20) -> IndicatorResult:
    """
    Calculate rolling Z-score (standardized deviation from mean).
    
    Args:
        data: Input series
        period: Lookback period
        
    Returns:
        IndicatorResult with Z-score values
    """
    rolling_mean = data.rolling(window=period).mean()
    rolling_std = data.rolling(window=period).std()
    values = (data - rolling_mean) / rolling_std
    
    return IndicatorResult(
        values=values,
        name=f"ZScore_{period}",
        params={'period': period},
        timestamp=data.index
    )


def calculate_slope(data: pd.Series, period: int = 20) -> IndicatorResult:
    """
    Calculate linear regression slope.
    
    Args:
        data: Input series
        period: Lookback period
        
    Returns:
        IndicatorResult with slope values
    """
    def linear_slope(x):
        x_vals = np.arange(len(x))
        return np.polyfit(x_vals, x, 1)[0]
    
    values = data.rolling(window=period).apply(linear_slope, raw=True)
    
    return IndicatorResult(
        values=values,
        name=f"Slope_{period}",
        params={'period': period},
        timestamp=data.index
    )


def calculate_r2(data: pd.Series, period: int = 20) -> IndicatorResult:
    """
    Calculate R-squared (coefficient of determination) for trend strength.
    
    Args:
        data: Input series
        period: Lookback period
        
    Returns:
        IndicatorResult with R-squared values
    """
    def r_squared(x):
        x_vals = np.arange(len(x))
        slope, intercept = np.polyfit(x_vals, x, 1)
        y_pred = slope * x_vals + intercept
        ss_res = np.sum((x - y_pred) ** 2)
        ss_tot = np.sum((x - np.mean(x)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    values = data.rolling(window=period).apply(r_squared, raw=True)
    
    return IndicatorResult(
        values=values,
        name=f"R2_{period}",
        params={'period': period},
        timestamp=data.index
    )


# Export all functions
__all__ = [
    # Moving Averages
    'sma', 'ema', 'wma', 'hull_ma', 'vwma', 'vwap',
    # Momentum
    'rsi', 'macd', 'stochastic', 'williams_r', 'cci', 'momentum',
    # Volatility
    'atr', 'bollinger_bands', 'keltner_channels', 'donchian_channels', 
    'historical_volatility',
    # Trend
    'adx', 'supertrend', 'parabolic_sar',
    # Volume
    'obv', 'volume_profile', 'vwap_with_std',
    # Fibonacci
    'fibonacci_retracement', 'fibonacci_extension', 'find_swing_points',
    # Patterns
    'detect_engulfing', 'detect_doji', 'detect_hammer',
    # Utilities
    'crossover', 'crossunder', 'rolling_zscore', 'calculate_slope', 'calculate_r2',
    # Classes
    'IndicatorResult', 'TrendDirection'
]
