"""
TradeOS Market Regime Filter Module
===================================
Market regime classification and filtering system.
Identifies trending vs ranging markets and volatility states.

Author: TradeOS Team
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
import logging

from ..indicators.technical import (
    adx, ema, sma, atr, bollinger_bands, 
    calculate_slope, calculate_r2, historical_volatility
)

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classification."""
    STRONG_UPTREND = "strong_uptrend"
    UPTREND = "uptrend"
    WEAK_UPTREND = "weak_uptrend"
    RANGING = "ranging"
    WEAK_DOWNTREND = "weak_downtrend"
    DOWNTREND = "downtrend"
    STRONG_DOWNTREND = "strong_downtrend"
    UNKNOWN = "unknown"


class VolatilityRegime(Enum):
    """Volatility regime classification."""
    VERY_LOW = "very_low"
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class RegimeState:
    """Container for regime state information."""
    timestamp: datetime
    regime: MarketRegime
    volatility_regime: VolatilityRegime
    adx: float
    trend_direction: int  # 1 for up, -1 for down, 0 for neutral
    trend_strength: float  # 0-1 scale
    volatility: float
    alignment_score: float  # Multi-timeframe alignment
    
    def is_trending(self, threshold: float = 0.5) -> bool:
        """Check if market is trending."""
        return self.trend_strength >= threshold
    
    def is_ranging(self, threshold: float = 0.3) -> bool:
        """Check if market is ranging."""
        return self.trend_strength < threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'regime': self.regime.value,
            'volatility_regime': self.volatility_regime.value,
            'adx': self.adx,
            'trend_direction': self.trend_direction,
            'trend_strength': self.trend_strength,
            'volatility': self.volatility,
            'alignment_score': self.alignment_score
        }


class RegimeFilter:
    """
    Market regime detection and filtering system.
    
    Uses multiple indicators to classify market conditions:
    - ADX for trend strength
    - Price vs EMA for trend direction
    - Bollinger Bandwidth for ranging detection
    - Historical volatility for volatility regime
    """
    
    def __init__(self,
                 adx_period: int = 14,
                 adx_threshold: float = 25.0,
                 ema_period: int = 50,
                 bb_period: int = 20,
                 vol_period: int = 20,
                 alignment_lookback: int = 5):
        """
        Initialize regime filter.
        
        Args:
            adx_period: ADX calculation period
            adx_threshold: ADX threshold for trending
            ema_period: EMA period for trend direction
            bb_period: Bollinger Bands period
            vol_period: Volatility calculation period
            alignment_lookback: Bars for regime persistence
        """
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.ema_period = ema_period
        self.bb_period = bb_period
        self.vol_period = vol_period
        self.alignment_lookback = alignment_lookback
        
        self.regime_history: List[RegimeState] = []
        self.current_state: Optional[RegimeState] = None
    
    def calculate_trend_strength(self, data: pd.DataFrame) -> Tuple[float, int]:
        """
        Calculate trend strength and direction.
        
        Returns:
            Tuple of (trend_strength, trend_direction)
        """
        if len(data) < self.adx_period + 10:
            return 0.0, 0
        
        # Calculate ADX
        adx_data = adx(data, self.adx_period)
        adx_val = adx_data['adx'].values.iloc[-1]
        plus_di = adx_data['plus_di'].values.iloc[-1]
        minus_di = adx_data['minus_di'].values.iloc[-1]
        
        # Normalize ADX to 0-1 scale
        trend_strength = min(adx_val / 50.0, 1.0)
        
        # Determine direction
        if plus_di > minus_di:
            trend_direction = 1
        elif minus_di > plus_di:
            trend_direction = -1
        else:
            trend_direction = 0
        
        return trend_strength, trend_direction
    
    def calculate_price_momentum(self, data: pd.DataFrame) -> float:
        """Calculate price momentum score."""
        if len(data) < self.ema_period:
            return 0.0
        
        close = data['close']
        ema_line = ema(close, self.ema_period).values
        
        # Price vs EMA
        current_price = close.iloc[-1]
        current_ema = ema_line.iloc[-1]
        
        # Slope of EMA
        ema_slope = calculate_slope(ema_line, 10).values.iloc[-1]
        
        # Price position relative to EMA
        price_position = (current_price - current_ema) / current_ema * 100
        
        # Combine signals
        momentum = np.sign(ema_slope) * min(abs(ema_slope) * 10 + abs(price_position), 10)
        
        return momentum
    
    def detect_ranging(self, data: pd.DataFrame) -> float:
        """
        Detect ranging market conditions.
        
        Returns:
            Ranging score (0-1, higher = more ranging)
        """
        if len(data) < self.bb_period + 10:
            return 0.5
        
        # Bollinger Bandwidth
        bb_data = bollinger_bands(data['close'], self.bb_period, 2.0)
        bandwidth = bb_data['bandwidth'].values.iloc[-1]
        
        # R-squared of price (low R2 = ranging)
        r2 = calculate_r2(data['close'], self.bb_period).values.iloc[-1]
        
        # Combine metrics
        # Low bandwidth and low R2 indicate ranging
        ranging_score = (1 - min(bandwidth / 10, 1)) * 0.5 + (1 - r2) * 0.5
        
        return ranging_score
    
    def classify_volatility_regime(self, data: pd.DataFrame) -> VolatilityRegime:
        """Classify current volatility regime."""
        if len(data) < self.vol_period:
            return VolatilityRegime.NORMAL
        
        hv = historical_volatility(data['close'], self.vol_period).values.iloc[-1]
        
        # Classify based on historical volatility percentage
        if hv < 10:
            return VolatilityRegime.VERY_LOW
        elif hv < 20:
            return VolatilityRegime.LOW
        elif hv < 40:
            return VolatilityRegime.NORMAL
        elif hv < 60:
            return VolatilityRegime.HIGH
        else:
            return VolatilityRegime.VERY_HIGH
    
    def classify_regime(self, 
                        trend_strength: float,
                        trend_direction: int,
                        momentum: float,
                        ranging_score: float) -> MarketRegime:
        """
        Classify market regime based on indicators.
        
        Args:
            trend_strength: 0-1 trend strength
            trend_direction: 1 (up), -1 (down), 0 (neutral)
            momentum: Price momentum score
            ranging_score: 0-1 ranging score
            
        Returns:
            MarketRegime classification
        """
        # If ranging score is high, market is ranging
        if ranging_score > 0.6 and trend_strength < 0.4:
            return MarketRegime.RANGING
        
        # Classify based on trend strength and direction
        if trend_direction == 1:  # Uptrend
            if trend_strength > 0.7:
                return MarketRegime.STRONG_UPTREND
            elif trend_strength > 0.4:
                return MarketRegime.UPTREND
            else:
                return MarketRegime.WEAK_UPTREND
        
        elif trend_direction == -1:  # Downtrend
            if trend_strength > 0.7:
                return MarketRegime.STRONG_DOWNTREND
            elif trend_strength > 0.4:
                return MarketRegime.DOWNTREND
            else:
                return MarketRegime.WEAK_DOWNTREND
        
        return MarketRegime.RANGING
    
    def calculate_alignment_score(self, data: pd.DataFrame) -> float:
        """
        Calculate multi-timeframe alignment score.
        
        Checks if trend direction aligns across multiple timeframes.
        
        Returns:
            Alignment score (0-1)
        """
        if len(data) < 100:
            return 0.5
        
        close = data['close']
        
        # Multiple EMAs
        ema_20 = ema(close, 20).values
        ema_50 = ema(close, 50).values
        ema_100 = ema(close, 100).values
        
        current_price = close.iloc[-1]
        
        # Check alignment
        above_20 = current_price > ema_20.iloc[-1]
        above_50 = current_price > ema_50.iloc[-1]
        above_100 = current_price > ema_100.iloc[-1]
        
        ema_20_above_50 = ema_20.iloc[-1] > ema_50.iloc[-1]
        ema_50_above_100 = ema_50.iloc[-1] > ema_100.iloc[-1]
        
        # Count aligned conditions
        aligned_count = sum([above_20, above_50, above_100, ema_20_above_50, ema_50_above_100])
        
        # If mostly bearish, invert
        if aligned_count < 2:
            aligned_count = 5 - aligned_count
        
        return aligned_count / 5.0
    
    def update(self, data: pd.DataFrame) -> RegimeState:
        """
        Update regime filter with new data.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            Current regime state
        """
        timestamp = data.index[-1]
        
        # Calculate all metrics
        trend_strength, trend_direction = self.calculate_trend_strength(data)
        momentum = self.calculate_price_momentum(data)
        ranging_score = self.detect_ranging(data)
        volatility_regime = self.classify_volatility_regime(data)
        alignment_score = self.calculate_alignment_score(data)
        
        # Get ADX value
        adx_val = 0
        if len(data) >= self.adx_period + 10:
            adx_val = adx(data, self.adx_period)['adx'].values.iloc[-1]
        
        # Get volatility value
        hv = 0
        if len(data) >= self.vol_period:
            hv = historical_volatility(data['close'], self.vol_period).values.iloc[-1]
        
        # Classify regime
        regime = self.classify_regime(trend_strength, trend_direction, momentum, ranging_score)
        
        # Create state
        state = RegimeState(
            timestamp=timestamp,
            regime=regime,
            volatility_regime=volatility_regime,
            adx=adx_val,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            volatility=hv,
            alignment_score=alignment_score
        )
        
        self.current_state = state
        self.regime_history.append(state)
        
        # Keep history manageable
        if len(self.regime_history) > 1000:
            self.regime_history = self.regime_history[-500:]
        
        return state
    
    def get_current_state(self) -> Optional[RegimeState]:
        """Get current regime state."""
        return self.current_state
    
    def get_regime_persistence(self) -> int:
        """Get number of bars current regime has persisted."""
        if not self.regime_history or len(self.regime_history) < 2:
            return 0
        
        current_regime = self.current_state.regime if self.current_state else None
        if not current_regime:
            return 0
        
        persistence = 0
        for state in reversed(self.regime_history):
            if state.regime == current_regime:
                persistence += 1
            else:
                break
        
        return persistence
    
    def should_trade_trend_following(self) -> bool:
        """Check if trend following strategies should be active."""
        if not self.current_state:
            return False
        
        return self.current_state.is_trending(0.5)
    
    def should_trade_mean_reversion(self) -> bool:
        """Check if mean reversion strategies should be active."""
        if not self.current_state:
            return False
        
        return self.current_state.is_ranging(0.4)
    
    def should_trade_breakout(self) -> bool:
        """Check if breakout strategies should be active."""
        if not self.current_state:
            return False
        
        # Breakouts work well in low volatility before expansion
        return (self.current_state.volatility_regime in 
                [VolatilityRegime.LOW, VolatilityRegime.VERY_LOW])
    
    def get_recommended_strategies(self) -> List[str]:
        """Get list of recommended strategy types for current regime."""
        if not self.current_state:
            return []
        
        strategies = []
        
        if self.should_trade_trend_following():
            strategies.extend(['trend_following', 'momentum'])
        
        if self.should_trade_mean_reversion():
            strategies.extend(['mean_reversion', 'range_trading'])
        
        if self.should_trade_breakout():
            strategies.append('breakout')
        
        if self.current_state.volatility_regime in [VolatilityRegime.HIGH, VolatilityRegime.VERY_HIGH]:
            strategies.append('volatility')
        
        return strategies
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert regime history to DataFrame."""
        if not self.regime_history:
            return pd.DataFrame()
        
        return pd.DataFrame([s.to_dict() for s in self.regime_history])


class MultiTimeframeRegimeFilter:
    """
    Multi-timeframe regime filter.
    
    Combines regime information from multiple timeframes
    for more robust classification.
    """
    
    def __init__(self, 
                 timeframes: List[str] = ['1h', '4h', '1d'],
                 weights: Optional[List[float]] = None):
        """
        Initialize multi-timeframe filter.
        
        Args:
            timeframes: List of timeframe strings
            weights: Weights for each timeframe (must sum to 1)
        """
        self.timeframes = timeframes
        
        if weights is None:
            # Default: higher weight for longer timeframes
            self.weights = [0.2, 0.3, 0.5]
        else:
            self.weights = weights
        
        self.filters: Dict[str, RegimeFilter] = {
            tf: RegimeFilter() for tf in timeframes
        }
        
        self.combined_state: Optional[RegimeState] = None
    
    def update(self, data_dict: Dict[str, pd.DataFrame]) -> RegimeState:
        """
        Update all timeframe filters.
        
        Args:
            data_dict: Dictionary mapping timeframe to OHLCV DataFrame
            
        Returns:
            Combined regime state
        """
        states = []
        
        for tf, data in data_dict.items():
            if tf in self.filters:
                state = self.filters[tf].update(data)
                states.append((state, self.weights[self.timeframes.index(tf)]))
        
        # Combine states
        if not states:
            return None
        
        # Weighted average of trend strength
        combined_trend_strength = sum(s.trend_strength * w for s, w in states)
        
        # Majority vote for trend direction
        up_votes = sum(w for s, w in states if s.trend_direction == 1)
        down_votes = sum(w for s, w in states if s.trend_direction == -1)
        
        if up_votes > down_votes:
            combined_direction = 1
        elif down_votes > up_votes:
            combined_direction = -1
        else:
            combined_direction = 0
        
        # Average ADX
        combined_adx = sum(s.adx * w for s, w in states)
        
        # Average volatility
        combined_vol = sum(s.volatility * w for s, w in states)
        
        # Average alignment
        combined_alignment = sum(s.alignment_score * w for s, w in states)
        
        # Classify combined regime
        filter_instance = RegimeFilter()
        combined_regime = filter_instance.classify_regime(
            combined_trend_strength,
            combined_direction,
            0,  # momentum not used in classification
            1 - combined_trend_strength  # inverse for ranging
        )
        
        # Use most common volatility regime
        vol_regimes = [s.volatility_regime for s, _ in states]
        combined_vol_regime = max(set(vol_regimes), key=vol_regimes.count)
        
        # Create combined state
        self.combined_state = RegimeState(
            timestamp=states[0][0].timestamp,
            regime=combined_regime,
            volatility_regime=combined_vol_regime,
            adx=combined_adx,
            trend_direction=combined_direction,
            trend_strength=combined_trend_strength,
            volatility=combined_vol,
            alignment_score=combined_alignment
        )
        
        return self.combined_state
    
    def get_combined_state(self) -> Optional[RegimeState]:
        """Get combined regime state."""
        return self.combined_state
    
    def get_timeframe_states(self) -> Dict[str, RegimeState]:
        """Get individual timeframe states."""
        return {tf: f.get_current_state() for tf, f in self.filters.items()}


class RegimeBasedStrategyFilter:
    """
    Strategy filter that enables/disables strategies based on regime.
    
    Can be used as a wrapper around other strategies.
    """
    
    def __init__(self, regime_filter: RegimeFilter):
        """
        Initialize strategy filter.
        
        Args:
            regime_filter: RegimeFilter instance
        """
        self.regime_filter = regime_filter
        
        # Strategy type to regime mapping
        self.strategy_regime_map = {
            'trend_following': [
                MarketRegime.UPTREND,
                MarketRegime.STRONG_UPTREND,
                MarketRegime.DOWNTREND,
                MarketRegime.STRONG_DOWNTREND
            ],
            'mean_reversion': [
                MarketRegime.RANGING,
                MarketRegime.WEAK_UPTREND,
                MarketRegime.WEAK_DOWNTREND
            ],
            'breakout': [
                MarketRegime.RANGING,
                MarketRegime.LOW
            ],
            'momentum': [
                MarketRegime.UPTREND,
                MarketRegime.STRONG_UPTREND,
                MarketRegime.DOWNTREND,
                MarketRegime.STRONG_DOWNTREND
            ]
        }
    
    def should_run_strategy(self, strategy_type: str) -> bool:
        """
        Check if a strategy type should be active.
        
        Args:
            strategy_type: Type of strategy
            
        Returns:
            True if strategy should run
        """
        state = self.regime_filter.get_current_state()
        if not state:
            return True  # Default to allowing if no state
        
        allowed_regimes = self.strategy_regime_map.get(strategy_type, [])
        
        if not allowed_regimes:
            return True
        
        return state.regime in allowed_regimes
    
    def set_allowed_regimes(self, 
                           strategy_type: str, 
                           regimes: List[MarketRegime]) -> None:
        """Set allowed regimes for a strategy type."""
        self.strategy_regime_map[strategy_type] = regimes
    
    def get_position_size_modifier(self) -> float:
        """Get position size modifier based on regime."""
        state = self.regime_filter.get_current_state()
        if not state:
            return 1.0
        
        # Reduce size in extreme volatility
        if state.volatility_regime == VolatilityRegime.VERY_HIGH:
            return 0.5
        elif state.volatility_regime == VolatilityRegime.HIGH:
            return 0.75
        
        # Reduce size in weak trends
        if state.regime in [MarketRegime.WEAK_UPTREND, MarketRegime.WEAK_DOWNTREND]:
            return 0.8
        
        return 1.0


# Export all classes
__all__ = [
    'MarketRegime',
    'VolatilityRegime',
    'RegimeState',
    'RegimeFilter',
    'MultiTimeframeRegimeFilter',
    'RegimeBasedStrategyFilter'
]
