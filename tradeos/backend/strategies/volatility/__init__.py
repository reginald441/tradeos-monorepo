"""
TradeOS Volatility-Based Strategies Module
==========================================
Collection of volatility-based trading strategies.
"""

from .vol_strategies import (
    VolatilityRegime,
    VolatilityBreakoutStrategy,
    ATRChannelStrategy,
    VolatilityRegimeStrategy,
    VolatilityContractionStrategy,
    GapVolatilityStrategy
)

__all__ = [
    'VolatilityRegime',
    'VolatilityBreakoutStrategy',
    'ATRChannelStrategy',
    'VolatilityRegimeStrategy',
    'VolatilityContractionStrategy',
    'GapVolatilityStrategy'
]
