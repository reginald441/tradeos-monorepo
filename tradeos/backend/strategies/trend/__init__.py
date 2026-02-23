"""
TradeOS Trend Following Strategies Module
=========================================
Collection of trend-following trading strategies.
"""

from .trend_following import (
    EMACrossoverStrategy,
    ADXTrendStrategy,
    BreakoutStrategy,
    SuperTrendStrategy,
    MACDTrendStrategy
)

__all__ = [
    'EMACrossoverStrategy',
    'ADXTrendStrategy',
    'BreakoutStrategy',
    'SuperTrendStrategy',
    'MACDTrendStrategy'
]
