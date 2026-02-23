"""
Quant Engine Module
===================

Main coordinator for all quantitative analysis capabilities.
"""

from .quant_engine import (
    TradeOSQuantEngine,
    QuantResult,
    get_quant_engine,
    quick_analyze
)

__all__ = [
    'TradeOSQuantEngine',
    'QuantResult',
    'get_quant_engine',
    'quick_analyze'
]
