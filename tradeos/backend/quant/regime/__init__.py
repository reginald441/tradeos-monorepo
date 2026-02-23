"""
Regime Detection Module
=======================

Provides market regime detection using:
- Hidden Markov Models (HMM)
- Structural break detection
- Regime-based strategy switching
"""

from .hmm_model import (
    MarketRegimeHMM,
    StructuralBreakDetector,
    RegimeSwitchingStrategy,
    RegimeResult,
    HMMConfig,
    detect_regimes,
    detect_structural_breaks
)

__all__ = [
    'MarketRegimeHMM',
    'StructuralBreakDetector',
    'RegimeSwitchingStrategy',
    'RegimeResult',
    'HMMConfig',
    'detect_regimes',
    'detect_structural_breaks'
]
