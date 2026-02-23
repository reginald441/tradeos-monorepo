"""
Performance Analytics Module
============================

Provides comprehensive performance metrics:
- Sharpe, Sortino, Calmar ratios
- Alpha, Beta calculations
- Drawdown analysis
- Trade statistics
"""

from .metrics import (
    PerformanceAnalyzer,
    RollingMetrics,
    PerformanceMetrics,
    quick_metrics,
    compare_strategies
)

__all__ = [
    'PerformanceAnalyzer',
    'RollingMetrics',
    'PerformanceMetrics',
    'quick_metrics',
    'compare_strategies'
]
