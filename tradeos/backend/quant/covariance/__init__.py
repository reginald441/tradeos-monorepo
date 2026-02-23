"""
Dynamic Covariance Module
=========================

Provides dynamic covariance and correlation modeling:
- Exponentially weighted covariance
- GARCH volatility modeling
- DCC (Dynamic Conditional Correlation)
- Volatility clustering detection
"""

from .dynamic_matrix import (
    DynamicCovarianceEstimator,
    VolatilityClusteringDetector,
    CrossAssetExposureAdjuster,
    CovarianceConfig,
    estimate_covariance,
    detect_volatility_clustering,
    adjust_for_correlation
)

__all__ = [
    'DynamicCovarianceEstimator',
    'VolatilityClusteringDetector',
    'CrossAssetExposureAdjuster',
    'CovarianceConfig',
    'estimate_covariance',
    'detect_volatility_clustering',
    'adjust_for_correlation'
]
