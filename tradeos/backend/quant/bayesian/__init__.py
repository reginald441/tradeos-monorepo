"""
Bayesian Inference Module
=========================

Provides Bayesian methods for trading:
- Probability updating
- Signal confidence estimation
- Regime probability scoring
- Bayesian optimization
"""

from .inference import (
    BayesianProbabilityUpdater,
    SignalConfidenceEstimator,
    RegimeProbabilityScorer,
    BayesianOptimizer,
    BayesianModelAveraging,
    ProbabilisticModel,
    quick_bayesian_update,
    estimate_signal_confidence
)

__all__ = [
    'BayesianProbabilityUpdater',
    'SignalConfidenceEstimator',
    'RegimeProbabilityScorer',
    'BayesianOptimizer',
    'BayesianModelAveraging',
    'ProbabilisticModel',
    'quick_bayesian_update',
    'estimate_signal_confidence'
]
