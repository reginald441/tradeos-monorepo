"""
TradeOS Quant Module
====================

Advanced quantitative analysis layer for TradeOS.

This module provides comprehensive quantitative finance capabilities:
- Monte Carlo simulation
- Portfolio optimization
- Reinforcement learning
- Bayesian inference
- Dynamic covariance modeling
- Regime detection
- Performance analytics

Example:
    >>> from tradeos.backend.quant import TradeOSQuantEngine
    >>> engine = TradeOSQuantEngine()
    >>> results = engine.quick_metrics(returns)
"""

__version__ = "1.0.0"
__author__ = "TradeOS Quant Team"

# Main engine
from .engine.quant_engine import TradeOSQuantEngine, get_quant_engine, quick_analyze

# Configuration
from .config.quant_config import (
    QuantEngineConfig,
    MonteCarloConfig,
    PortfolioConfig,
    RLConfig,
    BayesianConfig,
    CovarianceConfig,
    HMMConfig,
    MetricsConfig,
    BacktestConfig,
    PresetConfigs,
    default_config
)

# Monte Carlo
from .monte_carlo.engine import (
    MonteCarloEngine,
    MonteCarloResult,
    RiskOfRuinResult,
    BootstrapAnalyzer,
    quick_monte_carlo,
    quick_risk_of_ruin
)

# Portfolio Optimization
from .portfolio.optimizer import (
    PortfolioOptimizer,
    BlackLittermanModel,
    HierarchicalRiskParity,
    TransactionCostOptimizer,
    OptimizationObjective,
    OptimizationResult,
    OptimizationConstraints,
    optimize_portfolio,
    get_efficient_frontier
)

# Reinforcement Learning
from .rl.agent import (
    DQNAgent,
    PPOAgent,
    TradingEnvironment,
    ReplayBuffer,
    StrategyParameterAdapter,
    RegimeBasedAgent,
    train_rl_agent,
    evaluate_agent
)

# Bayesian Inference
from .bayesian.inference import (
    BayesianProbabilityUpdater,
    SignalConfidenceEstimator,
    RegimeProbabilityScorer,
    BayesianOptimizer,
    BayesianModelAveraging,
    ProbabilisticModel,
    quick_bayesian_update,
    estimate_signal_confidence
)

# Covariance Modeling
from .covariance.dynamic_matrix import (
    DynamicCovarianceEstimator,
    VolatilityClusteringDetector,
    CrossAssetExposureAdjuster,
    CovarianceConfig as CovConfig,
    estimate_covariance,
    detect_volatility_clustering,
    adjust_for_correlation
)

# Regime Detection
from .regime.hmm_model import (
    MarketRegimeHMM,
    StructuralBreakDetector,
    RegimeSwitchingStrategy,
    RegimeResult,
    HMMConfig as RegimeConfig,
    detect_regimes,
    detect_structural_breaks
)

# Performance Analytics
from .analytics.metrics import (
    PerformanceAnalyzer,
    RollingMetrics,
    PerformanceMetrics,
    quick_metrics,
    compare_strategies
)

# Backtest Reporting
from .reports.backtest_report import (
    BacktestReport,
    BacktestConfig as ReportConfig,
    TradeRecord,
    generate_backtest_report
)

# Convenience imports
import numpy as np
import pandas as pd

__all__ = [
    # Main engine
    'TradeOSQuantEngine',
    'get_quant_engine',
    'quick_analyze',
    
    # Configuration
    'QuantEngineConfig',
    'MonteCarloConfig',
    'PortfolioConfig',
    'RLConfig',
    'BayesianConfig',
    'CovarianceConfig',
    'HMMConfig',
    'MetricsConfig',
    'BacktestConfig',
    'PresetConfigs',
    'default_config',
    
    # Monte Carlo
    'MonteCarloEngine',
    'MonteCarloResult',
    'RiskOfRuinResult',
    'BootstrapAnalyzer',
    'quick_monte_carlo',
    'quick_risk_of_ruin',
    
    # Portfolio
    'PortfolioOptimizer',
    'BlackLittermanModel',
    'HierarchicalRiskParity',
    'TransactionCostOptimizer',
    'OptimizationObjective',
    'OptimizationResult',
    'OptimizationConstraints',
    'optimize_portfolio',
    'get_efficient_frontier',
    
    # RL
    'DQNAgent',
    'PPOAgent',
    'TradingEnvironment',
    'ReplayBuffer',
    'StrategyParameterAdapter',
    'RegimeBasedAgent',
    'train_rl_agent',
    'evaluate_agent',
    
    # Bayesian
    'BayesianProbabilityUpdater',
    'SignalConfidenceEstimator',
    'RegimeProbabilityScorer',
    'BayesianOptimizer',
    'BayesianModelAveraging',
    'ProbabilisticModel',
    'quick_bayesian_update',
    'estimate_signal_confidence',
    
    # Covariance
    'DynamicCovarianceEstimator',
    'VolatilityClusteringDetector',
    'CrossAssetExposureAdjuster',
    'estimate_covariance',
    'detect_volatility_clustering',
    'adjust_for_correlation',
    
    # Regime
    'MarketRegimeHMM',
    'StructuralBreakDetector',
    'RegimeSwitchingStrategy',
    'RegimeResult',
    'detect_regimes',
    'detect_structural_breaks',
    
    # Analytics
    'PerformanceAnalyzer',
    'RollingMetrics',
    'PerformanceMetrics',
    'quick_metrics',
    'compare_strategies',
    
    # Reporting
    'BacktestReport',
    'TradeRecord',
    'generate_backtest_report',
]


def get_version() -> str:
    """Get the version of the quant module."""
    return __version__


def get_available_modules() -> dict:
    """Get information about available modules and their dependencies."""
    modules = {
        'numpy': True,
        'pandas': True,
        'scipy': True,
        'sklearn': True,
    }
    
    # Check optional dependencies
    try:
        import torch
        modules['torch'] = True
    except ImportError:
        modules['torch'] = False
    
    try:
        import hmmlearn
        modules['hmmlearn'] = True
    except ImportError:
        modules['hmmlearn'] = False
    
    try:
        import pymc
        modules['pymc'] = True
    except ImportError:
        modules['pymc'] = False
    
    try:
        import pyro
        modules['pyro'] = True
    except ImportError:
        modules['pyro'] = False
    
    return modules


# Print welcome message when module is imported
if __name__ != "__main__":
    import sys
    if 'ipykernel' in sys.modules or 'IPython' in sys.modules:
        # Running in Jupyter/IPython
        pass  # Don't print in notebooks
    else:
        # Print minimal info
        print(f"TradeOS Quant Module v{__version__} loaded")
