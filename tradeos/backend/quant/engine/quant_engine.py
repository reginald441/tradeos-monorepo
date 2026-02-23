"""
Main Quant Engine for TradeOS
=============================

Central coordinator for all quantitative analysis modules.
Provides unified interface for:
- Monte Carlo simulations
- Portfolio optimization
- Reinforcement learning
- Bayesian inference
- Covariance modeling
- Regime detection
- Performance analytics

Author: TradeOS Quant Team
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass
from datetime import datetime
import warnings
import json

# Import all quant modules
from ..config.quant_config import QuantEngineConfig, default_config
from ..monte_carlo.engine import MonteCarloEngine, MonteCarloConfig, quick_monte_carlo, quick_risk_of_ruin
from ..portfolio.optimizer import (
    PortfolioOptimizer, BlackLittermanModel, HierarchicalRiskParity,
    OptimizationObjective, optimize_portfolio, get_efficient_frontier
)
from ..rl.agent import DQNAgent, PPOAgent, TradingEnvironment, RLConfig, train_rl_agent
from ..bayesian.inference import (
    BayesianProbabilityUpdater, SignalConfidenceEstimator, RegimeProbabilityScorer,
    BayesianOptimizer, BayesianModelAveraging, ProbabilisticModel,
    quick_bayesian_update, estimate_signal_confidence
)
from ..covariance.dynamic_matrix import (
    DynamicCovarianceEstimator, VolatilityClusteringDetector, CrossAssetExposureAdjuster,
    estimate_covariance, detect_volatility_clustering
)
from ..regime.hmm_model import (
    MarketRegimeHMM, StructuralBreakDetector, RegimeSwitchingStrategy,
    detect_regimes, detect_structural_breaks
)
from ..analytics.metrics import (
    PerformanceAnalyzer, RollingMetrics, PerformanceMetrics,
    quick_metrics, compare_strategies
)
from ..reports.backtest_report import BacktestReport, BacktestConfig, TradeRecord, generate_backtest_report


@dataclass
class QuantResult:
    """Container for quant engine results."""
    success: bool
    data: Dict[str, Any]
    errors: List[str]
    execution_time: float
    timestamp: datetime


class TradeOSQuantEngine:
    """
    Main Quantitative Engine for TradeOS.
    
    Provides a unified interface to all quantitative analysis capabilities:
    - Risk analysis via Monte Carlo
    - Portfolio construction and optimization
    - RL-based strategy adaptation
    - Bayesian signal processing
    - Dynamic risk modeling
    - Market regime detection
    - Performance attribution
    """
    
    def __init__(self, config: Optional[QuantEngineConfig] = None):
        """
        Initialize the Quant Engine.
        
        Args:
            config: QuantEngineConfig instance, uses defaults if None
        """
        self.config = config or default_config
        self._init_engines()
        self._cache = {}
        
        if self.config.verbose:
            print("🚀 TradeOS Quant Engine initialized")
            print(f"   Configuration: {self._get_config_summary()}")
    
    def _init_engines(self):
        """Initialize all sub-engines."""
        # Monte Carlo
        self.mc_engine = MonteCarloEngine(
            MonteCarloConfig(
                n_simulations=self.config.monte_carlo.n_simulations,
                confidence_level=self.config.monte_carlo.confidence_level,
                random_seed=self.config.random_seed,
                parallel=self.config.monte_carlo.parallel,
                n_workers=self.config.monte_carlo.n_workers
            )
        )
        
        # Performance analyzer
        self.analyzer = PerformanceAnalyzer(
            risk_free_rate=self.config.metrics.risk_free_rate,
            frequency=self.config.metrics.frequency,
            annualize=self.config.metrics.annualize
        )
        
        # Covariance estimator
        self.cov_estimator = DynamicCovarianceEstimator(
            config=self.config.covariance
        )
        
        # Regime detector
        self.regime_detector = MarketRegimeHMM(
            config=self.config.hmm
        )
        
        if self.config.verbose:
            print("   ✓ Sub-engines initialized")
    
    def _get_config_summary(self) -> str:
        """Get configuration summary string."""
        return (f"MC={self.config.monte_carlo.n_simulations}, "
                f"Portfolio={self.config.portfolio.objective}, "
                f"Freq={self.config.metrics.frequency}")
    
    # ==================== MONTE CARLO METHODS ====================
    
    def run_monte_carlo(
        self,
        returns: np.ndarray,
        initial_capital: float = 100000.0,
        method: str = "bootstrap",
        **kwargs
    ) -> Dict:
        """
        Run Monte Carlo simulation.
        
        Args:
            returns: Historical returns
            initial_capital: Starting capital
            method: Simulation method
            **kwargs: Additional parameters
            
        Returns:
            Simulation results
        """
        if self.config.verbose:
            print(f"🎲 Running Monte Carlo ({self.config.monte_carlo.n_simulations} simulations)...")
        
        result = self.mc_engine.simulate_equity_curve(
            returns=returns,
            initial_capital=initial_capital,
            method=method,
            **kwargs
        )
        
        return {
            'median_final_equity': float(np.median(result.final_equities)),
            'mean_final_equity': float(np.mean(result.final_equities)),
            'worst_case_equity': float(np.min(result.final_equities)),
            'best_case_equity': float(np.max(result.final_equities)),
            'median_max_drawdown': float(np.median(result.max_drawdowns)),
            'worst_drawdown': float(np.min(result.max_drawdowns)),
            'median_sharpe': float(np.median(result.sharpe_ratios)),
            'value_at_risk_95': result.value_at_risk.get('var_95', 0),
            'confidence_intervals': result.confidence_intervals,
            'path_statistics': result.path_statistics
        }
    
    def calculate_risk_of_ruin(
        self,
        returns: np.ndarray,
        initial_capital: float = 100000.0,
        ruin_threshold: float = 0.5,
        **kwargs
    ) -> Dict:
        """
        Calculate risk of ruin.
        
        Args:
            returns: Historical returns
            initial_capital: Starting capital
            ruin_threshold: Ruin threshold
            **kwargs: Additional parameters
            
        Returns:
            Risk of ruin results
        """
        result = self.mc_engine.calculate_risk_of_ruin(
            returns=returns,
            initial_capital=initial_capital,
            ruin_threshold=ruin_threshold,
            **kwargs
        )
        
        return {
            'ruin_probability': result.ruin_probability,
            'confidence_interval': result.confidence_interval,
            'time_to_ruin_mean': result.time_to_ruin_mean,
            'simulations_ruined': result.simulations_ruined,
            'total_simulations': result.total_simulations
        }
    
    # ==================== PORTFOLIO METHODS ====================
    
    def optimize_portfolio(
        self,
        returns_data: np.ndarray,
        objective: Optional[str] = None,
        constraints: Optional[Dict] = None,
        **kwargs
    ) -> Dict:
        """
        Optimize portfolio weights.
        
        Args:
            returns_data: Historical returns (n_periods x n_assets)
            objective: Optimization objective
            constraints: Optional constraints
            **kwargs: Additional parameters
            
        Returns:
            Optimization results
        """
        if self.config.verbose:
            print(f"📊 Optimizing portfolio ({objective or self.config.portfolio.objective})...")
        
        optimizer = PortfolioOptimizer(
            returns_data=returns_data,
            risk_free_rate=self.config.portfolio.risk_free_rate
        )
        
        obj_map = {
            'max_sharpe': OptimizationObjective.MAX_SHARPE,
            'min_variance': OptimizationObjective.MIN_VARIANCE,
            'risk_parity': OptimizationObjective.RISK_PARITY,
            'max_diversification': OptimizationObjective.MAX_DIVERSIFICATION
        }
        
        obj = obj_map.get(objective or self.config.portfolio.objective, OptimizationObjective.MAX_SHARPE)
        result = optimizer.optimize(obj)
        
        return {
            'weights': result.weights.tolist(),
            'expected_return': result.expected_return,
            'volatility': result.volatility,
            'sharpe_ratio': result.sharpe_ratio,
            'diversification_ratio': result.diversification_ratio,
            'success': result.optimization_success,
            'iterations': result.iterations
        }
    
    def get_efficient_frontier(
        self,
        returns_data: np.ndarray,
        n_points: int = 50,
        **kwargs
    ) -> pd.DataFrame:
        """
        Calculate efficient frontier.
        
        Args:
            returns_data: Historical returns
            n_points: Number of frontier points
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with efficient frontier
        """
        optimizer = PortfolioOptimizer(
            returns_data=returns_data,
            risk_free_rate=self.config.portfolio.risk_free_rate
        )
        
        return optimizer.efficient_frontier(n_points=n_points)
    
    def hierarchical_risk_parity(
        self,
        returns_data: np.ndarray,
        **kwargs
    ) -> Dict:
        """
        Calculate Hierarchical Risk Parity weights.
        
        Args:
            returns_data: Historical returns
            **kwargs: Additional parameters
            
        Returns:
            HRP weights
        """
        hrp = HierarchicalRiskParity(returns_data)
        weights = hrp.allocate()
        
        return {
            'weights': weights.tolist(),
            'method': 'hierarchical_risk_parity'
        }
    
    # ==================== BAYESIAN METHODS ====================
    
    def bayesian_update(
        self,
        prior_params: Dict,
        data: np.ndarray,
        model_type: str = "normal"
    ) -> Dict:
        """
        Perform Bayesian update.
        
        Args:
            prior_params: Prior parameters
            data: Observed data
            model_type: Type of model ('normal', 'beta', 'gamma')
            
        Returns:
            Posterior parameters
        """
        updater = BayesianProbabilityUpdater(self.config.bayesian)
        
        if model_type == "normal":
            return updater.update_normal_normal(
                prior_mean=prior_params.get('mean', 0),
                prior_variance=prior_params.get('variance', 1),
                data=data
            )
        elif model_type == "beta":
            return updater.update_beta_binomial(
                prior_alpha=prior_params.get('alpha', 1),
                prior_beta=prior_params.get('beta', 1),
                successes=np.sum(data > 0),
                trials=len(data)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def estimate_signal_confidence(
        self,
        signal_history: List[Tuple[float, float]],
        current_signal: float
    ) -> Dict:
        """
        Estimate confidence in a trading signal.
        
        Args:
            signal_history: List of (signal, outcome) pairs
            current_signal: Current signal value
            
        Returns:
            Confidence metrics
        """
        estimator = SignalConfidenceEstimator(self.config.bayesian)
        return estimator.estimate_signal_confidence(signal_history, current_signal)
    
    def bayesian_optimize(
        self,
        objective_func: Callable,
        param_bounds: Dict[str, Tuple[float, float]],
        n_iterations: int = 50
    ) -> Dict:
        """
        Bayesian optimization for parameter tuning.
        
        Args:
            objective_func: Function to optimize
            param_bounds: Parameter bounds
            n_iterations: Number of iterations
            
        Returns:
            Optimization results
        """
        optimizer = BayesianOptimizer(param_bounds)
        return optimizer.optimize(objective_func, n_iterations)
    
    # ==================== COVARIANCE METHODS ====================
    
    def estimate_covariance(
        self,
        returns: np.ndarray,
        method: Optional[str] = None,
        **kwargs
    ) -> Dict:
        """
        Estimate covariance matrix.
        
        Args:
            returns: Returns matrix
            method: Estimation method
            **kwargs: Additional parameters
            
        Returns:
            Covariance estimate
        """
        method = method or self.config.covariance.method
        
        config = self.config.covariance
        config.method = method
        
        estimator = DynamicCovarianceEstimator(config)
        return estimator.fit(returns)
    
    def detect_volatility_clustering(self, returns: np.ndarray) -> Dict:
        """
        Detect volatility clustering.
        
        Args:
            returns: Return series
            
        Returns:
            Detection results
        """
        detector = VolatilityClusteringDetector()
        return detector.detect(returns)
    
    def adjust_for_correlation(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray,
        max_correlation: float = 0.7
    ) -> Dict:
        """
        Adjust weights for cross-asset correlation.
        
        Args:
            weights: Target weights
            cov_matrix: Covariance matrix
            max_correlation: Maximum allowed correlation
            
        Returns:
            Adjusted weights and analysis
        """
        adjuster = CrossAssetExposureAdjuster(max_correlation)
        return adjuster.adjust_exposures(weights, cov_matrix)
    
    # ==================== REGIME METHODS ====================
    
    def detect_regimes(
        self,
        features: np.ndarray,
        returns: Optional[np.ndarray] = None,
        n_regimes: Optional[int] = None
    ) -> Dict:
        """
        Detect market regimes using HMM.
        
        Args:
            features: Feature matrix
            returns: Optional returns for labeling
            n_regimes: Number of regimes
            
        Returns:
            Regime detection results
        """
        if self.config.verbose:
            print(f"🔍 Detecting market regimes...")
        
        n_regimes = n_regimes or self.config.hmm.n_components
        
        config = self.config.hmm
        config.n_components = n_regimes
        
        detector = MarketRegimeHMM(config)
        result = detector.fit(features, returns)
        
        return {
            'regimes': result.regimes.tolist(),
            'regime_probabilities': result.regime_probs.tolist(),
            'transition_matrix': result.transition_matrix.tolist(),
            'means': result.means.tolist(),
            'log_likelihood': result.log_likelihood,
            'aic': result.aic,
            'bic': result.bic,
            'convergence': result.convergence
        }
    
    def detect_structural_breaks(
        self,
        data: np.ndarray,
        max_breaks: int = 5
    ) -> Dict:
        """
        Detect structural breaks in time series.
        
        Args:
            data: Time series data
            max_breaks: Maximum number of breaks
            
        Returns:
            Break detection results
        """
        detector = StructuralBreakDetector()
        return detector.detect_breaks(data, max_breaks)
    
    # ==================== PERFORMANCE METHODS ====================
    
    def calculate_metrics(
        self,
        returns: np.ndarray,
        benchmark: Optional[np.ndarray] = None,
        trades: Optional[List[Dict]] = None
    ) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            returns: Strategy returns
            benchmark: Optional benchmark returns
            trades: Optional trade list
            
        Returns:
            PerformanceMetrics object
        """
        return self.analyzer.calculate_all_metrics(returns, benchmark, trades)
    
    def quick_metrics(
        self,
        returns: np.ndarray,
        benchmark: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Quick performance metrics.
        
        Args:
            returns: Strategy returns
            benchmark: Optional benchmark returns
            
        Returns:
            Key metrics dictionary
        """
        return quick_metrics(returns, benchmark, self.config.metrics.risk_free_rate)
    
    def compare_strategies(
        self,
        returns_dict: Dict[str, np.ndarray],
        benchmark: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Compare multiple strategies.
        
        Args:
            returns_dict: Dictionary of strategy returns
            benchmark: Optional benchmark
            
        Returns:
            Comparison DataFrame
        """
        return compare_strategies(returns_dict, benchmark, self.config.metrics.risk_free_rate)
    
    def rolling_metrics(
        self,
        returns: np.ndarray,
        metrics: List[str],
        window: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Calculate rolling metrics.
        
        Args:
            returns: Returns series
            metrics: List of metrics to calculate
            window: Rolling window size
            
        Returns:
            DataFrame with rolling metrics
        """
        window = window or 63  # Default 3 months
        
        rolling = RollingMetrics(
            window=window,
            risk_free_rate=self.config.metrics.risk_free_rate,
            frequency=self.config.metrics.frequency
        )
        
        return rolling.calculate_multiple(returns, metrics)
    
    # ==================== REPORTING METHODS ====================
    
    def generate_backtest_report(
        self,
        returns: np.ndarray,
        equity_curve: np.ndarray,
        dates: Optional[List[datetime]] = None,
        trades: Optional[List[TradeRecord]] = None
    ) -> Dict:
        """
        Generate comprehensive backtest report.
        
        Args:
            returns: Strategy returns
            equity_curve: Equity curve
            dates: Optional dates
            trades: Optional trades
            
        Returns:
            Report dictionary
        """
        report = BacktestReport(self.config.backtest)
        report.set_equity_curve(equity_curve, dates)
        
        if trades:
            for trade in trades:
                report.add_trade(trade)
        
        return report.generate_report()
    
    # ==================== INTEGRATED WORKFLOWS ====================
    
    def full_strategy_analysis(
        self,
        returns: np.ndarray,
        equity_curve: np.ndarray,
        features: Optional[np.ndarray] = None,
        benchmark: Optional[np.ndarray] = None,
        trades: Optional[List[TradeRecord]] = None,
        dates: Optional[List[datetime]] = None
    ) -> Dict:
        """
        Perform full strategy analysis.
        
        This is a comprehensive analysis that includes:
        - Performance metrics
        - Risk analysis (VaR, CVaR, drawdown)
        - Monte Carlo simulation
        - Regime detection (if features provided)
        - Volatility clustering detection
        
        Args:
            returns: Strategy returns
            equity_curve: Equity curve
            features: Optional features for regime detection
            benchmark: Optional benchmark
            trades: Optional trades
            dates: Optional dates
            
        Returns:
            Comprehensive analysis results
        """
        if self.config.verbose:
            print("\n" + "=" * 60)
            print("🔬 FULL STRATEGY ANALYSIS")
            print("=" * 60)
        
        results = {}
        
        # 1. Performance Metrics
        if self.config.verbose:
            print("\n📈 Calculating performance metrics...")
        results['performance'] = self.quick_metrics(returns, benchmark)
        
        # 2. Monte Carlo
        if self.config.verbose:
            print("🎲 Running Monte Carlo simulation...")
        results['monte_carlo'] = self.run_monte_carlo(returns, equity_curve[0])
        
        # 3. Risk of Ruin
        if self.config.verbose:
            print("⚠️  Calculating risk of ruin...")
        results['risk_of_ruin'] = self.calculate_risk_of_ruin(returns, equity_curve[0])
        
        # 4. Regime Detection
        if features is not None:
            if self.config.verbose:
                print("🔍 Detecting market regimes...")
            results['regimes'] = self.detect_regimes(features, returns)
        
        # 5. Volatility Clustering
        if self.config.verbose:
            print("📊 Analyzing volatility clustering...")
        results['volatility_clustering'] = self.detect_volatility_clustering(returns)
        
        # 6. Backtest Report
        if self.config.verbose:
            print("📋 Generating backtest report...")
        results['backtest_report'] = self.generate_backtest_report(
            returns, equity_curve, dates, trades
        )
        
        if self.config.verbose:
            print("\n✅ Analysis complete!")
            print("=" * 60)
        
        return results
    
    def portfolio_construction_workflow(
        self,
        returns_data: np.ndarray,
        asset_names: Optional[List[str]] = None,
        current_weights: Optional[np.ndarray] = None,
        constraints: Optional[Dict] = None
    ) -> Dict:
        """
        Complete portfolio construction workflow.
        
        Includes:
        - Covariance estimation
        - Multiple optimization methods
        - Risk analysis
        - Correlation adjustment
        
        Args:
            returns_data: Asset returns matrix
            asset_names: Optional asset names
            current_weights: Optional current portfolio weights
            constraints: Optional constraints
            
        Returns:
            Portfolio construction results
        """
        if self.config.verbose:
            print("\n" + "=" * 60)
            print("📊 PORTFOLIO CONSTRUCTION WORKFLOW")
            print("=" * 60)
        
        results = {}
        
        # 1. Covariance Estimation
        if self.config.verbose:
            print("\n📈 Estimating covariance matrix...")
        cov_result = self.estimate_covariance(returns_data)
        results['covariance'] = cov_result
        
        # 2. Multiple Optimizations
        if self.config.verbose:
            print("\n🎯 Running portfolio optimizations...")
        
        objectives = ['max_sharpe', 'min_variance', 'risk_parity']
        optimizations = {}
        
        for obj in objectives:
            try:
                opt_result = self.optimize_portfolio(returns_data, objective=obj)
                optimizations[obj] = opt_result
            except Exception as e:
                warnings.warn(f"Optimization failed for {obj}: {e}")
        
        results['optimizations'] = optimizations
        
        # 3. Hierarchical Risk Parity
        if self.config.verbose:
            print("\n🌳 Calculating HRP weights...")
        try:
            hrp_result = self.hierarchical_risk_parity(returns_data)
            results['hrp'] = hrp_result
        except Exception as e:
            warnings.warn(f"HRP failed: {e}")
        
        # 4. Correlation Adjustment (if current weights provided)
        if current_weights is not None:
            if self.config.verbose:
                print("\n🔗 Adjusting for correlations...")
            adj_result = self.adjust_for_correlation(
                current_weights,
                cov_result['covariance']
            )
            results['correlation_adjustment'] = adj_result
        
        # 5. Efficient Frontier
        if self.config.verbose:
            print("\n📉 Calculating efficient frontier...")
        try:
            frontier = self.get_efficient_frontier(returns_data, n_points=20)
            results['efficient_frontier'] = frontier.to_dict()
        except Exception as e:
            warnings.warn(f"Efficient frontier failed: {e}")
        
        if self.config.verbose:
            print("\n✅ Portfolio construction complete!")
            print("=" * 60)
        
        return results
    
    # ==================== UTILITY METHODS ====================
    
    def save_config(self, filepath: str):
        """Save current configuration to file."""
        self.config.to_json(filepath)
        if self.config.verbose:
            print(f"💾 Configuration saved to {filepath}")
    
    def load_config(self, filepath: str):
        """Load configuration from file."""
        self.config = QuantEngineConfig.from_json(filepath)
        self._init_engines()
        if self.config.verbose:
            print(f"📂 Configuration loaded from {filepath}")
    
    def clear_cache(self):
        """Clear internal cache."""
        self._cache = {}
        if self.config.verbose:
            print("🗑️  Cache cleared")
    
    def get_info(self) -> Dict:
        """Get engine information."""
        return {
            'version': '1.0.0',
            'config': self.config.to_dict(),
            'modules': {
                'monte_carlo': True,
                'portfolio': True,
                'rl': True,
                'bayesian': True,
                'covariance': True,
                'regime': True,
                'analytics': True
            }
        }


# Convenience function for quick analysis
def quick_analyze(
    returns: np.ndarray,
    equity_curve: Optional[np.ndarray] = None,
    **kwargs
) -> Dict:
    """
    Quick strategy analysis.
    
    Args:
        returns: Strategy returns
        equity_curve: Optional equity curve
        **kwargs: Additional parameters
        
    Returns:
        Analysis results
    """
    if equity_curve is None:
        equity_curve = 100000 * np.cumprod(1 + returns)
    
    engine = TradeOSQuantEngine()
    return engine.full_strategy_analysis(returns, equity_curve, **kwargs)


# Singleton instance for global use
_quant_engine: Optional[TradeOSQuantEngine] = None


def get_quant_engine(config: Optional[QuantEngineConfig] = None) -> TradeOSQuantEngine:
    """
    Get or create global quant engine instance.
    
    Args:
        config: Optional configuration
        
    Returns:
        TradeOSQuantEngine instance
    """
    global _quant_engine
    
    if _quant_engine is None or config is not None:
        _quant_engine = TradeOSQuantEngine(config)
    
    return _quant_engine


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    print("\n" + "=" * 60)
    print("🚀 TradeOS Quant Engine Demo")
    print("=" * 60)
    
    # Initialize engine
    engine = TradeOSQuantEngine()
    
    # Generate sample data
    n_days = 252
    returns = np.random.normal(0.0005, 0.02, n_days)
    equity_curve = 100000 * np.cumprod(1 + returns)
    
    # Quick metrics
    print("\n1. Quick Metrics:")
    metrics = engine.quick_metrics(returns)
    for key, value in metrics.items():
        print(f"   {key}: {value:.4f}")
    
    # Monte Carlo
    print("\n2. Monte Carlo Simulation:")
    mc_result = engine.run_monte_carlo(returns)
    print(f"   Median Final Equity: ${mc_result['median_final_equity']:,.2f}")
    print(f"   Worst Drawdown: {mc_result['worst_drawdown']:.2%}")
    
    # Risk of Ruin
    print("\n3. Risk of Ruin:")
    ror = engine.calculate_risk_of_ruin(returns)
    print(f"   Ruin Probability: {ror['ruin_probability']:.2%}")
    
    # Full analysis
    print("\n4. Full Strategy Analysis:")
    analysis = engine.full_strategy_analysis(returns, equity_curve)
    print(f"   Performance Sharpe: {analysis['performance']['sharpe_ratio']:.2f}")
    print(f"   Monte Carlo VaR 95: {analysis['monte_carlo']['value_at_risk_95']:.2%}")
    
    print("\n" + "=" * 60)
    print("✅ Demo complete!")
    print("=" * 60)
