"""
TradeOS Walk-Forward Optimization Framework
===========================================
Implements walk-forward analysis for robust strategy optimization.
Prevents curve-fitting by testing on out-of-sample data.

Author: TradeOS Team
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import itertools
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import pickle
from pathlib import Path

from ..base_strategy import BaseStrategy, BacktestResult

logger = logging.getLogger(__name__)


class OptimizationMethod(Enum):
    """Optimization methods."""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    GENETIC = "genetic"


@dataclass
class ParameterSpace:
    """Defines parameter search space."""
    name: str
    param_type: str  # 'int', 'float', 'categorical'
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    values: Optional[List[Any]] = None
    step: Optional[Union[int, float]] = None
    
    def get_values(self) -> List[Any]:
        """Get all possible values for this parameter."""
        if self.param_type == 'categorical':
            return self.values or []
        
        if self.param_type == 'int':
            return list(range(int(self.min_value), int(self.max_value) + 1, 
                            int(self.step or 1)))
        
        if self.param_type == 'float':
            step = self.step or (self.max_value - self.min_value) / 10
            values = []
            current = self.min_value
            while current <= self.max_value:
                values.append(round(current, 6))
                current += step
            return values
        
        return []
    
    def sample_random(self) -> Any:
        """Sample a random value from the space."""
        if self.param_type == 'categorical':
            return np.random.choice(self.values)
        
        if self.param_type == 'int':
            return np.random.randint(self.min_value, self.max_value + 1)
        
        if self.param_type == 'float':
            return np.random.uniform(self.min_value, self.max_value)
        
        return None


@dataclass
class WFOResult:
    """Walk-forward optimization result."""
    strategy_name: str
    in_sample_results: List[BacktestResult] = field(default_factory=list)
    out_of_sample_results: List[BacktestResult] = field(default_factory=list)
    optimal_params_per_window: List[Dict[str, Any]] = field(default_factory=list)
    window_dates: List[Tuple[datetime, datetime, datetime, datetime]] = field(default_factory=list)
    
    # Aggregate metrics
    is_sharpe: float = 0.0
    oos_sharpe: float = 0.0
    is_return: float = 0.0
    oos_return: float = 0.0
    is_max_dd: float = 0.0
    oos_max_dd: float = 0.0
    
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate aggregate metrics."""
        if self.in_sample_results:
            self.is_sharpe = np.mean([r.sharpe_ratio for r in self.in_sample_results])
            self.is_return = np.mean([r.total_return for r in self.in_sample_results])
            self.is_max_dd = np.mean([r.max_drawdown for r in self.in_sample_results])
        
        if self.out_of_sample_results:
            self.oos_sharpe = np.mean([r.sharpe_ratio for r in self.out_of_sample_results])
            self.oos_return = np.mean([r.total_return for r in self.out_of_sample_results])
            self.oos_max_dd = np.mean([r.max_drawdown for r in self.out_of_sample_results])
        
        return {
            'is_sharpe': self.is_sharpe,
            'oos_sharpe': self.oos_sharpe,
            'is_return': self.is_return,
            'oos_return': self.oos_return,
            'is_max_dd': self.is_max_dd,
            'oos_max_dd': self.oos_max_dd,
            'sharpe_degradation': self.is_sharpe - self.oos_sharpe,
            'return_degradation': self.is_return - self.oos_return
        }
    
    def get_consolidated_params(self) -> Dict[str, Any]:
        """Get most common parameters across windows."""
        if not self.optimal_params_per_window:
            return {}
        
        # Count occurrences of each parameter value
        param_votes: Dict[str, Dict[Any, int]] = {}
        
        for params in self.optimal_params_per_window:
            for key, value in params.items():
                if key not in param_votes:
                    param_votes[key] = {}
                if value not in param_votes[key]:
                    param_votes[key][value] = 0
                param_votes[key][value] += 1
        
        # Select most common value for each parameter
        consolidated = {}
        for key, votes in param_votes.items():
            consolidated[key] = max(votes.items(), key=lambda x: x[1])[0]
        
        return consolidated
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'strategy_name': self.strategy_name,
            'metrics': self.calculate_metrics(),
            'consolidated_params': self.get_consolidated_params(),
            'num_windows': len(self.in_sample_results)
        }


class WalkForwardOptimizer:
    """
    Walk-forward optimization framework.
    
    Splits data into multiple in-sample / out-of-sample windows,
    optimizes on in-sample, validates on out-of-sample.
    """
    
    def __init__(self,
                 strategy_class: type,
                 param_spaces: List[ParameterSpace],
                 objective_func: Callable[[BacktestResult], float],
                 window_type: str = 'expanding',
                 train_pct: float = 0.7,
                 n_windows: int = 5,
                 min_bars_per_window: int = 100,
                 optimization_method: OptimizationMethod = OptimizationMethod.GRID_SEARCH,
                 n_random_samples: int = 50,
                 n_jobs: int = 1):
        """
        Initialize walk-forward optimizer.
        
        Args:
            strategy_class: Strategy class to optimize
            param_spaces: List of parameter search spaces
            objective_func: Function to maximize (takes BacktestResult, returns score)
            window_type: 'expanding' or 'rolling'
            train_pct: Percentage of window for training
            n_windows: Number of walk-forward windows
            min_bars_per_window: Minimum bars required per window
            optimization_method: Optimization method to use
            n_random_samples: Number of samples for random search
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        self.strategy_class = strategy_class
        self.param_spaces = param_spaces
        self.objective_func = objective_func
        self.window_type = window_type
        self.train_pct = train_pct
        self.n_windows = n_windows
        self.min_bars_per_window = min_bars_per_window
        self.optimization_method = optimization_method
        self.n_random_samples = n_random_samples
        self.n_jobs = n_jobs
        
        self.results: Optional[WFOResult] = None
    
    def create_windows(self, data: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create train/test windows for walk-forward analysis.
        
        Args:
            data: Full historical data
            
        Returns:
            List of (train_data, test_data) tuples
        """
        n_bars = len(data)
        bars_per_window = n_bars // self.n_windows
        
        if bars_per_window < self.min_bars_per_window:
            raise ValueError(f"Not enough data for {self.n_windows} windows. "
                           f"Need at least {self.min_bars_per_window * self.n_windows} bars.")
        
        windows = []
        
        for i in range(self.n_windows):
            if self.window_type == 'expanding':
                # Expanding window: train set grows
                train_end = (i + 1) * bars_per_window
                train_start = 0
                test_start = train_end
                test_end = min(test_start + int(bars_per_window * (1 - self.train_pct)), n_bars)
            else:  # rolling
                # Rolling window: fixed size train set
                window_start = i * bars_per_window
                train_start = window_start
                train_end = window_start + int(bars_per_window * self.train_pct)
                test_start = train_end
                test_end = min(window_start + bars_per_window, n_bars)
            
            train_data = data.iloc[train_start:train_end]
            test_data = data.iloc[test_start:test_end]
            
            if len(train_data) >= self.min_bars_per_window and len(test_data) >= 20:
                windows.append((train_data, test_data))
                
                logger.info(f"Window {i+1}: Train {train_start}-{train_end}, "
                          f"Test {test_start}-{test_end}")
        
        return windows
    
    def generate_param_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations for grid search."""
        if self.optimization_method == OptimizationMethod.RANDOM_SEARCH:
            combinations = []
            for _ in range(self.n_random_samples):
                params = {ps.name: ps.sample_random() for ps in self.param_spaces}
                combinations.append(params)
            return combinations
        
        # Grid search
        param_values = {ps.name: ps.get_values() for ps in self.param_spaces}
        keys = list(param_values.keys())
        values = [param_values[k] for k in keys]
        
        combinations = []
        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))
            combinations.append(params)
        
        return combinations
    
    def evaluate_params(self, 
                       params: Dict[str, Any],
                       train_data: pd.DataFrame,
                       test_data: pd.DataFrame) -> Tuple[float, BacktestResult, BacktestResult]:
        """
        Evaluate a parameter set on train and test data.
        
        Returns:
            Tuple of (score, train_result, test_result)
        """
        try:
            # Train on in-sample
            strategy = self.strategy_class(params=params)
            train_result = strategy.backtest(train_data)
            train_score = self.objective_func(train_result)
            
            # Test on out-of-sample
            strategy.reset()
            test_result = strategy.backtest(test_data)
            
            return train_score, train_result, test_result
        
        except Exception as e:
            logger.error(f"Error evaluating params {params}: {e}")
            return -np.inf, None, None
    
    def optimize_window(self, 
                       train_data: pd.DataFrame,
                       test_data: pd.DataFrame) -> Tuple[Dict[str, Any], BacktestResult, BacktestResult]:
        """
        Optimize parameters for a single window.
        
        Returns:
            Tuple of (best_params, train_result, test_result)
        """
        param_combinations = self.generate_param_combinations()
        logger.info(f"Testing {len(param_combinations)} parameter combinations")
        
        best_score = -np.inf
        best_params = None
        best_train_result = None
        best_test_result = None
        
        if self.n_jobs == 1:
            # Sequential evaluation
            for params in param_combinations:
                score, train_result, test_result = self.evaluate_params(
                    params, train_data, test_data
                )
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_train_result = train_result
                    best_test_result = test_result
        
        else:
            # Parallel evaluation
            n_workers = self.n_jobs if self.n_jobs > 0 else None
            
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(self.evaluate_params, params, train_data, test_data): params
                    for params in param_combinations
                }
                
                for future in as_completed(futures):
                    params = futures[future]
                    try:
                        score, train_result, test_result = future.result()
                        
                        if score > best_score:
                            best_score = score
                            best_params = params
                            best_train_result = train_result
                            best_test_result = test_result
                    
                    except Exception as e:
                        logger.error(f"Error in parallel evaluation: {e}")
        
        logger.info(f"Best params: {best_params}, Score: {best_score:.4f}")
        
        return best_params, best_train_result, best_test_result
    
    def run(self, data: pd.DataFrame) -> WFOResult:
        """
        Run walk-forward optimization.
        
        Args:
            data: Full historical data
            
        Returns:
            WFOResult with optimization results
        """
        logger.info("Starting walk-forward optimization...")
        
        windows = self.create_windows(data)
        
        result = WFOResult(strategy_name=self.strategy_class.__name__)
        
        for i, (train_data, test_data) in enumerate(windows):
            logger.info(f"\n=== Window {i+1}/{len(windows)} ===")
            
            best_params, train_result, test_result = self.optimize_window(
                train_data, test_data
            )
            
            if best_params:
                result.in_sample_results.append(train_result)
                result.out_of_sample_results.append(test_result)
                result.optimal_params_per_window.append(best_params)
                result.window_dates.append((
                    train_data.index[0],
                    train_data.index[-1],
                    test_data.index[0],
                    test_data.index[-1]
                ))
        
        result.calculate_metrics()
        self.results = result
        
        logger.info("\n=== Walk-Forward Optimization Complete ===")
        metrics = result.calculate_metrics()
        logger.info(f"In-Sample Sharpe: {metrics['is_sharpe']:.3f}")
        logger.info(f"Out-of-Sample Sharpe: {metrics['oos_sharpe']:.3f}")
        logger.info(f"Sharpe Degradation: {metrics['sharpe_degradation']:.3f}")
        
        return result
    
    def get_robust_params(self) -> Dict[str, Any]:
        """Get parameters that performed well across all windows."""
        if not self.results:
            return {}
        
        return self.results.get_consolidated_params()
    
    def save_results(self, filepath: str) -> None:
        """Save optimization results to file."""
        if not self.results:
            logger.warning("No results to save")
            return
        
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.results, f)
        
        logger.info(f"Results saved to {filepath}")
    
    def load_results(self, filepath: str) -> WFOResult:
        """Load optimization results from file."""
        with open(filepath, 'rb') as f:
            self.results = pickle.load(f)
        
        logger.info(f"Results loaded from {filepath}")
        return self.results


class MonteCarloSimulator:
    """
    Monte Carlo simulation for strategy robustness testing.
    
    Randomly perturbs trade results to assess strategy robustness.
    """
    
    def __init__(self, n_simulations: int = 1000):
        """
        Initialize Monte Carlo simulator.
        
        Args:
            n_simulations: Number of Monte Carlo simulations
        """
        self.n_simulations = n_simulations
    
    def simulate(self, 
                backtest_result: BacktestResult,
                confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation on backtest results.
        
        Args:
            backtest_result: Original backtest result
            confidence_level: Confidence level for intervals
            
        Returns:
            Dictionary with simulation statistics
        """
        if not backtest_result.trades:
            return {}
        
        trades_df = backtest_result.to_dataframe()
        returns = trades_df['pnl'].values
        
        # Run simulations
        simulated_returns = []
        simulated_max_dd = []
        
        for _ in range(self.n_simulations):
            # Bootstrap sample trades
            sample_indices = np.random.choice(len(returns), size=len(returns), replace=True)
            sample_returns = returns[sample_indices]
            
            # Calculate metrics
            total_return = sample_returns.sum()
            simulated_returns.append(total_return)
            
            # Calculate max drawdown for this simulation
            cumulative = np.cumsum(sample_returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = cumulative - running_max
            max_dd = drawdown.min()
            simulated_max_dd.append(max_dd)
        
        # Calculate statistics
        alpha = 1 - confidence_level
        
        return {
            'mean_return': np.mean(simulated_returns),
            'std_return': np.std(simulated_returns),
            'median_return': np.median(simulated_returns),
            'worst_case_return': np.percentile(simulated_returns, alpha * 100),
            'best_case_return': np.percentile(simulated_returns, (1 - alpha/2) * 100),
            'prob_profit': np.mean([r > 0 for r in simulated_returns]),
            'mean_max_dd': np.mean(simulated_max_dd),
            'worst_max_dd': np.percentile(simulated_max_dd, alpha * 100),
            'confidence_level': confidence_level,
            'n_simulations': self.n_simulations
        }


class StrategyOptimizer:
    """
    General strategy optimizer with multiple optimization methods.
    """
    
    def __init__(self,
                 strategy_class: type,
                 param_spaces: List[ParameterSpace],
                 objective_func: Optional[Callable[[BacktestResult], float]] = None):
        """
        Initialize strategy optimizer.
        
        Args:
            strategy_class: Strategy class to optimize
            param_spaces: Parameter search spaces
            objective_func: Objective function (default: Sharpe ratio)
        """
        self.strategy_class = strategy_class
        self.param_spaces = param_spaces
        
        if objective_func is None:
            self.objective_func = lambda r: r.sharpe_ratio if r.sharpe_ratio != 0 else r.total_return
        else:
            self.objective_func = objective_func
    
    def optimize(self,
                data: pd.DataFrame,
                method: OptimizationMethod = OptimizationMethod.GRID_SEARCH,
                n_samples: int = 100,
                n_jobs: int = 1) -> Tuple[Dict[str, Any], BacktestResult]:
        """
        Run optimization.
        
        Args:
            data: Historical data
            method: Optimization method
            n_samples: Number of samples (for random search)
            n_jobs: Number of parallel jobs
            
        Returns:
            Tuple of (best_params, backtest_result)
        """
        wfo = WalkForwardOptimizer(
            strategy_class=self.strategy_class,
            param_spaces=self.param_spaces,
            objective_func=self.objective_func,
            n_windows=1,  # Single window for simple optimization
            train_pct=1.0,  # Use all data for training
            optimization_method=method,
            n_random_samples=n_samples,
            n_jobs=n_jobs
        )
        
        # Create single window
        result = wfo.optimize_window(data, data.iloc[-1:])  # Dummy test data
        
        return result[0], result[1]
    
    def walk_forward_optimize(self,
                             data: pd.DataFrame,
                             n_windows: int = 5,
                             train_pct: float = 0.7,
                             n_jobs: int = 1) -> WFOResult:
        """
        Run walk-forward optimization.
        
        Args:
            data: Historical data
            n_windows: Number of windows
            train_pct: Training percentage per window
            n_jobs: Number of parallel jobs
            
        Returns:
            WFOResult
        """
        wfo = WalkForwardOptimizer(
            strategy_class=self.strategy_class,
            param_spaces=self.param_spaces,
            objective_func=self.objective_func,
            n_windows=n_windows,
            train_pct=train_pct,
            n_jobs=n_jobs
        )
        
        return wfo.run(data)


# Common objective functions
def sharpe_objective(result: BacktestResult) -> float:
    """Maximize Sharpe ratio."""
    return result.sharpe_ratio


def return_objective(result: BacktestResult) -> float:
    """Maximize total return."""
    return result.total_return


def risk_adjusted_objective(result: BacktestResult, 
                           return_weight: float = 0.5,
                           dd_weight: float = 0.5) -> float:
    """Maximize risk-adjusted return."""
    return (return_weight * result.total_return - 
            dd_weight * abs(result.max_drawdown))


def profit_factor_objective(result: BacktestResult) -> float:
    """Maximize profit factor."""
    return result.profit_factor


def combined_objective(result: BacktestResult) -> float:
    """Combined objective function."""
    if result.total_trades < 10:
        return -np.inf
    
    score = (
        0.3 * result.sharpe_ratio +
        0.3 * result.total_return * 10 +  # Scale return
        0.2 * min(result.profit_factor / 5, 1) +
        0.2 * (1 + result.max_drawdown)  # Less negative DD is better
    )
    
    return score


# Export all classes and functions
__all__ = [
    'OptimizationMethod',
    'ParameterSpace',
    'WFOResult',
    'WalkForwardOptimizer',
    'MonteCarloSimulator',
    'StrategyOptimizer',
    'sharpe_objective',
    'return_objective',
    'risk_adjusted_objective',
    'profit_factor_objective',
    'combined_objective'
]
