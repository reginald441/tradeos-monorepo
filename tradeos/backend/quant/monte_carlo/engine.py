"""
Monte Carlo Simulation Engine for TradeOS
=========================================

Provides comprehensive Monte Carlo simulation capabilities for:
- Trade sequence reshuffling
- Equity curve simulation
- Risk-of-ruin calculation
- Confidence intervals
- Bootstrap resampling
- Worst-case scenario modeling

Author: TradeOS Quant Team
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Callable, Union
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import minimize_scalar
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from functools import partial


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulations."""
    n_simulations: int = 10000
    confidence_level: float = 0.95
    random_seed: Optional[int] = None
    parallel: bool = True
    n_workers: int = -1  # -1 uses all available cores
    block_bootstrap: bool = False
    block_size: int = 10
    use_antithetic: bool = False


@dataclass
class RiskOfRuinResult:
    """Result container for risk-of-ruin calculations."""
    ruin_probability: float
    confidence_interval: Tuple[float, float]
    time_to_ruin_mean: float
    time_to_ruin_std: float
    equity_at_ruin_mean: float
    max_drawdown_at_ruin: float
    simulations_ruined: int
    total_simulations: int


@dataclass
class MonteCarloResult:
    """Result container for Monte Carlo simulations."""
    final_equities: np.ndarray
    max_drawdowns: np.ndarray
    max_equities: np.ndarray
    min_equities: np.ndarray
    returns: np.ndarray
    volatilities: np.ndarray
    sharpe_ratios: np.ndarray
    sortino_ratios: np.ndarray
    calmar_ratios: np.ndarray
    win_rates: np.ndarray
    profit_factors: np.ndarray
    expectancy: np.ndarray
    confidence_intervals: Dict[str, Tuple[float, float]]
    value_at_risk: Dict[str, float]
    conditional_var: Dict[str, float]
    path_statistics: Dict[str, np.ndarray]


class MonteCarloEngine:
    """
    Advanced Monte Carlo simulation engine for trading strategy analysis.
    
    Features:
    - Trade sequence reshuffling with various methods
    - Equity curve simulation with realistic modeling
    - Risk-of-ruin calculation with confidence intervals
    - Bootstrap resampling for robust statistics
    - Worst-case scenario modeling
    - Parallel simulation execution
    """
    
    def __init__(self, config: Optional[MonteCarloConfig] = None):
        """
        Initialize the Monte Carlo engine.
        
        Args:
            config: MonteCarloConfig instance, uses defaults if None
        """
        self.config = config or MonteCarloConfig()
        self._validate_config()
        
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
        
        # Determine number of workers
        if self.config.n_workers == -1:
            self.n_workers = max(1, mp.cpu_count() - 1)
        else:
            self.n_workers = self.config.n_workers
    
    def _validate_config(self):
        """Validate configuration parameters."""
        if self.config.n_simulations < 100:
            warnings.warn("n_simulations < 100 may produce unreliable results")
        if not 0 < self.config.confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1")
    
    def simulate_equity_curve(
        self,
        returns: np.ndarray,
        initial_capital: float = 100000.0,
        method: str = "bootstrap",
        compound: bool = True,
        transaction_costs: float = 0.0,
        slippage: float = 0.0,
        position_sizing: str = "fixed",
        position_size_params: Optional[Dict] = None
    ) -> MonteCarloResult:
        """
        Simulate equity curves using Monte Carlo methods.
        
        Args:
            returns: Historical returns array
            initial_capital: Starting capital
            method: Simulation method ('bootstrap', 'parametric', 'block_bootstrap', 'circular')
            compound: Whether to compound returns
            transaction_costs: Transaction cost per trade (as fraction)
            slippage: Slippage per trade (as fraction)
            position_sizing: Position sizing method ('fixed', 'kelly', 'optimal_f', 'percent_risk')
            position_size_params: Parameters for position sizing
            
        Returns:
            MonteCarloResult with simulation statistics
        """
        returns = np.array(returns).flatten()
        n_returns = len(returns)
        
        # Generate all simulation paths
        if method == "bootstrap":
            paths = self._bootstrap_paths(returns, n_returns)
        elif method == "block_bootstrap":
            paths = self._block_bootstrap_paths(returns, n_returns)
        elif method == "parametric":
            paths = self._parametric_paths(returns, n_returns)
        elif method == "circular":
            paths = self._circular_bootstrap_paths(returns, n_returns)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Apply position sizing if specified
        if position_sizing != "fixed":
            paths = self._apply_position_sizing(
                paths, position_sizing, position_size_params or {}
            )
        
        # Apply costs
        paths = self._apply_costs(paths, transaction_costs, slippage)
        
        # Calculate equity curves
        if compound:
            equity_curves = initial_capital * np.cumprod(1 + paths, axis=1)
        else:
            equity_curves = initial_capital + np.cumsum(paths * initial_capital, axis=1)
        
        # Calculate statistics
        return self._calculate_statistics(equity_curves, paths)
    
    def _bootstrap_paths(self, returns: np.ndarray, n_steps: int) -> np.ndarray:
        """Generate paths using standard bootstrap resampling."""
        n_sim = self.config.n_simulations
        paths = np.random.choice(returns, size=(n_sim, n_steps), replace=True)
        return paths
    
    def _block_bootstrap_paths(self, returns: np.ndarray, n_steps: int) -> np.ndarray:
        """Generate paths using block bootstrap for serial correlation."""
        n_sim = self.config.n_simulations
        block_size = self.config.block_size
        n_returns = len(returns)
        
        n_blocks = int(np.ceil(n_steps / block_size))
        paths = np.zeros((n_sim, n_steps))
        
        for i in range(n_sim):
            for j in range(n_blocks):
                start_idx = np.random.randint(0, n_returns - block_size + 1)
                block = returns[start_idx:start_idx + block_size]
                start_pos = j * block_size
                end_pos = min(start_pos + block_size, n_steps)
                paths[i, start_pos:end_pos] = block[:end_pos - start_pos]
        
        return paths
    
    def _circular_bootstrap_paths(self, returns: np.ndarray, n_steps: int) -> np.ndarray:
        """Generate paths using circular bootstrap."""
        n_sim = self.config.n_simulations
        n_returns = len(returns)
        
        # Create circular buffer
        circular_returns = np.tile(returns, 2)
        
        paths = np.zeros((n_sim, n_steps))
        for i in range(n_sim):
            start_idx = np.random.randint(0, n_returns)
            for j in range(n_steps):
                paths[i, j] = circular_returns[start_idx + j]
        
        return paths
    
    def _parametric_paths(self, returns: np.ndarray, n_steps: int) -> np.ndarray:
        """Generate paths using parametric simulation (fitted distribution)."""
        n_sim = self.config.n_simulations
        
        # Fit distribution - try multiple distributions and select best
        distributions = [
            stats.norm, stats.t, stats.laplace, 
            stats.skewnorm, stats.johnsonsu
        ]
        
        best_dist = None
        best_ks_stat = np.inf
        
        for dist in distributions:
            try:
                params = dist.fit(returns)
                ks_stat, _ = stats.kstest(returns, lambda x: dist.cdf(x, *params))
                if ks_stat < best_ks_stat:
                    best_ks_stat = ks_stat
                    best_dist = (dist, params)
            except:
                continue
        
        if best_dist is None:
            # Fallback to normal distribution
            mu, std = np.mean(returns), np.std(returns)
            paths = np.random.normal(mu, std, size=(n_sim, n_steps))
        else:
            dist, params = best_dist
            paths = dist.rvs(*params, size=(n_sim, n_steps))
        
        return paths
    
    def _apply_position_sizing(
        self,
        paths: np.ndarray,
        method: str,
        params: Dict
    ) -> np.ndarray:
        """Apply position sizing to return paths."""
        if method == "kelly":
            # Kelly criterion: f* = (bp - q) / b
            win_rate = params.get("win_rate", 0.5)
            avg_win = params.get("avg_win", 0.02)
            avg_loss = params.get("avg_loss", 0.01)
            b = avg_win / avg_loss
            p = win_rate
            q = 1 - p
            kelly_f = (b * p - q) / b
            kelly_f = max(0, min(kelly_f, 1))  # Constrain to [0, 1]
            return paths * kelly_f
            
        elif method == "optimal_f":
            # Ralph Vince's Optimal f
            # Simplified implementation
            max_loss = params.get("max_loss", abs(np.min(paths)))
            optimal_f = params.get("optimal_f", 0.25)
            return paths * optimal_f / max_loss if max_loss > 0 else paths
            
        elif method == "percent_risk":
            # Fixed fractional position sizing
            risk_per_trade = params.get("risk_per_trade", 0.01)
            stop_loss = params.get("stop_loss", 0.02)
            position_frac = risk_per_trade / stop_loss if stop_loss > 0 else risk_per_trade
            return paths * position_frac
            
        return paths
    
    def _apply_costs(
        self,
        paths: np.ndarray,
        transaction_costs: float,
        slippage: float
    ) -> np.ndarray:
        """Apply transaction costs and slippage to returns."""
        total_cost = transaction_costs + slippage
        if total_cost > 0:
            # Assume each return represents a trade
            paths = paths - total_cost * np.sign(paths)
        return paths
    
    def _calculate_statistics(
        self,
        equity_curves: np.ndarray,
        returns: np.ndarray
    ) -> MonteCarloResult:
        """Calculate comprehensive statistics from simulation results."""
        n_sim = equity_curves.shape[0]
        
        # Final metrics
        final_equities = equity_curves[:, -1]
        
        # Drawdown calculations
        running_max = np.maximum.accumulate(equity_curves, axis=1)
        drawdowns = (equity_curves - running_max) / running_max
        max_drawdowns = np.min(drawdowns, axis=1)
        
        # Max/min equities
        max_equities = np.max(equity_curves, axis=1)
        min_equities = np.min(equity_curves, axis=1)
        
        # Return statistics
        total_returns = (final_equities - equity_curves[:, 0]) / equity_curves[:, 0]
        
        # Annualized metrics (assuming daily returns)
        n_periods = returns.shape[1]
        annualization_factor = np.sqrt(252) if n_periods > 252 else 1
        
        # Volatility
        volatilities = np.std(returns, axis=1) * annualization_factor
        
        # Sharpe ratio (assuming risk-free rate of 0 for simplicity)
        mean_returns = np.mean(returns, axis=1) * 252
        sharpe_ratios = np.where(
            volatilities > 0,
            mean_returns / volatilities,
            0
        )
        
        # Sortino ratio (downside deviation)
        downside_returns = np.where(returns < 0, returns, 0)
        downside_std = np.std(downside_returns, axis=1) * annualization_factor
        sortino_ratios = np.where(
            downside_std > 0,
            mean_returns / downside_std,
            0
        )
        
        # Calmar ratio
        calmar_ratios = np.where(
            max_drawdowns < 0,
            mean_returns / abs(max_drawdowns),
            0
        )
        
        # Win rate
        win_rates = np.mean(returns > 0, axis=1)
        
        # Profit factor
        gross_profits = np.sum(np.where(returns > 0, returns, 0), axis=1)
        gross_losses = np.abs(np.sum(np.where(returns < 0, returns, 0), axis=1))
        profit_factors = np.where(
            gross_losses > 0,
            gross_profits / gross_losses,
            np.inf
        )
        
        # Expectancy
        avg_win = np.mean(np.where(returns > 0, returns, np.nan), axis=1)
        avg_loss = np.mean(np.where(returns < 0, returns, np.nan), axis=1)
        avg_win = np.nan_to_num(avg_win, nan=0)
        avg_loss = np.nan_to_num(avg_loss, nan=0)
        expectancy = win_rates * avg_win - (1 - win_rates) * np.abs(avg_loss)
        
        # Confidence intervals
        alpha = 1 - self.config.confidence_level
        confidence_intervals = {
            "final_equity": self._confidence_interval(final_equities, alpha),
            "max_drawdown": self._confidence_interval(max_drawdowns, alpha),
            "total_return": self._confidence_interval(total_returns, alpha),
            "sharpe_ratio": self._confidence_interval(sharpe_ratios, alpha),
            "volatility": self._confidence_interval(volatilities, alpha)
        }
        
        # Value at Risk and Conditional VaR
        var_levels = [0.95, 0.99]
        value_at_risk = {}
        conditional_var = {}
        
        for level in var_levels:
            var_key = f"var_{int(level*100)}"
            cvar_key = f"cvar_{int(level*100)}"
            value_at_risk[var_key] = np.percentile(total_returns, (1 - level) * 100)
            conditional_var[cvar_key] = np.mean(
                total_returns[total_returns <= value_at_risk[var_key]]
            ) if np.any(total_returns <= value_at_risk[var_key]) else value_at_risk[var_key]
        
        # Path statistics over time
        path_statistics = {
            "mean_equity": np.mean(equity_curves, axis=0),
            "std_equity": np.std(equity_curves, axis=0),
            "percentile_5": np.percentile(equity_curves, 5, axis=0),
            "percentile_25": np.percentile(equity_curves, 25, axis=0),
            "percentile_50": np.percentile(equity_curves, 50, axis=0),
            "percentile_75": np.percentile(equity_curves, 75, axis=0),
            "percentile_95": np.percentile(equity_curves, 95, axis=0),
            "mean_drawdown": np.mean(drawdowns, axis=0),
            "max_drawdown_path": np.percentile(drawdowns, 5, axis=0)
        }
        
        return MonteCarloResult(
            final_equities=final_equities,
            max_drawdowns=max_drawdowns,
            max_equities=max_equities,
            min_equities=min_equities,
            returns=total_returns,
            volatilities=volatilities,
            sharpe_ratios=sharpe_ratios,
            sortino_ratios=sortino_ratios,
            calmar_ratios=calmar_ratios,
            win_rates=win_rates,
            profit_factors=profit_factors,
            expectancy=expectancy,
            confidence_intervals=confidence_intervals,
            value_at_risk=value_at_risk,
            conditional_var=conditional_var,
            path_statistics=path_statistics
        )
    
    def _confidence_interval(
        self,
        data: np.ndarray,
        alpha: float
    ) -> Tuple[float, float]:
        """Calculate confidence interval using bootstrap or normal approximation."""
        if len(data) < 1000:
            # Use bootstrap for small samples
            return self._bootstrap_ci(data, alpha)
        else:
            # Use normal approximation for large samples
            mean = np.mean(data)
            std_err = stats.sem(data)
            ci = stats.t.interval(1 - alpha, len(data) - 1, loc=mean, scale=std_err)
            return ci
    
    def _bootstrap_ci(
        self,
        data: np.ndarray,
        alpha: float,
        n_bootstrap: int = 1000
    ) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval."""
        bootstrap_means = []
        n = len(data)
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))
        
        lower = np.percentile(bootstrap_means, alpha / 2 * 100)
        upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)
        
        return (lower, upper)
    
    def calculate_risk_of_ruin(
        self,
        returns: np.ndarray,
        initial_capital: float = 100000.0,
        ruin_threshold: float = 0.5,
        max_trades: int = 1000,
        method: str = "monte_carlo"
    ) -> RiskOfRuinResult:
        """
        Calculate risk of ruin using various methods.
        
        Args:
            returns: Historical trade returns
            initial_capital: Starting capital
            ruin_threshold: Fraction of capital at which ruin occurs
            max_trades: Maximum number of trades to simulate
            method: Method for calculation ('monte_carlo', 'analytical', 'gambler')
            
        Returns:
            RiskOfRuinResult with detailed statistics
        """
        returns = np.array(returns).flatten()
        ruin_level = initial_capital * ruin_threshold
        
        if method == "monte_carlo":
            return self._ror_monte_carlo(returns, initial_capital, ruin_level, max_trades)
        elif method == "analytical":
            return self._ror_analytical(returns, initial_capital, ruin_level)
        elif method == "gambler":
            return self._ror_gambler(returns, initial_capital, ruin_level)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _ror_monte_carlo(
        self,
        returns: np.ndarray,
        initial_capital: float,
        ruin_level: float,
        max_trades: int
    ) -> RiskOfRuinResult:
        """Monte Carlo simulation for risk of ruin."""
        n_sim = self.config.n_simulations
        
        ruined = np.zeros(n_sim, dtype=bool)
        time_to_ruin = np.full(n_sim, max_trades)
        equity_at_ruin = np.zeros(n_sim)
        max_dd_at_ruin = np.zeros(n_sim)
        
        for i in range(n_sim):
            equity = initial_capital
            max_equity = equity
            
            for t in range(max_trades):
                ret = np.random.choice(returns)
                equity *= (1 + ret)
                max_equity = max(max_equity, equity)
                
                if equity <= ruin_level:
                    ruined[i] = True
                    time_to_ruin[i] = t
                    equity_at_ruin[i] = equity
                    max_dd_at_ruin[i] = (equity - max_equity) / max_equity
                    break
        
        ruin_prob = np.mean(ruined)
        
        # Confidence interval for ruin probability
        n_ruined = np.sum(ruined)
        ci = stats.proportion_confint(
            n_ruined, n_sim, alpha=1-self.config.confidence_level, method='wilson'
        )
        
        return RiskOfRuinResult(
            ruin_probability=ruin_prob,
            confidence_interval=ci,
            time_to_ruin_mean=np.mean(time_to_ruin[ruined]) if np.any(ruined) else max_trades,
            time_to_ruin_std=np.std(time_to_ruin[ruined]) if np.any(ruined) else 0,
            equity_at_ruin_mean=np.mean(equity_at_ruin[ruined]) if np.any(ruined) else 0,
            max_drawdown_at_ruin=np.mean(max_dd_at_ruin[ruined]) if np.any(ruined) else 0,
            simulations_ruined=int(n_ruined),
            total_simulations=n_sim
        )
    
    def _ror_analytical(
        self,
        returns: np.ndarray,
        initial_capital: float,
        ruin_level: float
    ) -> RiskOfRuinResult:
        """Analytical approximation for risk of ruin (simplified)."""
        mu = np.mean(returns)
        sigma = np.std(returns)
        
        if sigma == 0:
            # Deterministic case
            ruin_prob = 1.0 if mu < 0 and initial_capital > 0 else 0.0
        else:
            # Using approximation for geometric Brownian motion
            # ROR ≈ exp(-2 * mu * ln(S0/S_ruin) / sigma^2)
            if mu > 0:
                ruin_prob = (ruin_level / initial_capital) ** (2 * mu / (sigma ** 2))
            else:
                ruin_prob = 1.0
        
        ruin_prob = min(1.0, max(0.0, ruin_prob))
        
        # Approximate confidence interval
        se = np.sqrt(ruin_prob * (1 - ruin_prob) / self.config.n_simulations)
        ci = (
            max(0, ruin_prob - 1.96 * se),
            min(1, ruin_prob + 1.96 * se)
        )
        
        return RiskOfRuinResult(
            ruin_probability=ruin_prob,
            confidence_interval=ci,
            time_to_ruin_mean=np.nan,
            time_to_ruin_std=np.nan,
            equity_at_ruin_mean=ruin_level,
            max_drawdown_at_ruin=(ruin_level - initial_capital) / initial_capital,
            simulations_ruined=int(ruin_prob * self.config.n_simulations),
            total_simulations=self.config.n_simulations
        )
    
    def _ror_gambler(
        self,
        returns: np.ndarray,
        initial_capital: float,
        ruin_level: float
    ) -> RiskOfRuinResult:
        """Gambler's ruin approximation for risk of ruin."""
        wins = np.sum(returns > 0)
        losses = np.sum(returns <= 0)
        total = wins + losses
        
        if total == 0:
            return RiskOfRuinResult(
                ruin_probability=0.5,
                confidence_interval=(0, 1),
                time_to_ruin_mean=np.nan,
                time_to_ruin_std=np.nan,
                equity_at_ruin_mean=ruin_level,
                max_drawdown_at_ruin=0,
                simulations_ruined=0,
                total_simulations=self.config.n_simulations
            )
        
        p = wins / total
        q = 1 - p
        
        # Simplified gambler's ruin
        if p == q:
            ruin_prob = 1 - (initial_capital - ruin_level) / initial_capital
        elif p > q:
            ruin_prob = ((q/p) ** (initial_capital / (initial_capital - ruin_level)) - (q/p) ** (initial_capital / ruin_level)) / (1 - (q/p) ** (initial_capital / ruin_level))
            ruin_prob = max(0, min(1, 1 - ruin_prob))
        else:
            ruin_prob = 1.0
        
        ci = (max(0, ruin_prob - 0.1), min(1, ruin_prob + 0.1))
        
        return RiskOfRuinResult(
            ruin_probability=ruin_prob,
            confidence_interval=ci,
            time_to_ruin_mean=np.nan,
            time_to_ruin_std=np.nan,
            equity_at_ruin_mean=ruin_level,
            max_drawdown_at_ruin=(ruin_level - initial_capital) / initial_capital,
            simulations_ruined=int(ruin_prob * self.config.n_simulations),
            total_simulations=self.config.n_simulations
        )
    
    def worst_case_scenario(
        self,
        returns: np.ndarray,
        initial_capital: float = 100000.0,
        scenario_type: str = "stress_test",
        stress_factors: Optional[Dict] = None
    ) -> Dict:
        """
        Generate worst-case scenarios for stress testing.
        
        Args:
            returns: Historical returns
            initial_capital: Starting capital
            scenario_type: Type of stress test
            stress_factors: Additional stress parameters
            
        Returns:
            Dictionary with scenario results
        """
        returns = np.array(returns)
        stress_factors = stress_factors or {}
        
        scenarios = {}
        
        if scenario_type == "stress_test" or scenario_type == "all":
            # Historical stress - use worst historical periods
            scenarios["historical_worst"] = self._historical_stress(returns, initial_capital)
            
            # Volatility stress - increase volatility
            vol_mult = stress_factors.get("volatility_multiplier", 2.0)
            scenarios["volatility_stress"] = self._volatility_stress(
                returns, initial_capital, vol_mult
            )
            
            # Correlation stress - assume all assets move together
            scenarios["correlation_stress"] = self._correlation_stress(
                returns, initial_capital
            )
            
            # Tail risk stress - extreme events
            scenarios["tail_risk_stress"] = self._tail_risk_stress(
                returns, initial_capital, stress_factors.get("tail_percentile", 0.01)
            )
        
        if scenario_type == "monte_carlo_worst" or scenario_type == "all":
            # Find worst outcomes from Monte Carlo
            result = self.simulate_equity_curve(
                returns, initial_capital, n_simulations=10000
            )
            worst_idx = np.argmin(result.final_equities)
            scenarios["monte_carlo_worst"] = {
                "final_equity": result.final_equities[worst_idx],
                "max_drawdown": result.max_drawdowns[worst_idx],
                "return": result.returns[worst_idx]
            }
        
        return scenarios
    
    def _historical_stress(
        self,
        returns: np.ndarray,
        initial_capital: float
    ) -> Dict:
        """Apply historical worst-case scenario."""
        sorted_returns = np.sort(returns)
        worst_5pct = sorted_returns[:int(len(returns) * 0.05)]
        
        # Simulate using only worst returns
        equity = initial_capital
        max_equity = equity
        max_dd = 0
        
        for ret in worst_5pct:
            equity *= (1 + ret)
            max_equity = max(max_equity, equity)
            dd = (equity - max_equity) / max_equity
            max_dd = min(max_dd, dd)
        
        return {
            "final_equity": equity,
            "max_drawdown": max_dd,
            "return": (equity - initial_capital) / initial_capital,
            "n_periods": len(worst_5pct)
        }
    
    def _volatility_stress(
        self,
        returns: np.ndarray,
        initial_capital: float,
        multiplier: float
    ) -> Dict:
        """Stress test with increased volatility."""
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        
        stressed_returns = np.random.normal(
            mean_ret, std_ret * multiplier, size=len(returns)
        )
        
        equity = initial_capital * np.cumprod(1 + stressed_returns)
        max_dd = np.min((equity - np.maximum.accumulate(equity)) / np.maximum.accumulate(equity))
        
        return {
            "final_equity": equity[-1],
            "max_drawdown": max_dd,
            "return": (equity[-1] - initial_capital) / initial_capital,
            "volatility_multiplier": multiplier
        }
    
    def _correlation_stress(
        self,
        returns: np.ndarray,
        initial_capital: float
    ) -> Dict:
        """Stress test assuming perfect correlation."""
        # Simplified - assume all returns become more correlated
        # In practice, this would require multi-asset returns
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        
        # Generate perfectly correlated returns
        factor = np.random.normal(0, 1, size=len(returns))
        stressed_returns = mean_ret + std_ret * factor
        
        equity = initial_capital * np.cumprod(1 + stressed_returns)
        max_dd = np.min((equity - np.maximum.accumulate(equity)) / np.maximum.accumulate(equity))
        
        return {
            "final_equity": equity[-1],
            "max_drawdown": max_dd,
            "return": (equity[-1] - initial_capital) / initial_capital
        }
    
    def _tail_risk_stress(
        self,
        returns: np.ndarray,
        initial_capital: float,
        tail_percentile: float
    ) -> Dict:
        """Stress test focusing on tail risk."""
        var_threshold = np.percentile(returns, tail_percentile * 100)
        tail_returns = returns[returns <= var_threshold]
        
        if len(tail_returns) == 0:
            tail_returns = np.array([var_threshold])
        
        # Simulate with increased tail events
        n_tail = int(len(returns) * tail_percentile * 2)
        stressed_returns = np.concatenate([
            np.random.choice(tail_returns, size=n_tail),
            np.random.choice(returns, size=len(returns) - n_tail)
        ])
        np.random.shuffle(stressed_returns)
        
        equity = initial_capital * np.cumprod(1 + stressed_returns)
        max_dd = np.min((equity - np.maximum.accumulate(equity)) / np.maximum.accumulate(equity))
        
        return {
            "final_equity": equity[-1],
            "max_drawdown": max_dd,
            "return": (equity[-1] - initial_capital) / initial_capital,
            "tail_percentile": tail_percentile
        }
    
    def reshuffle_trades(
        self,
        trades: np.ndarray,
        method: str = "bootstrap",
        preserve_clusters: bool = False
    ) -> np.ndarray:
        """
        Reshuffle trade sequence for Monte Carlo analysis.
        
        Args:
            trades: Array of trade returns
            method: Reshuffling method
            preserve_clusters: Whether to preserve trade clusters
            
        Returns:
            Reshuffled trades
        """
        trades = np.array(trades)
        
        if method == "bootstrap":
            return np.random.choice(trades, size=len(trades), replace=True)
        
        elif method == "permutation":
            return np.random.permutation(trades)
        
        elif method == "circular":
            shift = np.random.randint(0, len(trades))
            return np.roll(trades, shift)
        
        elif method == "stationary_bootstrap":
            return self._stationary_bootstrap(trades)
        
        elif method == "markov":
            return self._markov_resample(trades)
        
        else:
            raise ValueError(f"Unknown reshuffle method: {method}")
    
    def _stationary_bootstrap(self, trades: np.ndarray) -> np.ndarray:
        """Stationary bootstrap for time series."""
        n = len(trades)
        p = 1 / self.config.block_size  # Probability of new block
        
        result = np.zeros(n)
        idx = np.random.randint(0, n)
        result[0] = trades[idx]
        
        for i in range(1, n):
            if np.random.random() < p:
                idx = np.random.randint(0, n)
            else:
                idx = (idx + 1) % n
            result[i] = trades[idx]
        
        return result
    
    def _markov_resample(self, trades: np.ndarray) -> np.ndarray:
        """Resample using Markov chain transition probabilities."""
        n = len(trades)
        
        # Discretize returns into states
        n_states = 5
        states = np.digitize(trades, np.percentile(trades, np.linspace(0, 100, n_states + 1)))
        
        # Build transition matrix
        trans_matrix = np.zeros((n_states + 1, n_states + 1))
        for i in range(n - 1):
            trans_matrix[states[i], states[i + 1]] += 1
        
        # Normalize
        row_sums = trans_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        trans_matrix = trans_matrix / row_sums
        
        # Generate new sequence
        current_state = states[0]
        new_states = [current_state]
        
        for _ in range(n - 1):
            current_state = np.random.choice(
                n_states + 1, p=trans_matrix[current_state]
            )
            new_states.append(current_state)
        
        # Map back to returns
        state_returns = [trades[states == s].mean() if np.any(states == s) else 0 
                        for s in range(n_states + 1)]
        
        return np.array([state_returns[s] for s in new_states])


class BootstrapAnalyzer:
    """
    Bootstrap resampling analysis for robust statistical inference.
    """
    
    def __init__(self, n_bootstrap: int = 10000, random_seed: Optional[int] = None):
        self.n_bootstrap = n_bootstrap
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def bootstrap_statistic(
        self,
        data: np.ndarray,
        statistic_func: Callable,
        confidence_level: float = 0.95
    ) -> Dict:
        """
        Calculate bootstrap distribution of a statistic.
        
        Args:
            data: Input data
            statistic_func: Function to compute statistic
            confidence_level: Confidence level for intervals
            
        Returns:
            Dictionary with bootstrap results
        """
        data = np.array(data)
        n = len(data)
        
        bootstrap_stats = []
        for _ in range(self.n_bootstrap):
            sample = np.random.choice(data, size=n, replace=True)
            bootstrap_stats.append(statistic_func(sample))
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        alpha = 1 - confidence_level
        
        return {
            "original": statistic_func(data),
            "bootstrap_mean": np.mean(bootstrap_stats),
            "bootstrap_std": np.std(bootstrap_stats),
            "confidence_interval": (
                np.percentile(bootstrap_stats, alpha / 2 * 100),
                np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)
            ),
            "bias": np.mean(bootstrap_stats) - statistic_func(data),
            "standard_error": np.std(bootstrap_stats),
            "distribution": bootstrap_stats
        }
    
    def bootstrap_comparison(
        self,
        data1: np.ndarray,
        data2: np.ndarray,
        statistic_func: Callable = np.mean
    ) -> Dict:
        """
        Compare two samples using bootstrap.
        
        Args:
            data1: First sample
            data2: Second sample
            statistic_func: Statistic to compare
            
        Returns:
            Dictionary with comparison results
        """
        n1, n2 = len(data1), len(data2)
        
        diff_stats = []
        for _ in range(self.n_bootstrap):
            sample1 = np.random.choice(data1, size=n1, replace=True)
            sample2 = np.random.choice(data2, size=n2, replace=True)
            diff_stats.append(statistic_func(sample1) - statistic_func(sample2))
        
        diff_stats = np.array(diff_stats)
        original_diff = statistic_func(data1) - statistic_func(data2)
        
        # Calculate p-value
        p_value = 2 * min(np.mean(diff_stats <= 0), np.mean(diff_stats >= 0))
        
        return {
            "difference": original_diff,
            "confidence_interval": (
                np.percentile(diff_stats, 2.5),
                np.percentile(diff_stats, 97.5)
            ),
            "p_value": p_value,
            "significant": p_value < 0.05,
            "distribution": diff_stats
        }


# Convenience functions for quick analysis
def quick_monte_carlo(
    returns: np.ndarray,
    initial_capital: float = 100000.0,
    n_simulations: int = 10000
) -> Dict:
    """
    Quick Monte Carlo simulation with default parameters.
    
    Args:
        returns: Historical returns
        initial_capital: Starting capital
        n_simulations: Number of simulations
        
    Returns:
        Summary dictionary of results
    """
    engine = MonteCarloEngine(MonteCarloConfig(n_simulations=n_simulations))
    result = engine.simulate_equity_curve(returns, initial_capital)
    
    return {
        "median_final_equity": np.median(result.final_equities),
        "mean_final_equity": np.mean(result.final_equities),
        "worst_case_equity": np.min(result.final_equities),
        "best_case_equity": np.max(result.final_equities),
        "median_max_drawdown": np.median(result.max_drawdowns),
        "worst_drawdown": np.min(result.max_drawdowns),
        "median_sharpe": np.median(result.sharpe_ratios),
        "value_at_risk_95": result.value_at_risk.get("var_95", 0),
        "confidence_intervals": result.confidence_intervals
    }


def quick_risk_of_ruin(
    returns: np.ndarray,
    initial_capital: float = 100000.0,
    ruin_threshold: float = 0.5
) -> Dict:
    """
    Quick risk of ruin calculation.
    
    Args:
        returns: Historical returns
        initial_capital: Starting capital
        ruin_threshold: Ruin threshold
        
    Returns:
        Summary dictionary
    """
    engine = MonteCarloEngine()
    result = engine.calculate_risk_of_ruin(returns, initial_capital, ruin_threshold)
    
    return {
        "risk_of_ruin": result.ruin_probability,
        "confidence_interval": result.confidence_interval,
        "time_to_ruin_mean": result.time_to_ruin_mean,
        "simulations_ruined": result.simulations_ruined
    }


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate sample returns
    sample_returns = np.random.normal(0.001, 0.02, 252)  # Daily returns
    
    # Run Monte Carlo simulation
    engine = MonteCarloEngine(MonteCarloConfig(n_simulations=5000))
    result = engine.simulate_equity_curve(sample_returns, initial_capital=100000)
    
    print("Monte Carlo Simulation Results:")
    print(f"Median Final Equity: ${np.median(result.final_equities):,.2f}")
    print(f"Mean Final Equity: ${np.mean(result.final_equities):,.2f}")
    print(f"Worst Case Equity: ${np.min(result.final_equities):,.2f}")
    print(f"Median Max Drawdown: {np.median(result.max_drawdowns)*100:.2f}%")
    print(f"Median Sharpe Ratio: {np.median(result.sharpe_ratios):.2f}")
    
    # Risk of ruin
    ror = engine.calculate_risk_of_ruin(sample_returns, ruin_threshold=0.5)
    print(f"\nRisk of Ruin (50% threshold): {ror.ruin_probability*100:.2f}%")
    print(f"95% CI: [{ror.confidence_interval[0]*100:.2f}%, {ror.confidence_interval[1]*100:.2f}%]")
