"""
Performance Metrics Module for TradeOS
======================================

Comprehensive performance analytics including:
- Sharpe, Sortino, Calmar ratios
- Alpha, Beta calculations
- Maximum drawdown analysis
- Win rate, profit factor
- Expectancy, SQN (System Quality Number)
- Tail risk metrics
- Rolling statistics

Author: TradeOS Quant Team
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union, Callable
from dataclasses import dataclass
from scipy import stats
from scipy.stats import skew, kurtosis
import warnings


@dataclass
class PerformanceMetrics:
    """Container for comprehensive performance metrics."""
    # Return metrics
    total_return: float
    annualized_return: float
    cagr: float
    
    # Risk metrics
    volatility: float
    downside_deviation: float
    max_drawdown: float
    max_drawdown_duration: int
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    
    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float
    kappa_three: float
    
    # Benchmark-relative
    alpha: float
    beta: float
    information_ratio: float
    treynor_ratio: float
    tracking_error: float
    
    # Trade statistics
    win_rate: float
    profit_factor: float
    payoff_ratio: float
    expectancy: float
    sqn: float
    
    # Distribution metrics
    skewness: float
    kurtosis: float
    jarque_bera: float
    jarque_bera_pvalue: float
    
    # Drawdown metrics
    avg_drawdown: float
    avg_drawdown_duration: float
    recovery_factor: float
    ulcer_index: float
    
    # Time-based
    monthly_returns: Dict[str, float]
    best_month: float
    worst_month: float
    positive_months_pct: float
    
    # Additional
    gain_to_pain_ratio: float
    common_sense_ratio: float
    tail_ratio: float


class PerformanceAnalyzer:
    """
    Comprehensive performance analyzer for trading strategies.
    
    Calculates all major performance metrics used in quantitative finance.
    """
    
    def __init__(
        self,
        risk_free_rate: float = 0.0,
        frequency: str = 'daily',
        annualize: bool = True
    ):
        """
        Initialize the performance analyzer.
        
        Args:
            risk_free_rate: Annual risk-free rate
            frequency: Data frequency ('daily', 'weekly', 'monthly')
            annualize: Whether to annualize metrics
        """
        self.risk_free_rate = risk_free_rate
        self.frequency = frequency
        self.annualize = annualize
        
        # Annualization factors
        self.periods_per_year = {
            'daily': 252,
            'weekly': 52,
            'monthly': 12,
            'quarterly': 4,
            'yearly': 1
        }
        
        self.n_periods = self.periods_per_year.get(frequency, 252)
    
    def calculate_all_metrics(
        self,
        returns: np.ndarray,
        benchmark_returns: Optional[np.ndarray] = None,
        trades: Optional[List[Dict]] = None
    ) -> PerformanceMetrics:
        """
        Calculate all performance metrics.
        
        Args:
            returns: Strategy returns series
            benchmark_returns: Optional benchmark returns
            trades: Optional list of trade dictionaries
            
        Returns:
            PerformanceMetrics object
        """
        returns = np.array(returns).flatten()
        
        # Basic return metrics
        total_return = self._total_return(returns)
        annualized_return = self._annualized_return(returns)
        cagr = self._cagr(returns)
        
        # Volatility
        volatility = self._volatility(returns)
        downside_deviation = self._downside_deviation(returns)
        
        # Drawdown
        max_dd, max_dd_duration, dd_series = self._drawdown_analysis(returns)
        avg_dd = np.mean(dd_series[dd_series < 0]) if np.any(dd_series < 0) else 0
        avg_dd_duration = self._average_drawdown_duration(dd_series)
        
        # VaR and CVaR
        var_95, var_99 = self._value_at_risk(returns)
        cvar_95, cvar_99 = self._conditional_var(returns)
        
        # Risk-adjusted ratios
        sharpe = self._sharpe_ratio(returns, volatility)
        sortino = self._sortino_ratio(returns, downside_deviation)
        calmar = self._calmar_ratio(returns, max_dd)
        omega = self._omega_ratio(returns)
        kappa = self._kappa_ratio(returns, 3)
        
        # Benchmark-relative
        if benchmark_returns is not None:
            benchmark_returns = np.array(benchmark_returns).flatten()
            alpha, beta = self._alpha_beta(returns, benchmark_returns)
            info_ratio = self._information_ratio(returns, benchmark_returns)
            treynor = self._treynor_ratio(returns, beta)
            tracking_err = self._tracking_error(returns, benchmark_returns)
        else:
            alpha, beta = 0.0, 1.0
            info_ratio = 0.0
            treynor = 0.0
            tracking_err = 0.0
        
        # Trade statistics
        if trades is not None:
            win_rate, profit_factor, payoff_ratio, expectancy, sqn = self._trade_statistics(trades)
        else:
            win_rate = self._win_rate(returns)
            profit_factor = self._profit_factor(returns)
            payoff_ratio = self._payoff_ratio(returns)
            expectancy = self._expectancy(returns)
            sqn = self._sqn(returns)
        
        # Distribution metrics
        skew = self._skewness(returns)
        kurt = self._kurtosis(returns)
        jb_stat, jb_pvalue = self._jarque_bera(returns)
        
        # Recovery factor
        recovery = self._recovery_factor(total_return, max_dd)
        
        # Ulcer index
        ulcer = self._ulcer_index(dd_series)
        
        # Monthly analysis
        monthly_rets, best_month, worst_month, pos_months = self._monthly_analysis(returns)
        
        # Additional ratios
        gain_to_pain = self._gain_to_pain_ratio(returns)
        common_sense = self._common_sense_ratio(returns)
        tail = self._tail_ratio(returns)
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            cagr=cagr,
            volatility=volatility,
            downside_deviation=downside_deviation,
            max_drawdown=max_dd,
            max_drawdown_duration=max_dd_duration,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            omega_ratio=omega,
            kappa_three=kappa,
            alpha=alpha,
            beta=beta,
            information_ratio=info_ratio,
            treynor_ratio=treynor,
            tracking_error=tracking_err,
            win_rate=win_rate,
            profit_factor=profit_factor,
            payoff_ratio=payoff_ratio,
            expectancy=expectancy,
            sqn=sqn,
            skewness=skew,
            kurtosis=kurt,
            jarque_bera=jb_stat,
            jarque_bera_pvalue=jb_pvalue,
            avg_drawdown=avg_dd,
            avg_drawdown_duration=avg_dd_duration,
            recovery_factor=recovery,
            ulcer_index=ulcer,
            monthly_returns=monthly_rets,
            best_month=best_month,
            worst_month=worst_month,
            positive_months_pct=pos_months,
            gain_to_pain_ratio=gain_to_pain,
            common_sense_ratio=common_sense,
            tail_ratio=tail
        )
    
    # Return metrics
    def _total_return(self, returns: np.ndarray) -> float:
        """Calculate total return."""
        return np.prod(1 + returns) - 1
    
    def _annualized_return(self, returns: np.ndarray) -> float:
        """Calculate annualized return."""
        total_ret = self._total_return(returns)
        n_years = len(returns) / self.n_periods
        if self.annualize:
            return (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
        return np.mean(returns) * self.n_periods
    
    def _cagr(self, returns: np.ndarray) -> float:
        """Calculate Compound Annual Growth Rate."""
        return self._annualized_return(returns)
    
    # Risk metrics
    def _volatility(self, returns: np.ndarray) -> float:
        """Calculate annualized volatility."""
        vol = np.std(returns, ddof=1)
        if self.annualize:
            return vol * np.sqrt(self.n_periods)
        return vol
    
    def _downside_deviation(self, returns: np.ndarray, target: float = 0) -> float:
        """Calculate downside deviation (Sortino denominator)."""
        downside = returns[returns < target]
        if len(downside) == 0:
            return 0
        dd = np.std(downside, ddof=1)
        if self.annualize:
            return dd * np.sqrt(self.n_periods)
        return dd
    
    def _drawdown_analysis(self, returns: np.ndarray) -> Tuple[float, int, np.ndarray]:
        """Calculate drawdown series and statistics."""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        max_drawdown = np.min(drawdown)
        
        # Find max drawdown duration
        is_in_drawdown = drawdown < 0
        max_duration = 0
        current_duration = 0
        
        for in_dd in is_in_drawdown:
            if in_dd:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
        
        return max_drawdown, max_duration, drawdown
    
    def _average_drawdown_duration(self, drawdown: np.ndarray) -> float:
        """Calculate average drawdown duration."""
        is_in_drawdown = drawdown < 0
        durations = []
        current_duration = 0
        
        for in_dd in is_in_drawdown:
            if in_dd:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                current_duration = 0
        
        if current_duration > 0:
            durations.append(current_duration)
        
        return np.mean(durations) if durations else 0
    
    def _value_at_risk(self, returns: np.ndarray) -> Tuple[float, float]:
        """Calculate Value at Risk."""
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        return var_95, var_99
    
    def _conditional_var(self, returns: np.ndarray) -> Tuple[float, float]:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        cvar_95 = np.mean(returns[returns <= var_95]) if np.any(returns <= var_95) else var_95
        cvar_99 = np.mean(returns[returns <= var_99]) if np.any(returns <= var_99) else var_99
        
        return cvar_95, cvar_99
    
    # Risk-adjusted returns
    def _sharpe_ratio(self, returns: np.ndarray, volatility: Optional[float] = None) -> float:
        """Calculate Sharpe ratio."""
        excess_return = np.mean(returns) * self.n_periods - self.risk_free_rate
        if volatility is None:
            volatility = self._volatility(returns)
        return excess_return / volatility if volatility > 0 else 0
    
    def _sortino_ratio(self, returns: np.ndarray, downside_dev: Optional[float] = None) -> float:
        """Calculate Sortino ratio."""
        excess_return = np.mean(returns) * self.n_periods - self.risk_free_rate
        if downside_dev is None:
            downside_dev = self._downside_deviation(returns)
        return excess_return / downside_dev if downside_dev > 0 else 0
    
    def _calmar_ratio(self, returns: np.ndarray, max_dd: Optional[float] = None) -> float:
        """Calculate Calmar ratio."""
        annual_ret = self._annualized_return(returns)
        if max_dd is None:
            max_dd, _, _ = self._drawdown_analysis(returns)
        return annual_ret / abs(max_dd) if max_dd < 0 else 0
    
    def _omega_ratio(self, returns: np.ndarray, threshold: float = 0) -> float:
        """Calculate Omega ratio."""
        excess = returns - threshold
        gains = np.sum(excess[excess > 0])
        losses = abs(np.sum(excess[excess < 0]))
        return gains / losses if losses > 0 else np.inf
    
    def _kappa_ratio(self, returns: np.ndarray, n: int = 3) -> float:
        """Calculate Kappa-n ratio."""
        excess_return = np.mean(returns) * self.n_periods - self.risk_free_rate
        
        # Lower partial moment of order n
        threshold = 0
        lpm = np.mean(np.maximum(threshold - returns, 0) ** n)
        lpm = lpm ** (1 / n)
        
        if self.annualize:
            lpm *= np.sqrt(self.n_periods)
        
        return excess_return / lpm if lpm > 0 else 0
    
    # Benchmark-relative
    def _alpha_beta(self, returns: np.ndarray, benchmark: np.ndarray) -> Tuple[float, float]:
        """Calculate alpha and beta."""
        # Align lengths
        min_len = min(len(returns), len(benchmark))
        returns = returns[:min_len]
        benchmark = benchmark[:min_len]
        
        # Regression
        beta, alpha, _, _, _ = stats.linregress(benchmark, returns)
        
        # Annualize alpha
        if self.annualize:
            alpha = (1 + alpha) ** self.n_periods - 1
        else:
            alpha = alpha * self.n_periods
        
        return alpha, beta
    
    def _information_ratio(self, returns: np.ndarray, benchmark: np.ndarray) -> float:
        """Calculate Information ratio."""
        min_len = min(len(returns), len(benchmark))
        returns = returns[:min_len]
        benchmark = benchmark[:min_len]
        
        active_return = returns - benchmark
        tracking_error = np.std(active_return, ddof=1)
        
        if self.annualize:
            tracking_error *= np.sqrt(self.n_periods)
        
        return np.mean(active_return) * self.n_periods / tracking_error if tracking_error > 0 else 0
    
    def _treynor_ratio(self, returns: np.ndarray, beta: float) -> float:
        """Calculate Treynor ratio."""
        excess_return = np.mean(returns) * self.n_periods - self.risk_free_rate
        return excess_return / beta if beta != 0 else 0
    
    def _tracking_error(self, returns: np.ndarray, benchmark: np.ndarray) -> float:
        """Calculate tracking error."""
        min_len = min(len(returns), len(benchmark))
        active_return = returns[:min_len] - benchmark[:min_len]
        te = np.std(active_return, ddof=1)
        if self.annualize:
            te *= np.sqrt(self.n_periods)
        return te
    
    # Trade statistics
    def _win_rate(self, returns: np.ndarray) -> float:
        """Calculate win rate."""
        return np.mean(returns > 0)
    
    def _profit_factor(self, returns: np.ndarray) -> float:
        """Calculate profit factor."""
        gross_profit = np.sum(returns[returns > 0])
        gross_loss = abs(np.sum(returns[returns < 0]))
        return gross_profit / gross_loss if gross_loss > 0 else np.inf
    
    def _payoff_ratio(self, returns: np.ndarray) -> float:
        """Calculate payoff ratio (avg win / avg loss)."""
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        avg_win = np.mean(wins) if len(wins) > 0 else 0
        avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 1
        
        return avg_win / avg_loss if avg_loss > 0 else 0
    
    def _expectancy(self, returns: np.ndarray) -> float:
        """Calculate expectancy."""
        win_rate = self._win_rate(returns)
        payoff = self._payoff_ratio(returns)
        return win_rate * payoff - (1 - win_rate)
    
    def _sqn(self, returns: np.ndarray) -> float:
        """Calculate System Quality Number (Van Tharp)."""
        n = len(returns)
        mean_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1)
        
        if std_ret == 0:
            return 0
        
        sqn = (mean_ret / std_ret) * np.sqrt(n)
        return sqn
    
    def _trade_statistics(self, trades: List[Dict]) -> Tuple[float, float, float, float, float]:
        """Calculate statistics from trade list."""
        profits = [t.get('profit', 0) for t in trades]
        profits = np.array(profits)
        
        wins = profits[profits > 0]
        losses = profits[profits < 0]
        
        win_rate = len(wins) / len(profits) if len(profits) > 0 else 0
        
        gross_profit = np.sum(wins)
        gross_loss = abs(np.sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        avg_win = np.mean(wins) if len(wins) > 0 else 0
        avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 1
        payoff_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        
        expectancy = win_rate * payoff_ratio - (1 - win_rate)
        
        # SQN
        n = len(profits)
        mean_profit = np.mean(profits)
        std_profit = np.std(profits, ddof=1)
        sqn = (mean_profit / std_profit) * np.sqrt(n) if std_profit > 0 else 0
        
        return win_rate, profit_factor, payoff_ratio, expectancy, sqn
    
    # Distribution metrics
    def _skewness(self, returns: np.ndarray) -> float:
        """Calculate skewness."""
        return skew(returns)
    
    def _kurtosis(self, returns: np.ndarray) -> float:
        """Calculate excess kurtosis."""
        return kurtosis(returns)
    
    def _jarque_bera(self, returns: np.ndarray) -> Tuple[float, float]:
        """Jarque-Bera test for normality."""
        n = len(returns)
        s = skew(returns)
        k = kurtosis(returns)
        
        jb_stat = n / 6 * (s**2 + k**2 / 4)
        p_value = 1 - stats.chi2.cdf(jb_stat, 2)
        
        return jb_stat, p_value
    
    # Additional metrics
    def _recovery_factor(self, total_return: float, max_drawdown: float) -> float:
        """Calculate recovery factor."""
        return total_return / abs(max_drawdown) if max_drawdown < 0 else 0
    
    def _ulcer_index(self, drawdown: np.ndarray) -> float:
        """Calculate Ulcer Index."""
        return np.sqrt(np.mean(drawdown**2))
    
    def _monthly_analysis(self, returns: np.ndarray) -> Tuple[Dict, float, float, float]:
        """Analyze monthly returns."""
        # This is a simplified version
        # In practice, you'd want actual monthly aggregation
        n_months = len(returns) // 21  # Approximate
        
        if n_months == 0:
            return {}, 0, 0, 0
        
        monthly_rets = []
        for i in range(n_months):
            start = i * 21
            end = min((i + 1) * 21, len(returns))
            monthly_ret = np.prod(1 + returns[start:end]) - 1
            monthly_rets.append(monthly_ret)
        
        monthly_rets = np.array(monthly_rets)
        
        return (
            {f'month_{i}': r for i, r in enumerate(monthly_rets)},
            np.max(monthly_rets),
            np.min(monthly_rets),
            np.mean(monthly_rets > 0)
        )
    
    def _gain_to_pain_ratio(self, returns: np.ndarray) -> float:
        """Calculate Gain to Pain ratio (Seykota)."""
        total_return = np.sum(returns)
        total_pain = np.sum(np.abs(returns[returns < 0]))
        return total_return / total_pain if total_pain > 0 else 0
    
    def _common_sense_ratio(self, returns: np.ndarray) -> float:
        """Calculate Common Sense Ratio (profit factor * payoff ratio)."""
        pf = self._profit_factor(returns)
        pr = self._payoff_ratio(returns)
        return pf * pr
    
    def _tail_ratio(self, returns: np.ndarray) -> float:
        """Calculate Tail Ratio (95th percentile / 5th percentile)."""
        p95 = np.percentile(returns, 95)
        p5 = np.percentile(returns, 5)
        return abs(p95 / p5) if p5 != 0 else 0


class RollingMetrics:
    """
    Calculate rolling performance metrics over time.
    """
    
    def __init__(
        self,
        window: int = 63,
        risk_free_rate: float = 0.0,
        frequency: str = 'daily'
    ):
        """
        Initialize rolling metrics calculator.
        
        Args:
            window: Rolling window size
            risk_free_rate: Annual risk-free rate
            frequency: Data frequency
        """
        self.window = window
        self.analyzer = PerformanceAnalyzer(risk_free_rate, frequency)
    
    def calculate(
        self,
        returns: np.ndarray,
        metric: str = 'sharpe'
    ) -> np.ndarray:
        """
        Calculate rolling metric.
        
        Args:
            returns: Returns series
            metric: Metric to calculate
            
        Returns:
            Array of rolling metric values
        """
        returns = np.array(returns)
        n = len(returns)
        
        rolling_values = np.full(n, np.nan)
        
        for i in range(self.window, n):
            window_returns = returns[i - self.window:i]
            
            if metric == 'sharpe':
                value = self.analyzer._sharpe_ratio(window_returns)
            elif metric == 'sortino':
                value = self.analyzer._sortino_ratio(window_returns)
            elif metric == 'volatility':
                value = self.analyzer._volatility(window_returns)
            elif metric == 'return':
                value = self.analyzer._total_return(window_returns)
            elif metric == 'max_drawdown':
                value, _, _ = self.analyzer._drawdown_analysis(window_returns)
            elif metric == 'win_rate':
                value = self.analyzer._win_rate(window_returns)
            else:
                value = np.nan
            
            rolling_values[i] = value
        
        return rolling_values
    
    def calculate_multiple(
        self,
        returns: np.ndarray,
        metrics: List[str]
    ) -> pd.DataFrame:
        """
        Calculate multiple rolling metrics.
        
        Args:
            returns: Returns series
            metrics: List of metrics to calculate
            
        Returns:
            DataFrame with rolling metrics
        """
        results = {}
        
        for metric in metrics:
            results[metric] = self.calculate(returns, metric)
        
        return pd.DataFrame(results)


# Convenience functions
def quick_metrics(
    returns: np.ndarray,
    benchmark: Optional[np.ndarray] = None,
    risk_free_rate: float = 0.0
) -> Dict:
    """
    Quick performance metrics calculation.
    
    Args:
        returns: Strategy returns
        benchmark: Optional benchmark returns
        risk_free_rate: Risk-free rate
        
    Returns:
        Dictionary with key metrics
    """
    analyzer = PerformanceAnalyzer(risk_free_rate)
    metrics = analyzer.calculate_all_metrics(returns, benchmark)
    
    return {
        'total_return': metrics.total_return,
        'annualized_return': metrics.annualized_return,
        'volatility': metrics.volatility,
        'sharpe_ratio': metrics.sharpe_ratio,
        'sortino_ratio': metrics.sortino_ratio,
        'max_drawdown': metrics.max_drawdown,
        'win_rate': metrics.win_rate,
        'profit_factor': metrics.profit_factor,
        'calmar_ratio': metrics.calmar_ratio,
        'var_95': metrics.var_95,
        'expectancy': metrics.expectancy,
        'sqn': metrics.sqn
    }


def compare_strategies(
    returns_dict: Dict[str, np.ndarray],
    benchmark: Optional[np.ndarray] = None,
    risk_free_rate: float = 0.0
) -> pd.DataFrame:
    """
    Compare multiple strategies.
    
    Args:
        returns_dict: Dictionary of strategy name to returns
        benchmark: Optional benchmark returns
        risk_free_rate: Risk-free rate
        
    Returns:
        DataFrame with comparison metrics
    """
    analyzer = PerformanceAnalyzer(risk_free_rate)
    
    results = []
    for name, returns in returns_dict.items():
        metrics = analyzer.calculate_all_metrics(returns, benchmark)
        results.append({
            'Strategy': name,
            'Total Return': f"{metrics.total_return:.2%}",
            'Ann. Return': f"{metrics.annualized_return:.2%}",
            'Volatility': f"{metrics.volatility:.2%}",
            'Sharpe': f"{metrics.sharpe_ratio:.2f}",
            'Sortino': f"{metrics.sortino_ratio:.2f}",
            'Max DD': f"{metrics.max_drawdown:.2%}",
            'Calmar': f"{metrics.calmar_ratio:.2f}",
            'Win Rate': f"{metrics.win_rate:.2%}",
            'Profit Factor': f"{metrics.profit_factor:.2f}",
            'SQN': f"{metrics.sqn:.2f}"
        })
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    print("Performance Metrics Demo")
    print("=" * 50)
    
    # Generate sample returns
    n_days = 252 * 2
    
    # Strategy with positive drift
    strategy_returns = np.random.normal(0.0005, 0.02, n_days)
    
    # Benchmark
    benchmark_returns = np.random.normal(0.0003, 0.015, n_days)
    
    # Calculate metrics
    print("\n1. Quick Metrics:")
    metrics = quick_metrics(strategy_returns, benchmark_returns, risk_free_rate=0.02)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
    
    # Full metrics
    print("\n2. Full Performance Analysis:")
    analyzer = PerformanceAnalyzer(risk_free_rate=0.02)
    full_metrics = analyzer.calculate_all_metrics(strategy_returns, benchmark_returns)
    
    print(f"   Sharpe Ratio: {full_metrics.sharpe_ratio:.2f}")
    print(f"   Sortino Ratio: {full_metrics.sortino_ratio:.2f}")
    print(f"   Calmar Ratio: {full_metrics.calmar_ratio:.2f}")
    print(f"   Alpha: {full_metrics.alpha:.4f}")
    print(f"   Beta: {full_metrics.beta:.2f}")
    print(f"   Max Drawdown: {full_metrics.max_drawdown:.2%}")
    print(f"   SQN: {full_metrics.sqn:.2f}")
    
    # Rolling metrics
    print("\n3. Rolling Sharpe Ratio (last 10 values):")
    rolling = RollingMetrics(window=63)
    rolling_sharpe = rolling.calculate(strategy_returns, 'sharpe')
    print(f"   {rolling_sharpe[-10:]}")
    
    # Strategy comparison
    print("\n4. Strategy Comparison:")
    returns_dict = {
        'Strategy A': strategy_returns,
        'Strategy B': np.random.normal(0.0003, 0.018, n_days),
        'Strategy C': np.random.normal(0.0007, 0.025, n_days)
    }
    comparison = compare_strategies(returns_dict, benchmark_returns)
    print(comparison.to_string(index=False))
