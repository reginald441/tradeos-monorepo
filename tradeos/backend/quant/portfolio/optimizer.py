"""
Portfolio Optimization Module for TradeOS
=========================================

Implements various portfolio optimization techniques:
- Mean-Variance Optimization (Markowitz)
- Efficient Frontier calculation
- Risk Parity weighting
- Maximum Sharpe Ratio portfolio
- Minimum Variance portfolio
- Black-Litterman model

Author: TradeOS Quant Team
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Callable, Union
from dataclasses import dataclass
from scipy import optimize
from scipy.stats import norm
import warnings
from enum import Enum


class OptimizationObjective(Enum):
    """Optimization objectives."""
    MIN_VARIANCE = "min_variance"
    MAX_SHARPE = "max_sharpe"
    MAX_RETURN = "max_return"
    RISK_PARITY = "risk_parity"
    MAX_QUADRATIC_UTILITY = "max_quadratic_utility"
    MAX_DIVERSIFICATION = "max_diversification"


@dataclass
class OptimizationConstraints:
    """Constraints for portfolio optimization."""
    min_weight: float = 0.0
    max_weight: float = 1.0
    target_return: Optional[float] = None
    target_risk: Optional[float] = None
    allow_short: bool = False
    max_turnover: Optional[float] = None
    group_constraints: Optional[Dict[str, Tuple[float, float]]] = None
    sector_exposure: Optional[Dict[str, Tuple[float, float]]] = None


@dataclass
class OptimizationResult:
    """Result of portfolio optimization."""
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float
    diversification_ratio: float
    turnover: Optional[float] = None
    optimization_success: bool = True
    message: str = ""
    iterations: int = 0


class PortfolioOptimizer:
    """
    Advanced portfolio optimizer implementing multiple optimization techniques.
    
    Features:
    - Mean-variance optimization (Markowitz)
    - Efficient frontier generation
    - Risk parity allocation
    - Maximum Sharpe ratio portfolio
    - Black-Litterman model for views integration
    - Transaction cost aware optimization
    """
    
    def __init__(
        self,
        expected_returns: Optional[np.ndarray] = None,
        cov_matrix: Optional[np.ndarray] = None,
        returns_data: Optional[np.ndarray] = None,
        risk_free_rate: float = 0.0,
        frequency: int = 252
    ):
        """
        Initialize the portfolio optimizer.
        
        Args:
            expected_returns: Expected returns for each asset
            cov_matrix: Covariance matrix of returns
            returns_data: Historical returns data (n_periods x n_assets)
            risk_free_rate: Risk-free rate (annualized)
            frequency: Number of periods per year
        """
        self.risk_free_rate = risk_free_rate
        self.frequency = frequency
        
        if expected_returns is not None and cov_matrix is not None:
            self.expected_returns = np.array(expected_returns)
            self.cov_matrix = np.array(cov_matrix)
            self.n_assets = len(expected_returns)
        elif returns_data is not None:
            self.returns_data = np.array(returns_data)
            self.expected_returns = np.mean(returns_data, axis=0) * frequency
            self.cov_matrix = np.cov(returns_data.T) * frequency
            self.n_assets = returns_data.shape[1]
        else:
            raise ValueError("Must provide either (expected_returns, cov_matrix) or returns_data")
        
        # Validate dimensions
        if self.expected_returns.shape[0] != self.cov_matrix.shape[0]:
            raise ValueError("Expected returns and covariance matrix dimensions don't match")
        
        self._validate_covariance()
    
    def _validate_covariance(self):
        """Validate and fix covariance matrix if needed."""
        # Check if positive semi-definite
        eigenvalues = np.linalg.eigvalsh(self.cov_matrix)
        
        if np.any(eigenvalues < 0):
            warnings.warn("Covariance matrix is not positive semi-definite. Applying correction.")
            # Add small value to diagonal
            min_eig = np.min(eigenvalues)
            self.cov_matrix += np.eye(self.n_assets) * (abs(min_eig) + 1e-6)
    
    def optimize(
        self,
        objective: OptimizationObjective,
        constraints: Optional[OptimizationConstraints] = None,
        initial_weights: Optional[np.ndarray] = None
    ) -> OptimizationResult:
        """
        Optimize portfolio weights based on objective.
        
        Args:
            objective: Optimization objective
            constraints: Optimization constraints
            initial_weights: Starting weights for optimization
            
        Returns:
            OptimizationResult with optimal weights and statistics
        """
        constraints = constraints or OptimizationConstraints()
        
        if objective == OptimizationObjective.MIN_VARIANCE:
            return self.min_variance_portfolio(constraints, initial_weights)
        elif objective == OptimizationObjective.MAX_SHARPE:
            return self.max_sharpe_portfolio(constraints, initial_weights)
        elif objective == OptimizationObjective.RISK_PARITY:
            return self.risk_parity_portfolio(constraints, initial_weights)
        elif objective == OptimizationObjective.MAX_QUADRATIC_UTILITY:
            return self.max_quadratic_utility(constraints, initial_weights)
        elif objective == OptimizationObjective.MAX_DIVERSIFICATION:
            return self.max_diversification_portfolio(constraints, initial_weights)
        else:
            raise ValueError(f"Unknown objective: {objective}")
    
    def min_variance_portfolio(
        self,
        constraints: Optional[OptimizationConstraints] = None,
        initial_weights: Optional[np.ndarray] = None
    ) -> OptimizationResult:
        """
        Find minimum variance portfolio.
        
        Args:
            constraints: Optimization constraints
            initial_weights: Starting weights
            
        Returns:
            OptimizationResult
        """
        constraints = constraints or OptimizationConstraints()
        
        def portfolio_variance(w):
            return w @ self.cov_matrix @ w
        
        # Set up constraints
        opt_constraints = self._get_constraints(constraints)
        bounds = self._get_bounds(constraints)
        
        # Initial guess
        if initial_weights is None:
            initial_weights = np.ones(self.n_assets) / self.n_assets
        
        # Optimize
        result = optimize.minimize(
            portfolio_variance,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=opt_constraints,
            options={'ftol': 1e-9, 'maxiter': 1000}
        )
        
        weights = result.x
        weights = self._clean_weights(weights)
        
        return self._create_result(weights, result.success, result.message, result.nit)
    
    def max_sharpe_portfolio(
        self,
        constraints: Optional[OptimizationConstraints] = None,
        initial_weights: Optional[np.ndarray] = None
    ) -> OptimizationResult:
        """
        Find maximum Sharpe ratio portfolio.
        
        Args:
            constraints: Optimization constraints
            initial_weights: Starting weights
            
        Returns:
            OptimizationResult
        """
        constraints = constraints or OptimizationConstraints()
        
        def neg_sharpe(w):
            ret = w @ self.expected_returns
            vol = np.sqrt(w @ self.cov_matrix @ w)
            if vol == 0:
                return 0
            return -(ret - self.risk_free_rate) / vol
        
        opt_constraints = self._get_constraints(constraints)
        bounds = self._get_bounds(constraints)
        
        if initial_weights is None:
            initial_weights = np.ones(self.n_assets) / self.n_assets
        
        result = optimize.minimize(
            neg_sharpe,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=opt_constraints,
            options={'ftol': 1e-9, 'maxiter': 1000}
        )
        
        weights = result.x
        weights = self._clean_weights(weights)
        
        return self._create_result(weights, result.success, result.message, result.nit)
    
    def max_quadratic_utility(
        self,
        constraints: Optional[OptimizationConstraints] = None,
        initial_weights: Optional[np.ndarray] = None,
        risk_aversion: float = 1.0
    ) -> OptimizationResult:
        """
        Maximize quadratic utility: E[R] - 0.5 * lambda * Var[R]
        
        Args:
            constraints: Optimization constraints
            initial_weights: Starting weights
            risk_aversion: Risk aversion parameter (lambda)
            
        Returns:
            OptimizationResult
        """
        constraints = constraints or OptimizationConstraints()
        
        def neg_utility(w):
            ret = w @ self.expected_returns
            var = w @ self.cov_matrix @ w
            return -(ret - 0.5 * risk_aversion * var)
        
        opt_constraints = self._get_constraints(constraints)
        bounds = self._get_bounds(constraints)
        
        if initial_weights is None:
            initial_weights = np.ones(self.n_assets) / self.n_assets
        
        result = optimize.minimize(
            neg_utility,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=opt_constraints,
            options={'ftol': 1e-9, 'maxiter': 1000}
        )
        
        weights = result.x
        weights = self._clean_weights(weights)
        
        return self._create_result(weights, result.success, result.message, result.nit)
    
    def risk_parity_portfolio(
        self,
        constraints: Optional[OptimizationConstraints] = None,
        initial_weights: Optional[np.ndarray] = None,
        risk_budget: Optional[np.ndarray] = None
    ) -> OptimizationResult:
        """
        Construct risk parity portfolio where each asset contributes equally to risk.
        
        Args:
            constraints: Optimization constraints
            initial_weights: Starting weights
            risk_budget: Target risk contribution for each asset (default: equal)
            
        Returns:
            OptimizationResult
        """
        constraints = constraints or OptimizationConstraints()
        
        if risk_budget is None:
            risk_budget = np.ones(self.n_assets) / self.n_assets
        else:
            risk_budget = np.array(risk_budget)
            risk_budget = risk_budget / risk_budget.sum()
        
        def risk_parity_objective(w):
            # Calculate risk contributions
            portfolio_vol = np.sqrt(w @ self.cov_matrix @ w)
            if portfolio_vol == 0:
                return 1e10
            
            marginal_risk = (self.cov_matrix @ w) / portfolio_vol
            risk_contrib = w * marginal_risk
            
            # Target: risk_contrib / portfolio_vol = risk_budget
            target_risk_contrib = risk_budget * portfolio_vol
            return np.sum((risk_contrib - target_risk_contrib) ** 2)
        
        opt_constraints = self._get_constraints(constraints)
        bounds = self._get_bounds(constraints)
        
        if initial_weights is None:
            initial_weights = risk_budget
        
        result = optimize.minimize(
            risk_parity_objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=opt_constraints,
            options={'ftol': 1e-12, 'maxiter': 2000}
        )
        
        weights = result.x
        weights = self._clean_weights(weights)
        
        return self._create_result(weights, result.success, result.message, result.nit)
    
    def max_diversification_portfolio(
        self,
        constraints: Optional[OptimizationConstraints] = None,
        initial_weights: Optional[np.ndarray] = None
    ) -> OptimizationResult:
        """
        Find maximum diversification portfolio.
        
        Diversification ratio = (w^T * sigma) / sqrt(w^T * Sigma * w)
        where sigma is vector of asset volatilities
        
        Args:
            constraints: Optimization constraints
            initial_weights: Starting weights
            
        Returns:
            OptimizationResult
        """
        constraints = constraints or OptimizationConstraints()
        
        asset_vols = np.sqrt(np.diag(self.cov_matrix))
        
        def neg_diversification_ratio(w):
            weighted_vol = w @ asset_vols
            portfolio_vol = np.sqrt(w @ self.cov_matrix @ w)
            if portfolio_vol == 0:
                return 0
            return -weighted_vol / portfolio_vol
        
        opt_constraints = self._get_constraints(constraints)
        bounds = self._get_bounds(constraints)
        
        if initial_weights is None:
            initial_weights = np.ones(self.n_assets) / self.n_assets
        
        result = optimize.minimize(
            neg_diversification_ratio,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=opt_constraints,
            options={'ftol': 1e-9, 'maxiter': 1000}
        )
        
        weights = result.x
        weights = self._clean_weights(weights)
        
        opt_result = self._create_result(weights, result.success, result.message, result.nit)
        opt_result.diversification_ratio = -result.fun
        
        return opt_result
    
    def efficient_frontier(
        self,
        n_points: int = 100,
        constraints: Optional[OptimizationConstraints] = None,
        return_range: Optional[Tuple[float, float]] = None
    ) -> pd.DataFrame:
        """
        Calculate efficient frontier.
        
        Args:
            n_points: Number of points on frontier
            constraints: Optimization constraints
            return_range: (min_return, max_return) for frontier
            
        Returns:
            DataFrame with efficient frontier points
        """
        constraints = constraints or OptimizationConstraints()
        
        # Determine return range
        if return_range is None:
            min_ret = np.min(self.expected_returns)
            max_ret = np.max(self.expected_returns)
        else:
            min_ret, max_ret = return_range
        
        target_returns = np.linspace(min_ret, max_ret, n_points)
        
        results = []
        for target in target_returns:
            # Create constraint for target return
            target_constraints = OptimizationConstraints(
                min_weight=constraints.min_weight,
                max_weight=constraints.max_weight,
                target_return=target,
                allow_short=constraints.allow_short
            )
            
            result = self.min_variance_portfolio(target_constraints)
            
            if result.optimization_success:
                results.append({
                    'return': result.expected_return,
                    'volatility': result.volatility,
                    'sharpe': result.sharpe_ratio,
                    'weights': result.weights
                })
        
        return pd.DataFrame(results)
    
    def _get_constraints(
        self,
        constraints: OptimizationConstraints
    ) -> List[Dict]:
        """Build optimization constraints."""
        opt_constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        if constraints.target_return is not None:
            opt_constraints.append({
                'type': 'eq',
                'fun': lambda x: x @ self.expected_returns - constraints.target_return
            })
        
        if constraints.target_risk is not None:
            opt_constraints.append({
                'type': 'eq',
                'fun': lambda x: np.sqrt(x @ self.cov_matrix @ x) - constraints.target_risk
            })
        
        return opt_constraints
    
    def _get_bounds(
        self,
        constraints: OptimizationConstraints
    ) -> List[Tuple[float, float]]:
        """Build weight bounds."""
        if constraints.allow_short:
            return [(-np.inf, np.inf)] * self.n_assets
        else:
            return [(constraints.min_weight, constraints.max_weight)] * self.n_assets
    
    def _clean_weights(self, weights: np.ndarray, threshold: float = 1e-8) -> np.ndarray:
        """Clean weights by removing small values and normalizing."""
        weights = np.where(np.abs(weights) < threshold, 0, weights)
        weights = weights / np.sum(np.abs(weights))
        return weights
    
    def _create_result(
        self,
        weights: np.ndarray,
        success: bool,
        message: str,
        iterations: int
    ) -> OptimizationResult:
        """Create optimization result from weights."""
        expected_return = weights @ self.expected_returns
        volatility = np.sqrt(weights @ self.cov_matrix @ weights)
        
        sharpe = (expected_return - self.risk_free_rate) / volatility if volatility > 0 else 0
        
        # Diversification ratio
        asset_vols = np.sqrt(np.diag(self.cov_matrix))
        div_ratio = (weights @ asset_vols) / volatility if volatility > 0 else 0
        
        return OptimizationResult(
            weights=weights,
            expected_return=expected_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            diversification_ratio=div_ratio,
            optimization_success=success,
            message=message,
            iterations=iterations
        )
    
    def get_risk_contributions(self, weights: np.ndarray) -> np.ndarray:
        """Calculate risk contribution of each asset."""
        weights = np.array(weights)
        portfolio_vol = np.sqrt(weights @ self.cov_matrix @ weights)
        
        if portfolio_vol == 0:
            return np.zeros(self.n_assets)
        
        marginal_risk = (self.cov_matrix @ weights) / portfolio_vol
        risk_contrib = weights * marginal_risk
        
        return risk_contrib / portfolio_vol


class BlackLittermanModel:
    """
    Black-Litterman model for incorporating investor views into portfolio optimization.
    
    The model combines market equilibrium returns with investor views to produce
    posterior expected returns that can be used in mean-variance optimization.
    """
    
    def __init__(
        self,
        cov_matrix: np.ndarray,
        market_weights: np.ndarray,
        risk_aversion: float = 2.5,
        risk_free_rate: float = 0.0
    ):
        """
        Initialize Black-Litterman model.
        
        Args:
            cov_matrix: Covariance matrix of asset returns
            market_weights: Market capitalization weights
            risk_aversion: Risk aversion parameter
            risk_free_rate: Risk-free rate
        """
        self.cov_matrix = np.array(cov_matrix)
        self.market_weights = np.array(market_weights)
        self.risk_aversion = risk_aversion
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(market_weights)
        
        # Calculate equilibrium returns (implied returns)
        self.equilibrium_returns = self._calculate_equilibrium_returns()
        
        # Prior covariance (scaled)
        self.tau = 0.05  # Uncertainty scaling factor
    
    def _calculate_equilibrium_returns(self) -> np.ndarray:
        """Calculate equilibrium returns from market weights."""
        return self.risk_aversion * self.cov_matrix @ self.market_weights
    
    def add_views(
        self,
        view_matrix: np.ndarray,
        view_returns: np.ndarray,
        view_confidences: Optional[np.ndarray] = None,
        omega: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Incorporate investor views and calculate posterior returns.
        
        Args:
            view_matrix: P matrix (k x n) where k is number of views
            view_returns: Q vector (k) of expected returns for each view
            view_confidences: Confidence in each view (0-1)
            omega: View uncertainty matrix (k x k), overrides confidences
            
        Returns:
            Posterior expected returns
        """
        P = np.array(view_matrix)
        Q = np.array(view_returns)
        
        k = len(Q)
        
        # Calculate view uncertainty (omega)
        if omega is None:
            if view_confidences is None:
                view_confidences = np.ones(k) * 0.5
            
            # Use Idzorek's method for confidence
            omega = np.zeros((k, k))
            for i in range(k):
                # Uncertainty proportional to view variance
                view_var = P[i] @ self.cov_matrix @ P[i]
                omega[i, i] = view_var * (1 - view_confidences[i]) / view_confidences[i]
        else:
            omega = np.array(omega)
        
        # Calculate posterior returns
        # Posterior = [(tau*Sigma)^-1 + P^T * Omega^-1 * P]^-1 * 
        #             [(tau*Sigma)^-1 * Pi + P^T * Omega^-1 * Q]
        
        tau_sigma_inv = np.linalg.inv(self.tau * self.cov_matrix)
        omega_inv = np.linalg.inv(omega)
        
        middle_term = np.linalg.inv(tau_sigma_inv + P.T @ omega_inv @ P)
        right_term = tau_sigma_inv @ self.equilibrium_returns + P.T @ omega_inv @ Q
        
        posterior_returns = middle_term @ right_term
        
        # Posterior covariance
        self.posterior_cov = self.cov_matrix + middle_term
        
        return posterior_returns
    
    def optimize_with_views(
        self,
        view_matrix: np.ndarray,
        view_returns: np.ndarray,
        view_confidences: Optional[np.ndarray] = None,
        constraints: Optional[OptimizationConstraints] = None
    ) -> OptimizationResult:
        """
        Optimize portfolio using Black-Litterman posterior returns.
        
        Args:
            view_matrix: View matrix P
            view_returns: View returns Q
            view_confidences: Confidence in views
            constraints: Optimization constraints
            
        Returns:
            OptimizationResult
        """
        posterior_returns = self.add_views(view_matrix, view_returns, view_confidences)
        
        # Create optimizer with posterior returns
        optimizer = PortfolioOptimizer(
            expected_returns=posterior_returns,
            cov_matrix=self.posterior_cov,
            risk_free_rate=self.risk_free_rate
        )
        
        return optimizer.max_sharpe_portfolio(constraints)


class HierarchicalRiskParity:
    """
    Hierarchical Risk Parity (HRP) portfolio allocation.
    
    Based on Marcos Lopez de Prado's approach:
    1. Hierarchical clustering of assets
    2. Recursive bisection for weight allocation
    """
    
    def __init__(self, returns_data: np.ndarray):
        """
        Initialize HRP allocator.
        
        Args:
            returns_data: Historical returns (n_periods x n_assets)
        """
        self.returns_data = np.array(returns_data)
        self.n_assets = returns_data.shape[1]
        self.cov_matrix = np.cov(returns_data.T)
        self.corr_matrix = np.corrcoef(returns_data.T)
    
    def allocate(self, linkage_method: str = 'single') -> np.ndarray:
        """
        Allocate weights using HRP algorithm.
        
        Args:
            linkage_method: Clustering linkage method
            
        Returns:
            Portfolio weights
        """
        from scipy.cluster.hierarchy import linkage, dendrogram
        from scipy.spatial.distance import squareform
        
        # Calculate distance matrix
        dist_matrix = np.sqrt(0.5 * (1 - self.corr_matrix))
        
        # Hierarchical clustering
        dist_linkage = linkage(squareform(dist_matrix), method=linkage_method)
        
        # Get quasi-diagonalization order
        sort_ix = self._get_quasi_diag(dist_linkage)
        
        # Recursive bisection
        weights = self._recursive_bisection(sort_ix)
        
        return weights
    
    def _get_quasi_diag(self, linkage: np.ndarray) -> List[int]:
        """Get quasi-diagonal ordering from linkage matrix."""
        n = linkage.shape[0] + 1
        
        def get_cluster_items(cluster_id):
            if cluster_id < n:
                return [int(cluster_id)]
            else:
                left = int(linkage[cluster_id - n, 0])
                right = int(linkage[cluster_id - n, 1])
                return get_cluster_items(left) + get_cluster_items(right)
        
        root = 2 * n - 2
        return get_cluster_items(root)
    
    def _recursive_bisection(self, sort_ix: List[int]) -> np.ndarray:
        """Allocate weights using recursive bisection."""
        weights = np.ones(len(sort_ix))
        
        # Initialize clusters
        clusters = [sort_ix]
        
        while len(clusters) > 0:
            # Bisect each cluster
            new_clusters = []
            
            for cluster in clusters:
                if len(cluster) == 1:
                    continue
                
                # Split cluster in half
                mid = len(cluster) // 2
                left_cluster = cluster[:mid]
                right_cluster = cluster[mid:]
                
                # Calculate cluster variances
                left_var = self._get_cluster_variance(left_cluster)
                right_var = self._get_cluster_variance(right_cluster)
                
                # Allocate weight based on inverse variance
                alpha = 1 - left_var / (left_var + right_var)
                
                # Update weights
                left_indices = [sort_ix.index(i) for i in left_cluster]
                right_indices = [sort_ix.index(i) for i in right_cluster]
                
                weights[left_indices] *= alpha
                weights[right_indices] *= (1 - alpha)
                
                # Add sub-clusters for further bisection
                if len(left_cluster) > 1:
                    new_clusters.append(left_cluster)
                if len(right_cluster) > 1:
                    new_clusters.append(right_cluster)
            
            clusters = new_clusters
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
        # Reorder to original asset order
        original_weights = np.zeros(self.n_assets)
        for i, idx in enumerate(sort_ix):
            original_weights[idx] = weights[i]
        
        return original_weights
    
    def _get_cluster_variance(self, cluster: List[int]) -> float:
        """Calculate variance of a cluster."""
        if len(cluster) == 0:
            return 0
        
        cluster_cov = self.cov_matrix[np.ix_(cluster, cluster)]
        
        # Use inverse variance allocation within cluster
        ivp = 1 / np.diag(cluster_cov)
        ivp = ivp / np.sum(ivp)
        
        return ivp @ cluster_cov @ ivp


class TransactionCostOptimizer:
    """
    Portfolio optimizer that accounts for transaction costs.
    """
    
    def __init__(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        current_weights: np.ndarray,
        transaction_costs: np.ndarray,
        risk_free_rate: float = 0.0
    ):
        """
        Initialize transaction cost optimizer.
        
        Args:
            expected_returns: Expected returns
            cov_matrix: Covariance matrix
            current_weights: Current portfolio weights
            transaction_costs: Transaction cost per asset (proportional)
            risk_free_rate: Risk-free rate
        """
        self.expected_returns = np.array(expected_returns)
        self.cov_matrix = np.array(cov_matrix)
        self.current_weights = np.array(current_weights)
        self.transaction_costs = np.array(transaction_costs)
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(expected_returns)
    
    def optimize(
        self,
        cost_aversion: float = 1.0,
        constraints: Optional[OptimizationConstraints] = None
    ) -> OptimizationResult:
        """
        Optimize considering transaction costs.
        
        Args:
            cost_aversion: Weight on transaction costs in objective
            constraints: Optimization constraints
            
        Returns:
            OptimizationResult
        """
        constraints = constraints or OptimizationConstraints()
        
        def objective(w):
            # Expected utility minus transaction costs
            ret = w @ self.expected_returns
            var = w @ self.cov_matrix @ w
            utility = ret - 0.5 * var
            
            # Transaction costs
            turnover = np.abs(w - self.current_weights)
            costs = np.sum(turnover * self.transaction_costs)
            
            return -(utility - cost_aversion * costs)
        
        opt_constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        ]
        
        if not constraints.allow_short:
            bounds = [(0, 1)] * self.n_assets
        else:
            bounds = [(-1, 1)] * self.n_assets
        
        initial_weights = self.current_weights
        
        result = optimize.minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=opt_constraints,
            options={'ftol': 1e-9, 'maxiter': 1000}
        )
        
        weights = result.x
        weights = weights / np.sum(np.abs(weights))
        
        turnover = np.sum(np.abs(weights - self.current_weights))
        
        opt_result = self._create_result(weights, result.success, result.message, result.nit)
        opt_result.turnover = turnover
        
        return opt_result
    
    def _create_result(
        self,
        weights: np.ndarray,
        success: bool,
        message: str,
        iterations: int
    ) -> OptimizationResult:
        """Create optimization result."""
        expected_return = weights @ self.expected_returns
        volatility = np.sqrt(weights @ self.cov_matrix @ weights)
        sharpe = (expected_return - self.risk_free_rate) / volatility if volatility > 0 else 0
        
        return OptimizationResult(
            weights=weights,
            expected_return=expected_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            diversification_ratio=0,
            optimization_success=success,
            message=message,
            iterations=iterations
        )


# Convenience functions
def optimize_portfolio(
    returns_data: np.ndarray,
    objective: str = "max_sharpe",
    risk_free_rate: float = 0.0
) -> Dict:
    """
    Quick portfolio optimization.
    
    Args:
        returns_data: Historical returns (n_periods x n_assets)
        objective: Optimization objective
        risk_free_rate: Risk-free rate
        
    Returns:
        Dictionary with optimization results
    """
    optimizer = PortfolioOptimizer(
        returns_data=returns_data,
        risk_free_rate=risk_free_rate
    )
    
    obj_map = {
        "max_sharpe": OptimizationObjective.MAX_SHARPE,
        "min_variance": OptimizationObjective.MIN_VARIANCE,
        "risk_parity": OptimizationObjective.RISK_PARITY
    }
    
    result = optimizer.optimize(obj_map.get(objective, OptimizationObjective.MAX_SHARPE))
    
    return {
        "weights": result.weights.tolist(),
        "expected_return": result.expected_return,
        "volatility": result.volatility,
        "sharpe_ratio": result.sharpe_ratio,
        "success": result.optimization_success
    }


def get_efficient_frontier(
    returns_data: np.ndarray,
    n_points: int = 50,
    risk_free_rate: float = 0.0
) -> pd.DataFrame:
    """
    Quick efficient frontier calculation.
    
    Args:
        returns_data: Historical returns
        n_points: Number of frontier points
        risk_free_rate: Risk-free rate
        
    Returns:
        DataFrame with efficient frontier
    """
    optimizer = PortfolioOptimizer(
        returns_data=returns_data,
        risk_free_rate=risk_free_rate
    )
    
    return optimizer.efficient_frontier(n_points=n_points)


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate sample returns data
    n_assets = 5
    n_periods = 252
    
    # Create correlated returns
    mean_returns = np.array([0.12, 0.10, 0.08, 0.06, 0.04]) / 252
    volatilities = np.array([0.20, 0.18, 0.15, 0.12, 0.10]) / np.sqrt(252)
    
    # Correlation matrix
    corr = np.array([
        [1.0, 0.7, 0.5, 0.3, 0.2],
        [0.7, 1.0, 0.6, 0.4, 0.3],
        [0.5, 0.6, 1.0, 0.5, 0.4],
        [0.3, 0.4, 0.5, 1.0, 0.6],
        [0.2, 0.3, 0.4, 0.6, 1.0]
    ])
    
    cov = np.outer(volatilities, volatilities) * corr
    returns_data = np.random.multivariate_normal(mean_returns, cov, n_periods)
    
    # Optimize portfolio
    optimizer = PortfolioOptimizer(returns_data=returns_data, risk_free_rate=0.02)
    
    print("Portfolio Optimization Results")
    print("=" * 50)
    
    # Max Sharpe
    result = optimizer.max_sharpe_portfolio()
    print("\nMaximum Sharpe Ratio Portfolio:")
    print(f"  Weights: {result.weights}")
    print(f"  Expected Return: {result.expected_return:.2%}")
    print(f"  Volatility: {result.volatility:.2%}")
    print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
    
    # Min Variance
    result = optimizer.min_variance_portfolio()
    print("\nMinimum Variance Portfolio:")
    print(f"  Weights: {result.weights}")
    print(f"  Expected Return: {result.expected_return:.2%}")
    print(f"  Volatility: {result.volatility:.2%}")
    
    # Risk Parity
    result = optimizer.risk_parity_portfolio()
    print("\nRisk Parity Portfolio:")
    print(f"  Weights: {result.weights}")
    print(f"  Risk Contributions: {optimizer.get_risk_contributions(result.weights)}")
