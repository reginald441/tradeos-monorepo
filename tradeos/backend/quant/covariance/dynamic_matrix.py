"""
Dynamic Covariance and Correlation Modeling for TradeOS
=======================================================

Implements dynamic covariance modeling techniques:
- Dynamic correlation matrix
- Exponentially weighted covariance
- GARCH volatility modeling
- Volatility clustering detection
- Cross-asset exposure adjustment

Author: TradeOS Quant Team
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union, Callable
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import minimize
import warnings


@dataclass
class CovarianceConfig:
    """Configuration for covariance modeling."""
    method: str = "ewm"  # 'ewm', 'garch', 'dcc', 'realized'
    decay_factor: float = 0.94  # For EWM
    window_size: int = 252
    min_periods: int = 30
    annualize: bool = True
    frequency: int = 252


class DynamicCovarianceEstimator:
    """
    Dynamic covariance and correlation matrix estimator.
    
    Implements multiple methods for estimating time-varying
    covariance matrices with focus on financial applications.
    """
    
    def __init__(self, config: Optional[CovarianceConfig] = None):
        """Initialize the estimator."""
        self.config = config or CovarianceConfig()
        self.returns_history = []
        self.cov_history = []
        self.corr_history = []
        self.vol_history = []
    
    def fit(
        self,
        returns: np.ndarray,
        timestamps: Optional[pd.DatetimeIndex] = None
    ) -> Dict:
        """
        Fit dynamic covariance model.
        
        Args:
            returns: Returns matrix (n_periods x n_assets)
            timestamps: Optional timestamps for the returns
            
        Returns:
            Dictionary with fitted model parameters
        """
        returns = np.array(returns)
        self.returns_history = returns
        
        if self.config.method == "ewm":
            return self._fit_ewm(returns)
        elif self.config.method == "garch":
            return self._fit_garch(returns)
        elif self.config.method == "dcc":
            return self._fit_dcc(returns)
        elif self.config.method == "realized":
            return self._fit_realized(returns)
        else:
            raise ValueError(f"Unknown method: {self.config.method}")
    
    def _fit_ewm(self, returns: np.ndarray) -> Dict:
        """Fit exponentially weighted moving average covariance."""
        n_periods, n_assets = returns.shape
        lambda_param = self.config.decay_factor
        
        # Calculate EWM covariance
        weights = np.array([(1 - lambda_param) * lambda_param**i 
                           for i in range(n_periods)])[::-1]
        weights = weights / weights.sum()
        
        # Weighted mean
        weighted_mean = np.average(returns, axis=0, weights=weights)
        
        # Weighted covariance
        centered = returns - weighted_mean
        ewm_cov = np.zeros((n_assets, n_assets))
        
        for t in range(n_periods):
            ewm_cov += weights[t] * np.outer(centered[t], centered[t])
        
        if self.config.annualize:
            ewm_cov *= self.config.frequency
        
        # Store
        self.current_cov = ewm_cov
        self.current_corr = self._cov_to_corr(ewm_cov)
        self.current_vol = np.sqrt(np.diag(ewm_cov))
        
        return {
            'covariance': ewm_cov,
            'correlation': self.current_corr,
            'volatility': self.current_vol,
            'method': 'ewm',
            'lambda': lambda_param
        }
    
    def _fit_garch(self, returns: np.ndarray) -> Dict:
        """Fit GARCH-type volatility models."""
        n_periods, n_assets = returns.shape
        
        # Fit univariate GARCH for each asset
        garch_params = []
        conditional_vols = np.zeros((n_periods, n_assets))
        
        for i in range(n_assets):
            params, vols = self._fit_univariate_garch(returns[:, i])
            garch_params.append(params)
            conditional_vols[:, i] = vols
        
        # Calculate correlation matrix (using EWMA on standardized returns)
        standardized = returns / (conditional_vols + 1e-10)
        corr_matrix = np.corrcoef(standardized.T)
        
        # Construct covariance matrix
        vol_diag = np.diag(conditional_vols[-1])
        garch_cov = vol_diag @ corr_matrix @ vol_diag
        
        if self.config.annualize:
            garch_cov *= self.config.frequency
        
        self.current_cov = garch_cov
        self.current_corr = corr_matrix
        self.current_vol = conditional_vols[-1] * np.sqrt(self.config.frequency if self.config.annualize else 1)
        
        return {
            'covariance': garch_cov,
            'correlation': corr_matrix,
            'volatility': self.current_vol,
            'conditional_vols': conditional_vols,
            'garch_params': garch_params,
            'method': 'garch'
        }
    
    def _fit_univariate_garch(
        self,
        returns: np.ndarray,
        p: int = 1,
        q: int = 1
    ) -> Tuple[Dict, np.ndarray]:
        """
        Fit univariate GARCH(p,q) model.
        
        Args:
            returns: Return series
            p: GARCH order
            q: ARCH order
            
        Returns:
            (parameters, conditional variances)
        """
        returns = np.array(returns)
        T = len(returns)
        
        # Initial parameter guesses
        var = np.var(returns)
        initial_params = [0.01, 0.1, 0.85]  # omega, alpha, beta
        
        # Bounds
        bounds = [(1e-6, 1), (0, 1), (0, 1)]
        
        # Constraint: alpha + beta < 1 (stationarity)
        def stationarity_constraint(params):
            return 1 - params[1] - params[2]
        
        constraints = {'type': 'ineq', 'fun': stationarity_constraint}
        
        # Negative log-likelihood
        def neg_loglik(params):
            omega, alpha, beta = params
            sigma2 = np.zeros(T)
            sigma2[0] = var
            
            for t in range(1, T):
                sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
            
            loglik = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + returns**2 / sigma2)
            return -loglik
        
        # Optimize
        result = minimize(
            neg_loglik,
            initial_params,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        omega, alpha, beta = result.x
        
        # Calculate conditional variances
        sigma2 = np.zeros(T)
        sigma2[0] = var
        
        for t in range(1, T):
            sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
        
        return {
            'omega': omega,
            'alpha': alpha,
            'beta': beta,
            'persistence': alpha + beta
        }, np.sqrt(sigma2)
    
    def _fit_dcc(self, returns: np.ndarray) -> Dict:
        """
        Fit Dynamic Conditional Correlation (DCC) model.
        
        Two-step procedure:
        1. Fit univariate GARCH for each asset
        2. Model dynamic correlation
        """
        n_periods, n_assets = returns.shape
        
        # Step 1: Univariate GARCH
        standardized = np.zeros_like(returns)
        vols = np.zeros((n_periods, n_assets))
        
        for i in range(n_assets):
            _, vols[:, i] = self._fit_univariate_garch(returns[:, i])
            standardized[:, i] = returns[:, i] / (vols[:, i] + 1e-10)
        
        # Step 2: DCC parameters
        # Simplified DCC(1,1) estimation
        Q_bar = np.corrcoef(standardized.T)
        
        # Initialize
        Q_t = Q_bar.copy()
        R_history = []
        
        a, b = 0.05, 0.93  # Typical DCC parameters
        
        for t in range(1, n_periods):
            # Update Q
            outer_prod = np.outer(standardized[t-1], standardized[t-1])
            Q_t = Q_bar * (1 - a - b) + a * outer_prod + b * Q_t
            
            # Convert to correlation
            Q_diag_inv = np.diag(1 / np.sqrt(np.diag(Q_t)))
            R_t = Q_diag_inv @ Q_t @ Q_diag_inv
            R_history.append(R_t)
        
        # Final correlation
        final_R = R_history[-1] if R_history else Q_bar
        
        # Construct covariance
        final_vol = vols[-1]
        vol_diag = np.diag(final_vol)
        dcc_cov = vol_diag @ final_R @ vol_diag
        
        if self.config.annualize:
            dcc_cov *= self.config.frequency
        
        self.current_cov = dcc_cov
        self.current_corr = final_R
        self.current_vol = final_vol * np.sqrt(self.config.frequency if self.config.annualize else 1)
        
        return {
            'covariance': dcc_cov,
            'correlation': final_R,
            'volatility': self.current_vol,
            'dcc_params': {'a': a, 'b': b},
            'method': 'dcc'
        }
    
    def _fit_realized(self, returns: np.ndarray) -> Dict:
        """
        Fit using realized covariance (for high-frequency data).
        
        For daily data, this falls back to standard sample covariance.
        """
        n_periods, n_assets = returns.shape
        
        # Use rolling window for realized covariance
        window = min(self.config.window_size, n_periods)
        
        realized_cov = np.cov(returns[-window:].T)
        
        if self.config.annualize:
            realized_cov *= self.config.frequency
        
        self.current_cov = realized_cov
        self.current_corr = self._cov_to_corr(realized_cov)
        self.current_vol = np.sqrt(np.diag(realized_cov))
        
        return {
            'covariance': realized_cov,
            'correlation': self.current_corr,
            'volatility': self.current_vol,
            'method': 'realized',
            'window': window
        }
    
    def _cov_to_corr(self, cov: np.ndarray) -> np.ndarray:
        """Convert covariance matrix to correlation matrix."""
        vols = np.sqrt(np.diag(cov))
        vols_inv = np.diag(1 / (vols + 1e-10))
        return vols_inv @ cov @ vols_inv
    
    def update(self, new_returns: np.ndarray) -> Dict:
        """
        Update covariance estimate with new returns.
        
        Args:
            new_returns: New return observation(s)
            
        Returns:
            Updated covariance estimate
        """
        new_returns = np.array(new_returns)
        
        if new_returns.ndim == 1:
            new_returns = new_returns.reshape(1, -1)
        
        # Append to history
        self.returns_history = np.vstack([self.returns_history, new_returns])
        
        # Keep only necessary history
        max_history = max(self.config.window_size, 500)
        if len(self.returns_history) > max_history:
            self.returns_history = self.returns_history[-max_history:]
        
        # Refit
        return self.fit(self.returns_history)
    
    def predict(self, horizon: int = 1) -> Dict:
        """
        Predict covariance matrix for future horizon.
        
        Args:
            horizon: Prediction horizon
            
        Returns:
            Predicted covariance matrix
        """
        if self.config.method == "garch":
            # Use GARCH persistence for prediction
            return self._predict_garch(horizon)
        else:
            # Use current estimate for short-term prediction
            return {
                'covariance': self.current_cov,
                'correlation': self.current_corr,
                'volatility': self.current_vol,
                'horizon': horizon
            }
    
    def _predict_garch(self, horizon: int) -> Dict:
        """Predict using GARCH model."""
        # For GARCH, variance converges to long-run average
        # Simplified prediction
        return {
            'covariance': self.current_cov,
            'correlation': self.current_corr,
            'volatility': self.current_vol,
            'horizon': horizon
        }


class VolatilityClusteringDetector:
    """
    Detect and measure volatility clustering in return series.
    
    Volatility clustering is a key stylized fact in financial returns
    where large changes tend to be followed by large changes.
    """
    
    def __init__(self):
        """Initialize the detector."""
        self.tests = {}
    
    def detect(self, returns: np.ndarray) -> Dict:
        """
        Detect volatility clustering using multiple tests.
        
        Args:
            returns: Return series
            
        Returns:
            Dictionary with test results
        """
        returns = np.array(returns).flatten()
        
        results = {}
        
        # 1. ARCH Test (Engle's LM test)
        results['arch_test'] = self._arch_test(returns)
        
        # 2. Ljung-Box test on squared returns
        results['ljung_box'] = self._ljung_box_test(returns**2)
        
        # 3. Runs test on volatility regimes
        results['runs_test'] = self._runs_test(returns)
        
        # 4. Autocorrelation of absolute returns
        results['abs_autocorr'] = self._abs_autocorr(returns)
        
        # 5. Volatility regime analysis
        results['regime_analysis'] = self._regime_analysis(returns)
        
        # Overall assessment
        evidence_count = sum([
            results['arch_test']['significant'],
            results['ljung_box']['significant'],
            results['runs_test']['significant'],
            results['abs_autocorr']['significant']
        ])
        
        if evidence_count >= 3:
            clustering_strength = 'strong'
        elif evidence_count >= 2:
            clustering_strength = 'moderate'
        elif evidence_count >= 1:
            clustering_strength = 'weak'
        else:
            clustering_strength = 'none'
        
        results['overall'] = {
            'clustering_detected': evidence_count >= 2,
            'clustering_strength': clustering_strength,
            'evidence_count': evidence_count
        }
        
        return results
    
    def _arch_test(self, returns: np.ndarray, lags: int = 5) -> Dict:
        """
        Engle's ARCH test for volatility clustering.
        
        Tests whether squared returns show autocorrelation.
        """
        squared = returns**2
        T = len(squared)
        
        # Regress squared returns on lagged squared returns
        X = np.zeros((T - lags, lags))
        y = squared[lags:]
        
        for i in range(lags):
            X[:, i] = squared[lags - i - 1:T - i - 1]
        
        # OLS
        X_with_const = np.column_stack([np.ones(len(X)), X])
        beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
        
        # R-squared
        y_pred = X_with_const @ beta
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - ss_res / ss_tot
        
        # LM statistic
        lm_stat = (T - lags) * r_squared
        p_value = 1 - stats.chi2.cdf(lm_stat, lags)
        
        return {
            'lm_statistic': lm_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'r_squared': r_squared
        }
    
    def _ljung_box_test(self, data: np.ndarray, lags: int = 10) -> Dict:
        """
        Ljung-Box test for autocorrelation.
        """
        n = len(data)
        
        # Calculate autocorrelations
        autocorrs = []
        for lag in range(1, lags + 1):
            if lag < n:
                corr = np.corrcoef(data[lag:], data[:-lag])[0, 1]
                autocorrs.append(corr if not np.isnan(corr) else 0)
            else:
                autocorrs.append(0)
        
        # Ljung-Box statistic
        lb_stat = n * (n + 2) * sum([(autocorrs[i]**2) / (n - i - 1) for i in range(lags)])
        p_value = 1 - stats.chi2.cdf(lb_stat, lags)
        
        return {
            'lb_statistic': lb_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'autocorrelations': autocorrs
        }
    
    def _runs_test(self, returns: np.ndarray) -> Dict:
        """
        Runs test for randomness in volatility regimes.
        """
        # Define high/low volatility regimes
        vol_threshold = np.std(returns)
        high_vol = np.abs(returns) > vol_threshold
        
        # Count runs
        n = len(high_vol)
        n_runs = 1
        for i in range(1, n):
            if high_vol[i] != high_vol[i-1]:
                n_runs += 1
        
        n_high = np.sum(high_vol)
        n_low = n - n_high
        
        # Expected runs under randomness
        expected_runs = (2 * n_high * n_low) / n + 1
        variance_runs = (2 * n_high * n_low * (2 * n_high * n_low - n)) / (n**2 * (n - 1))
        
        if variance_runs > 0:
            z_stat = (n_runs - expected_runs) / np.sqrt(variance_runs)
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        else:
            z_stat = 0
            p_value = 1
        
        return {
            'n_runs': n_runs,
            'expected_runs': expected_runs,
            'z_statistic': z_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    def _abs_autocorr(self, returns: np.ndarray, max_lags: int = 10) -> Dict:
        """
        Analyze autocorrelation of absolute returns.
        """
        abs_returns = np.abs(returns)
        
        autocorrs = []
        for lag in range(1, max_lags + 1):
            corr = np.corrcoef(abs_returns[lag:], abs_returns[:-lag])[0, 1]
            autocorrs.append(corr if not np.isnan(corr) else 0)
        
        # Test if any autocorrelation is significant
        n = len(returns)
        threshold = 1.96 / np.sqrt(n)
        significant_autocorrs = sum(1 for c in autocorrs if abs(c) > threshold)
        
        return {
            'autocorrelations': autocorrs,
            'max_autocorr': max(autocorrs, key=abs),
            'significant_lags': significant_autocorrs,
            'significant': significant_autocorrs >= 2
        }
    
    def _regime_analysis(self, returns: np.ndarray) -> Dict:
        """
        Analyze volatility regimes.
        """
        # Calculate rolling volatility
        window = min(20, len(returns) // 4)
        rolling_vol = pd.Series(returns).rolling(window).std().dropna().values
        
        # Identify regimes using median split
        median_vol = np.median(rolling_vol)
        high_vol_regime = rolling_vol > median_vol
        
        # Calculate regime statistics
        high_vol_returns = returns[window-1:][high_vol_regime]
        low_vol_returns = returns[window-1:][~high_vol_regime]
        
        return {
            'high_vol_mean': np.mean(high_vol_returns) if len(high_vol_returns) > 0 else 0,
            'high_vol_std': np.std(high_vol_returns) if len(high_vol_returns) > 0 else 0,
            'low_vol_mean': np.mean(low_vol_returns) if len(low_vol_returns) > 0 else 0,
            'low_vol_std': np.std(low_vol_returns) if len(low_vol_returns) > 0 else 0,
            'high_vol_proportion': np.mean(high_vol_regime),
            'regime_persistence': np.mean(high_vol_regime[1:] == high_vol_regime[:-1])
        }


class CrossAssetExposureAdjuster:
    """
    Adjust portfolio exposures based on cross-asset correlations.
    """
    
    def __init__(self, max_correlation: float = 0.7):
        """
        Initialize the exposure adjuster.
        
        Args:
            max_correlation: Maximum allowed correlation for position sizing
        """
        self.max_correlation = max_correlation
    
    def adjust_exposures(
        self,
        target_weights: np.ndarray,
        cov_matrix: np.ndarray,
        asset_names: Optional[List[str]] = None
    ) -> Dict:
        """
        Adjust portfolio exposures based on correlation structure.
        
        Args:
            target_weights: Initial target weights
            cov_matrix: Covariance matrix
            asset_names: Optional asset names
            
        Returns:
            Dictionary with adjusted weights and analysis
        """
        target_weights = np.array(target_weights)
        cov_matrix = np.array(cov_matrix)
        
        # Calculate correlation matrix
        corr_matrix = self._cov_to_corr(cov_matrix)
        
        # Identify highly correlated pairs
        high_corr_pairs = []
        n = len(target_weights)
        
        for i in range(n):
            for j in range(i + 1, n):
                if abs(corr_matrix[i, j]) > self.max_correlation:
                    high_corr_pairs.append((i, j, corr_matrix[i, j]))
        
        # Adjust weights to reduce concentration in correlated assets
        adjusted_weights = target_weights.copy()
        
        for i, j, corr in high_corr_pairs:
            # Reduce weights of highly correlated assets
            if adjusted_weights[i] > 0 and adjusted_weights[j] > 0:
                # Both long - reduce the smaller position
                if adjusted_weights[i] < adjusted_weights[j]:
                    reduction = adjusted_weights[i] * (abs(corr) - self.max_correlation) / (1 - self.max_correlation)
                    adjusted_weights[i] -= reduction
                else:
                    reduction = adjusted_weights[j] * (abs(corr) - self.max_correlation) / (1 - self.max_correlation)
                    adjusted_weights[j] -= reduction
        
        # Normalize
        adjusted_weights = adjusted_weights / np.sum(np.abs(adjusted_weights))
        
        # Calculate portfolio statistics
        original_var = target_weights @ cov_matrix @ target_weights
        adjusted_var = adjusted_weights @ cov_matrix @ adjusted_weights
        
        return {
            'original_weights': target_weights,
            'adjusted_weights': adjusted_weights,
            'high_correlation_pairs': high_corr_pairs,
            'original_variance': original_var,
            'adjusted_variance': adjusted_var,
            'variance_reduction': (original_var - adjusted_var) / original_var if original_var > 0 else 0,
            'diversification_improvement': len(high_corr_pairs) == 0 or adjusted_var < original_var
        }
    
    def _cov_to_corr(self, cov: np.ndarray) -> np.ndarray:
        """Convert covariance to correlation."""
        vols = np.sqrt(np.diag(cov))
        vols_inv = np.diag(1 / (vols + 1e-10))
        return vols_inv @ cov @ vols_inv
    
    def calculate_diversification_ratio(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray
    ) -> float:
        """
        Calculate diversification ratio.
        
        DR = (w^T * sigma) / sqrt(w^T * Sigma * w)
        """
        weights = np.array(weights)
        cov_matrix = np.array(cov_matrix)
        
        asset_vols = np.sqrt(np.diag(cov_matrix))
        portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
        
        if portfolio_vol == 0:
            return 1.0
        
        weighted_avg_vol = weights @ asset_vols
        return weighted_avg_vol / portfolio_vol


# Convenience functions
def estimate_covariance(
    returns: np.ndarray,
    method: str = "ewm"
) -> Dict:
    """
    Quick covariance estimation.
    
    Args:
        returns: Returns matrix
        method: Estimation method
        
    Returns:
        Covariance estimate
    """
    config = CovarianceConfig(method=method)
    estimator = DynamicCovarianceEstimator(config)
    return estimator.fit(returns)


def detect_volatility_clustering(returns: np.ndarray) -> Dict:
    """
    Quick volatility clustering detection.
    
    Args:
        returns: Return series
        
    Returns:
        Detection results
    """
    detector = VolatilityClusteringDetector()
    return detector.detect(returns)


def adjust_for_correlation(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
    max_correlation: float = 0.7
) -> np.ndarray:
    """
    Quick correlation adjustment.
    
    Args:
        weights: Target weights
        cov_matrix: Covariance matrix
        max_correlation: Maximum allowed correlation
        
    Returns:
        Adjusted weights
    """
    adjuster = CrossAssetExposureAdjuster(max_correlation)
    result = adjuster.adjust_exposures(weights, cov_matrix)
    return result['adjusted_weights']


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    print("Dynamic Covariance Modeling Demo")
    print("=" * 50)
    
    # Generate correlated returns with GARCH effects
    n_assets = 3
    n_periods = 500
    
    # Base correlation structure
    base_corr = np.array([
        [1.0, 0.7, 0.4],
        [0.7, 1.0, 0.5],
        [0.4, 0.5, 1.0]
    ])
    
    # Generate returns with volatility clustering
    returns = np.random.multivariate_normal(
        np.zeros(n_assets),
        base_corr * 0.0001,
        n_periods
    )
    
    # Add volatility clustering
    for t in range(1, n_periods):
        returns[t] *= (1 + 0.5 * np.abs(returns[t-1]) * 10)
    
    # Estimate covariance
    print("\n1. Exponentially Weighted Covariance:")
    ewm_result = estimate_covariance(returns, method="ewm")
    print(f"   Volatilities: {ewm_result['volatility']}")
    
    print("\n2. GARCH Covariance:")
    garch_result = estimate_covariance(returns, method="garch")
    print(f"   Volatilities: {garch_result['volatility']}")
    
    # Detect volatility clustering
    print("\n3. Volatility Clustering Detection:")
    clustering = detect_volatility_clustering(returns[:, 0])
    print(f"   Clustering Detected: {clustering['overall']['clustering_detected']}")
    print(f"   Strength: {clustering['overall']['clustering_strength']}")
    
    # Adjust for correlation
    print("\n4. Correlation Adjustment:")
    weights = np.array([0.4, 0.4, 0.2])
    adjusted = adjust_for_correlation(weights, ewm_result['covariance'])
    print(f"   Original weights: {weights}")
    print(f"   Adjusted weights: {adjusted}")
