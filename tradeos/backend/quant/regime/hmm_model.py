"""
Hidden Markov Model for Market Regime Detection
===============================================

Implements HMM-based regime classification:
- Bull/Bear/Sideways regime detection
- State transition probabilities
- Regime probability estimation
- Structural break detection

Author: TradeOS Quant Team
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
from scipy import stats
from scipy.special import logsumexp
import warnings

try:
    from hmmlearn import hmm
    HMMLEARN_AVAILABLE = True
except ImportError:
    HMMLEARN_AVAILABLE = False
    warnings.warn("hmmlearn not available. Some HMM features will be limited.")


@dataclass
class HMMConfig:
    """Configuration for HMM models."""
    n_components: int = 3  # Number of regimes
    covariance_type: str = "full"
    n_iter: int = 100
    tol: float = 1e-6
    random_state: Optional[int] = None
    init_params: str = "stmc"  # s=startprob, t=transmat, m=means, c=covars
    params: str = "stmc"


@dataclass
class RegimeResult:
    """Result container for regime detection."""
    regimes: np.ndarray
    regime_probs: np.ndarray
    transition_matrix: np.ndarray
    means: np.ndarray
    covars: np.ndarray
    log_likelihood: float
    aic: float
    bic: float
    convergence: bool
    feature_importance: Optional[Dict] = None


class MarketRegimeHMM:
    """
    Hidden Markov Model for market regime detection.
    
    Identifies market regimes (bull, bear, sideways) based on
    return patterns and other market features.
    """
    
    REGIME_NAMES = {
        3: ['bear', 'sideways', 'bull'],
        4: ['crisis', 'bear', 'sideways', 'bull'],
        5: ['crisis', 'bear', 'sideways', 'bull', 'bubble']
    }
    
    def __init__(self, config: Optional[HMMConfig] = None):
        """
        Initialize the HMM regime detector.
        
        Args:
            config: HMM configuration
        """
        self.config = config or HMMConfig()
        self.model = None
        self.fitted = False
        self.regime_labels = None
    
    def fit(
        self,
        features: np.ndarray,
        returns: Optional[np.ndarray] = None
    ) -> RegimeResult:
        """
        Fit HMM to market data.
        
        Args:
            features: Feature matrix (n_samples x n_features)
            returns: Optional returns for regime labeling
            
        Returns:
            RegimeResult with fitted model parameters
        """
        features = np.array(features)
        
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        
        if HMMLEARN_AVAILABLE:
            return self._fit_hmmlearn(features, returns)
        else:
            return self._fit_manual(features, returns)
    
    def _fit_hmmlearn(
        self,
        features: np.ndarray,
        returns: Optional[np.ndarray]
    ) -> RegimeResult:
        """Fit using hmmlearn."""
        # Initialize model
        model = hmm.GaussianHMM(
            n_components=self.config.n_components,
            covariance_type=self.config.covariance_type,
            n_iter=self.config.n_iter,
            tol=self.config.tol,
            random_state=self.config.random_state,
            init_params=self.config.init_params,
            params=self.config.params
        )
        
        # Fit
        model.fit(features)
        
        # Get regime predictions
        regimes = model.predict(features)
        regime_probs = model.predict_proba(features)
        
        # Label regimes if returns provided
        if returns is not None:
            self.regime_labels = self._label_regimes(regimes, returns)
        else:
            self.regime_labels = self._auto_label_regimes(model.means_)
        
        # Calculate information criteria
        log_likelihood = model.score(features)
        n_params = self._count_parameters(model)
        n_samples = len(features)
        
        aic = 2 * n_params - 2 * log_likelihood
        bic = np.log(n_samples) * n_params - 2 * log_likelihood
        
        self.model = model
        self.fitted = True
        
        return RegimeResult(
            regimes=regimes,
            regime_probs=regime_probs,
            transition_matrix=model.transmat_,
            means=model.means_,
            covars=model.covars_ if hasattr(model, 'covars_') else None,
            log_likelihood=log_likelihood,
            aic=aic,
            bic=bic,
            convergence=model.monitor_.converged if hasattr(model, 'monitor_') else True,
            feature_importance=self._calculate_feature_importance(features, regimes)
        )
    
    def _fit_manual(
        self,
        features: np.ndarray,
        returns: Optional[np.ndarray]
    ) -> RegimeResult:
        """Manual HMM implementation (fallback)."""
        n_samples, n_features = features.shape
        n_components = self.config.n_components
        
        # Initialize parameters
        np.random.seed(self.config.random_state)
        
        # Start probabilities (uniform)
        startprob = np.ones(n_components) / n_components
        
        # Transition matrix (random but valid)
        transmat = np.random.dirichlet(np.ones(n_components), n_components)
        
        # Means (K-means initialization)
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_components, random_state=self.config.random_state)
        regimes = kmeans.fit_predict(features)
        means = kmeans.cluster_centers_
        
        # Covariances
        covars = np.array([np.cov(features[regimes == i].T) + np.eye(n_features) * 0.01 
                          for i in range(n_components)])
        
        # EM algorithm
        log_likelihood = -np.inf
        converged = False
        
        for iteration in range(self.config.n_iter):
            # E-step
            log_prob, fwd_lattice = self._forward_pass(features, startprob, transmat, means, covars)
            bwd_lattice = self._backward_pass(features, transmat, means, covars)
            
            # Compute posterior probabilities
            log_gamma = fwd_lattice + bwd_lattice
            log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
            gamma = np.exp(log_gamma)
            
            # M-step
            # Update start probabilities
            startprob = gamma[0] / np.sum(gamma[0])
            
            # Update transition matrix
            for i in range(n_components):
                for j in range(n_components):
                    xi_sum = 0
                    gamma_sum = 0
                    for t in range(n_samples - 1):
                        xi = fwd_lattice[t, i] + np.log(transmat[i, j] + 1e-10)
                        xi += self._log_multivariate_normal(features[t+1], means[j], covars[j])
                        xi += bwd_lattice[t+1, j]
                        xi_sum += np.exp(xi - log_prob)
                        gamma_sum += gamma[t, i]
                    transmat[i, j] = xi_sum / (gamma_sum + 1e-10)
            
            # Normalize transition matrix
            transmat = transmat / transmat.sum(axis=1, keepdims=True)
            
            # Update means
            for i in range(n_components):
                means[i] = np.sum(gamma[:, i:i+1] * features, axis=0) / (np.sum(gamma[:, i]) + 1e-10)
            
            # Update covariances
            for i in range(n_components):
                diff = features - means[i]
                covars[i] = (gamma[:, i:i+1].T @ (diff * diff)) / (np.sum(gamma[:, i]) + 1e-10)
                covars[i] += np.eye(n_features) * 0.001  # Regularization
            
            # Check convergence
            new_log_likelihood = self._compute_log_likelihood(features, startprob, transmat, means, covars)
            
            if abs(new_log_likelihood - log_likelihood) < self.config.tol:
                converged = True
                break
            
            log_likelihood = new_log_likelihood
        
        # Final predictions
        _, fwd_lattice = self._forward_pass(features, startprob, transmat, means, covars)
        regimes = np.argmax(fwd_lattice, axis=1)
        
        # Compute regime probabilities
        log_prob, fwd_lattice = self._forward_pass(features, startprob, transmat, means, covars)
        bwd_lattice = self._backward_pass(features, transmat, means, covars)
        log_gamma = fwd_lattice + bwd_lattice
        log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
        regime_probs = np.exp(log_gamma)
        
        # Label regimes
        if returns is not None:
            self.regime_labels = self._label_regimes(regimes, returns)
        else:
            self.regime_labels = self._auto_label_regimes(means)
        
        # Information criteria
        n_params = n_components * (n_features + n_features * (n_features + 1) / 2 + n_components - 1)
        aic = 2 * n_params - 2 * log_likelihood
        bic = np.log(n_samples) * n_params - 2 * log_likelihood
        
        return RegimeResult(
            regimes=regimes,
            regime_probs=regime_probs,
            transition_matrix=transmat,
            means=means,
            covars=covars,
            log_likelihood=log_likelihood,
            aic=aic,
            bic=bic,
            convergence=converged
        )
    
    def _forward_pass(
        self,
        features: np.ndarray,
        startprob: np.ndarray,
        transmat: np.ndarray,
        means: np.ndarray,
        covars: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """Forward pass of HMM."""
        n_samples = len(features)
        n_components = len(startprob)
        
        fwd_lattice = np.zeros((n_samples, n_components))
        
        # Initialize
        for i in range(n_components):
            fwd_lattice[0, i] = np.log(startprob[i] + 1e-10) + self._log_multivariate_normal(
                features[0], means[i], covars[i]
            )
        
        # Recursion
        for t in range(1, n_samples):
            for j in range(n_components):
                log_probs = fwd_lattice[t-1] + np.log(transmat[:, j] + 1e-10)
                fwd_lattice[t, j] = logsumexp(log_probs) + self._log_multivariate_normal(
                    features[t], means[j], covars[j]
                )
        
        log_prob = logsumexp(fwd_lattice[-1])
        
        return log_prob, fwd_lattice
    
    def _backward_pass(
        self,
        features: np.ndarray,
        transmat: np.ndarray,
        means: np.ndarray,
        covars: np.ndarray
    ) -> np.ndarray:
        """Backward pass of HMM."""
        n_samples = len(features)
        n_components = len(means)
        
        bwd_lattice = np.zeros((n_samples, n_components))
        
        # Initialize
        bwd_lattice[-1] = 0
        
        # Recursion
        for t in range(n_samples - 2, -1, -1):
            for i in range(n_components):
                log_probs = []
                for j in range(n_components):
                    log_prob = (np.log(transmat[i, j] + 1e-10) + 
                               self._log_multivariate_normal(features[t+1], means[j], covars[j]) +
                               bwd_lattice[t+1, j])
                    log_probs.append(log_prob)
                bwd_lattice[t, i] = logsumexp(log_probs)
        
        return bwd_lattice
    
    def _log_multivariate_normal(
        self,
        x: np.ndarray,
        mean: np.ndarray,
        cov: np.ndarray
    ) -> float:
        """Compute log of multivariate normal PDF."""
        n = len(x)
        diff = x - mean
        
        try:
            inv_cov = np.linalg.inv(cov)
            sign, logdet = np.linalg.slogdet(cov)
            
            if sign <= 0:
                # Add regularization
                cov = cov + np.eye(n) * 0.001
                inv_cov = np.linalg.inv(cov)
                sign, logdet = np.linalg.slogdet(cov)
            
            log_prob = -0.5 * (n * np.log(2 * np.pi) + logdet + diff @ inv_cov @ diff)
            return log_prob
        except:
            return -1e10
    
    def _compute_log_likelihood(
        self,
        features: np.ndarray,
        startprob: np.ndarray,
        transmat: np.ndarray,
        means: np.ndarray,
        covars: np.ndarray
    ) -> float:
        """Compute total log likelihood."""
        log_prob, _ = self._forward_pass(features, startprob, transmat, means, covars)
        return log_prob
    
    def _count_parameters(self, model) -> int:
        """Count number of parameters in HMM."""
        n_components = model.n_components
        n_features = model.means_.shape[1]
        
        # Start probabilities
        n_params = n_components - 1
        
        # Transition matrix
        n_params += n_components * (n_components - 1)
        
        # Means
        n_params += n_components * n_features
        
        # Covariances (full)
        n_params += n_components * n_features * (n_features + 1) // 2
        
        return int(n_params)
    
    def _label_regimes(
        self,
        regimes: np.ndarray,
        returns: np.ndarray
    ) -> Dict[int, str]:
        """Label regimes based on return characteristics."""
        labels = {}
        regime_returns = {}
        
        for i in range(self.config.n_components):
            mask = regimes == i
            if np.sum(mask) > 0:
                regime_returns[i] = returns[mask]
        
        # Sort regimes by mean return
        sorted_regimes = sorted(
            regime_returns.items(),
            key=lambda x: np.mean(x[1])
        )
        
        # Assign labels
        available_labels = self.REGIME_NAMES.get(
            self.config.n_components,
            [f'regime_{i}' for i in range(self.config.n_components)]
        )
        
        for idx, (regime_id, _) in enumerate(sorted_regimes):
            labels[regime_id] = available_labels[idx]
        
        return labels
    
    def _auto_label_regimes(self, means: np.ndarray) -> Dict[int, str]:
        """Auto-label regimes based on mean feature values."""
        labels = {}
        
        # Use first feature (typically returns) for labeling
        if means.ndim > 1:
            mean_values = means[:, 0]
        else:
            mean_values = means
        
        sorted_indices = np.argsort(mean_values)
        available_labels = self.REGIME_NAMES.get(
            self.config.n_components,
            [f'regime_{i}' for i in range(self.config.n_components)]
        )
        
        for idx, regime_id in enumerate(sorted_indices):
            labels[regime_id] = available_labels[idx]
        
        return labels
    
    def _calculate_feature_importance(
        self,
        features: np.ndarray,
        regimes: np.ndarray
    ) -> Dict:
        """Calculate feature importance for regime discrimination."""
        n_features = features.shape[1]
        importance = {}
        
        for i in range(n_features):
            # ANOVA F-statistic
            groups = [features[regimes == j, i] for j in range(self.config.n_components)]
            f_stat, p_value = stats.f_oneway(*groups)
            importance[f'feature_{i}'] = {'f_statistic': f_stat, 'p_value': p_value}
        
        return importance
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict regimes for new data.
        
        Args:
            features: Feature matrix
            
        Returns:
            Regime predictions
        """
        if not self.fitted and HMMLEARN_AVAILABLE:
            raise ValueError("Model must be fitted before prediction")
        
        features = np.array(features)
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        
        if HMMLEARN_AVAILABLE and self.model is not None:
            return self.model.predict(features)
        else:
            # Manual prediction
            # Find closest mean
            distances = np.array([
                np.sum((features - mean)**2, axis=1)
                for mean in self.model.means_ if hasattr(self.model, 'means_')
            ])
            return np.argmin(distances, axis=0)
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Predict regime probabilities for new data.
        
        Args:
            features: Feature matrix
            
        Returns:
            Regime probabilities
        """
        if not self.fitted and HMMLEARN_AVAILABLE:
            raise ValueError("Model must be fitted before prediction")
        
        features = np.array(features)
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        
        if HMMLEARN_AVAILABLE and self.model is not None:
            return self.model.predict_proba(features)
        else:
            # Manual probability calculation
            # Simplified - use softmax of negative distances
            distances = np.array([
                np.sum((features - mean)**2, axis=1)
                for mean in self.model.means_ if hasattr(self.model, 'means_')
            ])
            log_probs = -distances / 2
            log_probs = log_probs - logsumexp(log_probs, axis=0, keepdims=True)
            return np.exp(log_probs).T
    
    def get_regime_statistics(
        self,
        features: np.ndarray,
        returns: np.ndarray
    ) -> Dict:
        """
        Get statistics for each detected regime.
        
        Args:
            features: Feature matrix
            returns: Returns series
            
        Returns:
            Dictionary with regime statistics
        """
        regimes = self.predict(features)
        stats_dict = {}
        
        for i in range(self.config.n_components):
            mask = regimes == i
            if np.sum(mask) > 0:
                regime_returns = returns[mask]
                
                label = self.regime_labels.get(i, f'regime_{i}')
                
                stats_dict[label] = {
                    'n_periods': int(np.sum(mask)),
                    'mean_return': float(np.mean(regime_returns)),
                    'std_return': float(np.std(regime_returns)),
                    'sharpe': float(np.mean(regime_returns) / (np.std(regime_returns) + 1e-10)),
                    'max_return': float(np.max(regime_returns)),
                    'min_return': float(np.min(regime_returns)),
                    'skewness': float(stats.skew(regime_returns)),
                    'kurtosis': float(stats.kurtosis(regime_returns)),
                    'probability': float(np.mean(mask))
                }
        
        return stats_dict


class StructuralBreakDetector:
    """
    Detect structural breaks in time series data.
    
    Uses multiple methods including:
    - Chow test
    - CUSUM test
    - Bai-Perron test (simplified)
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize detector.
        
        Args:
            significance_level: Significance level for tests
        """
        self.significance_level = significance_level
    
    def detect_breaks(
        self,
        data: np.ndarray,
        max_breaks: int = 5
    ) -> Dict:
        """
        Detect structural breaks in data.
        
        Args:
            data: Time series data
            max_breaks: Maximum number of breaks to detect
            
        Returns:
            Dictionary with break points and statistics
        """
        data = np.array(data).flatten()
        n = len(data)
        
        breaks = []
        
        # CUSUM test
        cusum_result = self._cusum_test(data)
        if cusum_result['significant']:
            breaks.append({
                'method': 'cusum',
                'index': cusum_result['break_point'],
                'statistic': cusum_result['statistic'],
                'p_value': cusum_result['p_value']
            })
        
        # Recursive break detection
        remaining_breaks = max_breaks - len(breaks)
        if remaining_breaks > 0:
            recursive_breaks = self._recursive_breaks(data, remaining_breaks)
            breaks.extend(recursive_breaks)
        
        # Sort by index
        breaks = sorted(breaks, key=lambda x: x['index'])
        
        return {
            'break_points': [b['index'] for b in breaks],
            'break_details': breaks,
            'n_breaks': len(breaks),
            'has_breaks': len(breaks) > 0
        }
    
    def _cusum_test(self, data: np.ndarray) -> Dict:
        """CUSUM test for structural breaks."""
        n = len(data)
        
        # Calculate cumulative sum of standardized residuals
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return {'significant': False, 'break_point': 0, 'statistic': 0, 'p_value': 1}
        
        standardized = (data - mean) / std
        cusum = np.cumsum(standardized)
        
        # Find maximum deviation
        max_deviation = np.max(np.abs(cusum))
        break_point = np.argmax(np.abs(cusum))
        
        # Critical value (approximate)
        critical_value = 1.143 * np.sqrt(n)
        
        # P-value (approximate)
        p_value = 2 * (1 - stats.norm.cdf(max_deviation / np.sqrt(n)))
        
        return {
            'significant': max_deviation > critical_value,
            'break_point': int(break_point),
            'statistic': float(max_deviation),
            'p_value': float(p_value),
            'critical_value': float(critical_value)
        }
    
    def _recursive_breaks(
        self,
        data: np.ndarray,
        max_breaks: int
    ) -> List[Dict]:
        """Recursively detect break points."""
        breaks = []
        n = len(data)
        
        if n < 20:
            return breaks
        
        # Find best break point using F-statistic
        best_f_stat = 0
        best_break = 0
        
        for i in range(int(0.1 * n), int(0.9 * n)):
            # Split data
            data1 = data[:i]
            data2 = data[i:]
            
            # Calculate F-statistic (Chow test)
            rss_pooled = np.sum((data - np.mean(data))**2)
            rss1 = np.sum((data1 - np.mean(data1))**2)
            rss2 = np.sum((data2 - np.mean(data2))**2)
            
            k = 1  # Number of parameters
            f_stat = ((rss_pooled - rss1 - rss2) / k) / ((rss1 + rss2) / (n - 2 * k))
            
            if f_stat > best_f_stat:
                best_f_stat = f_stat
                best_break = i
        
        # Check significance
        p_value = 1 - stats.f.cdf(best_f_stat, 1, n - 2)
        
        if p_value < self.significance_level:
            breaks.append({
                'method': 'chow',
                'index': best_break,
                'statistic': float(best_f_stat),
                'p_value': float(p_value)
            })
            
            # Recursively search subsegments
            if max_breaks > 1:
                left_breaks = self._recursive_breaks(data[:best_break], (max_breaks - 1) // 2)
                right_breaks = self._recursive_breaks(data[best_break:], (max_breaks - 1) // 2)
                
                for b in left_breaks:
                    b['index'] = b['index']
                for b in right_breaks:
                    b['index'] = best_break + b['index']
                
                breaks.extend(left_breaks)
                breaks.extend(right_breaks)
        
        return breaks


class RegimeSwitchingStrategy:
    """
    Strategy that adapts based on detected market regime.
    """
    
    def __init__(
        self,
        regime_detector: MarketRegimeHMM,
        regime_strategies: Dict[str, Any]
    ):
        """
        Initialize regime-switching strategy.
        
        Args:
            regime_detector: Trained regime detector
            regime_strategies: Dictionary mapping regime names to strategies
        """
        self.regime_detector = regime_detector
        self.regime_strategies = regime_strategies
        self.current_regime = None
    
    def generate_signal(
        self,
        features: np.ndarray,
        current_position: float = 0
    ) -> Dict:
        """
        Generate trading signal based on current regime.
        
        Args:
            features: Current market features
            current_position: Current position
            
        Returns:
            Trading signal
        """
        # Detect regime
        regime_probs = self.regime_detector.predict_proba(features.reshape(1, -1))[0]
        regime = np.argmax(regime_probs)
        
        if self.regime_detector.regime_labels:
            regime_name = self.regime_detector.regime_labels.get(regime, f'regime_{regime}')
        else:
            regime_name = f'regime_{regime}'
        
        self.current_regime = regime_name
        
        # Get strategy for regime
        strategy = self.regime_strategies.get(regime_name, self.regime_strategies.get('default'))
        
        if strategy is None:
            return {'signal': 0, 'regime': regime_name, 'confidence': regime_probs[regime]}
        
        # Generate signal from strategy
        signal = strategy(features, current_position)
        
        return {
            'signal': signal,
            'regime': regime_name,
            'regime_confidence': regime_probs[regime],
            'regime_probs': regime_probs
        }


# Convenience functions
def detect_regimes(
    features: np.ndarray,
    returns: Optional[np.ndarray] = None,
    n_regimes: int = 3
) -> RegimeResult:
    """
    Quick regime detection.
    
    Args:
        features: Feature matrix
        returns: Optional returns for labeling
        n_regimes: Number of regimes
        
    Returns:
        Regime detection result
    """
    config = HMMConfig(n_components=n_regimes)
    detector = MarketRegimeHMM(config)
    return detector.fit(features, returns)


def detect_structural_breaks(
    data: np.ndarray,
    max_breaks: int = 5
) -> Dict:
    """
    Quick structural break detection.
    
    Args:
        data: Time series data
        max_breaks: Maximum number of breaks
        
    Returns:
        Break detection results
    """
    detector = StructuralBreakDetector()
    return detector.detect_breaks(data, max_breaks)


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    print("Market Regime Detection Demo")
    print("=" * 50)
    
    # Generate synthetic regime data
    n_samples = 500
    
    # Regime 1: Bear market (negative returns, high volatility)
    bear_returns = np.random.normal(-0.001, 0.03, n_samples // 3)
    
    # Regime 2: Sideways (near-zero returns, medium volatility)
    sideways_returns = np.random.normal(0.0002, 0.015, n_samples // 3)
    
    # Regime 3: Bull market (positive returns, low volatility)
    bull_returns = np.random.normal(0.001, 0.01, n_samples // 3)
    
    returns = np.concatenate([bear_returns, sideways_returns, bull_returns])
    
    # Create features
    features = np.column_stack([
        returns,
        np.abs(returns),  # Absolute returns
        pd.Series(returns).rolling(5).std().fillna(0).values,  # Rolling volatility
        pd.Series(returns).rolling(10).mean().fillna(0).values  # Rolling mean
    ])
    
    # Detect regimes
    print("\n1. HMM Regime Detection:")
    result = detect_regimes(features, returns, n_regimes=3)
    print(f"   Log-likelihood: {result.log_likelihood:.2f}")
    print(f"   AIC: {result.aic:.2f}, BIC: {result.bic:.2f}")
    print(f"   Transition Matrix:\n{result.transition_matrix}")
    
    # Regime statistics
    if HMMLEARN_AVAILABLE:
        hmm_detector = MarketRegimeHMM(HMMConfig(n_components=3))
        hmm_detector.fit(features, returns)
        stats = hmm_detector.get_regime_statistics(features, returns)
        print(f"\n   Regime Statistics:")
        for regime, stat in stats.items():
            print(f"   {regime}: mean={stat['mean_return']:.4f}, "
                  f"std={stat['std_return']:.4f}, "
                  f"sharpe={stat['sharpe']:.2f}")
    
    # Structural break detection
    print("\n2. Structural Break Detection:")
    breaks = detect_structural_breaks(returns)
    print(f"   Break points: {breaks['break_points']}")
    print(f"   Number of breaks: {breaks['n_breaks']}")
