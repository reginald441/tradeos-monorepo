"""
Bayesian Inference Engine for TradeOS
======================================

Implements Bayesian methods for trading:
- Bayesian probability updating
- Signal confidence estimation
- Regime probability scoring
- Bayesian optimization for parameters
- PyMC3/Pyro integration patterns

Author: TradeOS Quant Team
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Callable, Union, Any
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import minimize
from scipy.special import logsumexp
import warnings

# Optional imports for advanced Bayesian libraries
try:
    import pymc as pm
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False

try:
    import torch
    import pyro
    import pyro.distributions as dist
    from pyro.infer import SVI, Trace_ELBO
    from pyro.optim import Adam
    PYRO_AVAILABLE = True
except ImportError:
    PYRO_AVAILABLE = False


@dataclass
class BayesianConfig:
    """Configuration for Bayesian inference."""
    prior_strength: float = 1.0
    credibility_interval: float = 0.95
    n_samples: int = 10000
    n_chains: int = 4
    random_seed: Optional[int] = None
    use_mcmc: bool = True
    use_vi: bool = False  # Variational inference


class BayesianProbabilityUpdater:
    """
    Bayesian probability updating for trading signals.
    
    Implements conjugate prior updates for common distributions
    and numerical methods for non-conjugate cases.
    """
    
    def __init__(self, config: Optional[BayesianConfig] = None):
        """Initialize the Bayesian updater."""
        self.config = config or BayesianConfig()
        
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
    
    def update_beta_binomial(
        self,
        prior_alpha: float,
        prior_beta: float,
        successes: int,
        trials: int
    ) -> Dict:
        """
        Update Beta-Binomial model (e.g., win rate estimation).
        
        Args:
            prior_alpha: Prior alpha parameter
            prior_beta: Prior beta parameter
            successes: Number of successes observed
            trials: Total number of trials
            
        Returns:
            Dictionary with posterior parameters and statistics
        """
        posterior_alpha = prior_alpha + successes
        posterior_beta = prior_beta + (trials - successes)
        
        # Posterior statistics
        mean = posterior_alpha / (posterior_alpha + posterior_beta)
        mode = (posterior_alpha - 1) / (posterior_alpha + posterior_beta - 2) if posterior_alpha > 1 and posterior_beta > 1 else mean
        variance = (posterior_alpha * posterior_beta) / ((posterior_alpha + posterior_beta)**2 * (posterior_alpha + posterior_beta + 1))
        
        # Credibility interval
        ci_lower = stats.beta.ppf((1 - self.config.credibility_interval) / 2, posterior_alpha, posterior_beta)
        ci_upper = stats.beta.ppf((1 + self.config.credibility_interval) / 2, posterior_alpha, posterior_beta)
        
        return {
            'posterior_alpha': posterior_alpha,
            'posterior_beta': posterior_beta,
            'mean': mean,
            'mode': mode,
            'variance': variance,
            'std': np.sqrt(variance),
            'credibility_interval': (ci_lower, ci_upper),
            'probability_above_half': 1 - stats.beta.cdf(0.5, posterior_alpha, posterior_beta)
        }
    
    def update_normal_normal(
        self,
        prior_mean: float,
        prior_variance: float,
        data: np.ndarray,
        known_variance: Optional[float] = None
    ) -> Dict:
        """
        Update Normal-Normal model (e.g., return estimation).
        
        Args:
            prior_mean: Prior mean
            prior_variance: Prior variance
            data: Observed data
            known_variance: Known data variance (if None, estimated from data)
            
        Returns:
            Dictionary with posterior parameters
        """
        data = np.array(data)
        n = len(data)
        sample_mean = np.mean(data)
        
        if known_variance is None:
            data_variance = np.var(data, ddof=1) if n > 1 else prior_variance
        else:
            data_variance = known_variance
        
        # Posterior parameters
        posterior_variance = 1 / (1 / prior_variance + n / data_variance)
        posterior_mean = posterior_variance * (prior_mean / prior_variance + n * sample_mean / data_variance)
        
        # Credibility interval
        ci_lower = stats.norm.ppf((1 - self.config.credibility_interval) / 2, posterior_mean, np.sqrt(posterior_variance))
        ci_upper = stats.norm.ppf((1 + self.config.credibility_interval) / 2, posterior_mean, np.sqrt(posterior_variance))
        
        return {
            'posterior_mean': posterior_mean,
            'posterior_variance': posterior_variance,
            'posterior_std': np.sqrt(posterior_variance),
            'credibility_interval': (ci_lower, ci_upper),
            'probability_positive': 1 - stats.norm.cdf(0, posterior_mean, np.sqrt(posterior_variance))
        }
    
    def update_gamma_poisson(
        self,
        prior_alpha: float,
        prior_beta: float,
        counts: np.ndarray
    ) -> Dict:
        """
        Update Gamma-Poisson model (e.g., trade frequency).
        
        Args:
            prior_alpha: Prior alpha parameter
            prior_beta: Prior beta parameter
            counts: Observed counts
            
        Returns:
            Dictionary with posterior parameters
        """
        counts = np.array(counts)
        n = len(counts)
        sum_counts = np.sum(counts)
        
        posterior_alpha = prior_alpha + sum_counts
        posterior_beta = prior_beta + n
        
        mean = posterior_alpha / posterior_beta
        variance = posterior_alpha / (posterior_beta ** 2)
        
        # Credibility interval (Gamma distribution)
        ci_lower = stats.gamma.ppf((1 - self.config.credibility_interval) / 2, posterior_alpha, scale=1/posterior_beta)
        ci_upper = stats.gamma.ppf((1 + self.config.credibility_interval) / 2, posterior_alpha, scale=1/posterior_beta)
        
        return {
            'posterior_alpha': posterior_alpha,
            'posterior_beta': posterior_beta,
            'mean': mean,
            'variance': variance,
            'std': np.sqrt(variance),
            'credibility_interval': (ci_lower, ci_upper)
        }
    
    def update_dirichlet_multinomial(
        self,
        prior_alphas: np.ndarray,
        counts: np.ndarray
    ) -> Dict:
        """
        Update Dirichlet-Multinomial model (e.g., regime probabilities).
        
        Args:
            prior_alphas: Prior alpha parameters for each category
            counts: Observed counts for each category
            
        Returns:
            Dictionary with posterior parameters
        """
        prior_alphas = np.array(prior_alphas)
        counts = np.array(counts)
        
        posterior_alphas = prior_alphas + counts
        
        # Expected probabilities
        expected_probs = posterior_alphas / np.sum(posterior_alphas)
        
        # Variance for each probability
        alpha_sum = np.sum(posterior_alphas)
        variances = (posterior_alphas * (alpha_sum - posterior_alphas)) / (alpha_sum**2 * (alpha_sum + 1))
        
        return {
            'posterior_alphas': posterior_alphas,
            'expected_probabilities': expected_probs,
            'variances': variances,
            'stds': np.sqrt(variances)
        }


class SignalConfidenceEstimator:
    """
    Estimate confidence in trading signals using Bayesian methods.
    """
    
    def __init__(self, config: Optional[BayesianConfig] = None):
        """Initialize the confidence estimator."""
        self.config = config or BayesianConfig()
        self.updater = BayesianProbabilityUpdater(self.config)
    
    def estimate_signal_confidence(
        self,
        signal_history: List[Tuple[float, float]],  # (signal, outcome) pairs
        current_signal: float,
        signal_threshold: float = 0.0
    ) -> Dict:
        """
        Estimate confidence in a trading signal.
        
        Args:
            signal_history: List of (signal_value, outcome) tuples
            current_signal: Current signal value
            signal_threshold: Threshold for positive signal
            
        Returns:
            Dictionary with confidence metrics
        """
        if len(signal_history) < 10:
            return {
                'confidence': 0.5,
                'probability_success': 0.5,
                'credibility_interval': (0, 1),
                'reliability': 'insufficient_data'
            }
        
        # Separate signals into positive and negative
        positive_outcomes = [outcome for sig, outcome in signal_history if sig > signal_threshold]
        negative_outcomes = [outcome for sig, outcome in signal_history if sig <= signal_threshold]
        
        # Calculate success rates
        positive_success = sum(1 for o in positive_outcomes if o > 0)
        negative_success = sum(1 for o in negative_outcomes if o > 0)
        
        # Update Beta priors
        positive_posterior = self.updater.update_beta_binomial(
            1, 1, positive_success, len(positive_outcomes) if positive_outcomes else 1
        )
        negative_posterior = self.updater.update_beta_binomial(
            1, 1, negative_success, len(negative_outcomes) if negative_outcomes else 1
        )
        
        # Determine which signal type we're evaluating
        if current_signal > signal_threshold:
            posterior = positive_posterior
            signal_type = 'positive'
        else:
            posterior = negative_posterior
            signal_type = 'negative'
        
        # Calculate confidence score
        confidence = posterior['probability_above_half']
        
        # Reliability based on sample size
        n_samples = len(positive_outcomes) if signal_type == 'positive' else len(negative_outcomes)
        if n_samples < 20:
            reliability = 'low'
        elif n_samples < 50:
            reliability = 'medium'
        else:
            reliability = 'high'
        
        return {
            'confidence': confidence,
            'probability_success': posterior['mean'],
            'credibility_interval': posterior['credibility_interval'],
            'reliability': reliability,
            'signal_type': signal_type,
            'n_samples': n_samples,
            'posterior_alpha': posterior['posterior_alpha'],
            'posterior_beta': posterior['posterior_beta']
        }
    
    def kalman_filter_signal(
        self,
        observations: np.ndarray,
        initial_state: float = 0.0,
        initial_variance: float = 1.0,
        process_variance: float = 0.01,
        measurement_variance: float = 0.1
    ) -> Dict:
        """
        Apply Kalman filter to smooth signal estimates.
        
        Args:
            observations: Noisy signal observations
            initial_state: Initial state estimate
            initial_variance: Initial state variance
            process_variance: Process noise variance
            measurement_variance: Measurement noise variance
            
        Returns:
            Dictionary with filtered estimates
        """
        observations = np.array(observations)
        n = len(observations)
        
        # Initialize
        states = np.zeros(n)
        variances = np.zeros(n)
        
        states[0] = initial_state
        variances[0] = initial_variance
        
        # Kalman filter recursion
        for t in range(1, n):
            # Prediction
            predicted_state = states[t-1]
            predicted_variance = variances[t-1] + process_variance
            
            # Update
            kalman_gain = predicted_variance / (predicted_variance + measurement_variance)
            states[t] = predicted_state + kalman_gain * (observations[t] - predicted_state)
            variances[t] = (1 - kalman_gain) * predicted_variance
        
        return {
            'filtered_states': states,
            'state_variances': variances,
            'confidence_intervals': 1.96 * np.sqrt(variances)
        }


class RegimeProbabilityScorer:
    """
    Score probabilities of different market regimes using Bayesian methods.
    """
    
    def __init__(
        self,
        regimes: List[str],
        config: Optional[BayesianConfig] = None
    ):
        """
        Initialize regime scorer.
        
        Args:
            regimes: List of regime names
            config: Bayesian configuration
        """
        self.regimes = regimes
        self.n_regimes = len(regimes)
        self.config = config or BayesianConfig()
        self.updater = BayesianProbabilityUpdater(self.config)
        
        # Initialize uniform prior
        self.regime_alphas = np.ones(self.n_regimes)
    
    def update_regime_probabilities(
        self,
        regime_observations: np.ndarray,
        features: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Update regime probabilities based on observations.
        
        Args:
            regime_observations: Observed regime indicators (one-hot or indices)
            features: Additional features for context (optional)
            
        Returns:
            Dictionary with regime probabilities
        """
        # Count observations
        if len(regime_observations.shape) == 1:
            # Indices
            counts = np.bincount(regime_observations, minlength=self.n_regimes)
        else:
            # One-hot encoded
            counts = np.sum(regime_observations, axis=0)
        
        # Update Dirichlet
        posterior = self.updater.update_dirichlet_multinomial(
            self.regime_alphas, counts
        )
        
        self.regime_alphas = posterior['posterior_alphas']
        
        return {
            'regime_probabilities': posterior['expected_probabilities'],
            'regime_uncertainties': posterior['stds'],
            'most_likely_regime': self.regimes[np.argmax(posterior['expected_probabilities'])],
            'entropy': -np.sum(posterior['expected_probabilities'] * np.log(posterior['expected_probabilities'] + 1e-10))
        }
    
    def predict_regime_transition(
        self,
        transition_history: np.ndarray,
        current_regime: int
    ) -> Dict:
        """
        Predict probability of transitioning to each regime.
        
        Args:
            transition_history: Historical regime transitions
            current_regime: Current regime index
            
        Returns:
            Dictionary with transition probabilities
        """
        # Build transition count matrix
        transition_counts = np.zeros((self.n_regimes, self.n_regimes))
        
        for i in range(len(transition_history) - 1):
            from_regime = int(transition_history[i])
            to_regime = int(transition_history[i + 1])
            transition_counts[from_regime, to_regime] += 1
        
        # Add prior (Laplace smoothing)
        transition_counts += 1
        
        # Calculate transition probabilities from current regime
        row = transition_counts[current_regime]
        transition_probs = row / np.sum(row)
        
        return {
            'transition_probabilities': transition_probs,
            'most_likely_next': self.regimes[np.argmax(transition_probs)],
            'expected_duration': 1 / (1 - transition_counts[current_regime, current_regime] / np.sum(row))
        }


class BayesianOptimizer:
    """
    Bayesian Optimization for hyperparameter tuning.
    
    Uses Gaussian Process surrogate model with acquisition functions.
    """
    
    def __init__(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        n_initial_points: int = 5,
        acquisition: str = 'ei',  # 'ei', 'ucb', 'poi'
        xi: float = 0.01,
        kappa: float = 2.576
    ):
        """
        Initialize Bayesian optimizer.
        
        Args:
            param_bounds: Dictionary of parameter names to (min, max) bounds
            n_initial_points: Number of random initial points
            acquisition: Acquisition function type
            xi: Exploration parameter for EI
            kappa: Exploration parameter for UCB
        """
        self.param_bounds = param_bounds
        self.param_names = list(param_bounds.keys())
        self.n_params = len(param_bounds)
        self.n_initial_points = n_initial_points
        self.acquisition = acquisition
        self.xi = xi
        self.kappa = kappa
        
        # Storage
        self.X = []
        self.y = []
        self.best_y = -np.inf
        self.best_x = None
    
    def _kernel(self, x1: np.ndarray, x2: np.ndarray, length_scale: float = 1.0) -> float:
        """RBF kernel function."""
        return np.exp(-0.5 * np.sum((x1 - x2)**2) / length_scale**2)
    
    def _build_covariance_matrix(self, X: List[np.ndarray], noise: float = 1e-5) -> np.ndarray:
        """Build covariance matrix for GP."""
        n = len(X)
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = self._kernel(X[i], X[j])
        K += noise * np.eye(n)
        return K
    
    def _gp_predict(
        self,
        x: np.ndarray,
        return_std: bool = True
    ) -> Tuple[float, float]:
        """Make prediction with Gaussian Process."""
        if len(self.X) == 0:
            return 0.0, 1.0
        
        K = self._build_covariance_matrix(self.X)
        k_star = np.array([self._kernel(x, xi) for xi in self.X])
        
        try:
            K_inv = np.linalg.inv(K)
        except np.linalg.LinAlgError:
            K_inv = np.linalg.pinv(K)
        
        mu = k_star @ K_inv @ np.array(self.y)
        
        if return_std:
            k_star_star = self._kernel(x, x)
            sigma2 = k_star_star - k_star @ K_inv @ k_star
            sigma = np.sqrt(max(0, sigma2))
            return mu, sigma
        
        return mu
    
    def _acquisition_function(self, x: np.ndarray) -> float:
        """Evaluate acquisition function."""
        mu, sigma = self._gp_predict(x)
        
        if self.acquisition == 'ei':  # Expected Improvement
            if sigma == 0:
                return 0
            improvement = mu - self.best_y - self.xi
            z = improvement / sigma
            ei = improvement * stats.norm.cdf(z) + sigma * stats.norm.pdf(z)
            return ei
        
        elif self.acquisition == 'ucb':  # Upper Confidence Bound
            return mu + self.kappa * sigma
        
        elif self.acquisition == 'poi':  # Probability of Improvement
            if sigma == 0:
                return 0
            z = (mu - self.best_y - self.xi) / sigma
            return stats.norm.cdf(z)
        
        return mu
    
    def suggest_next_point(self) -> Dict[str, float]:
        """Suggest next point to evaluate."""
        # Random sampling for initial points
        if len(self.X) < self.n_initial_points:
            x = np.array([
                np.random.uniform(self.param_bounds[p][0], self.param_bounds[p][1])
                for p in self.param_names
            ])
        else:
            # Optimize acquisition function
            from scipy.optimize import differential_evolution
            
            bounds = [self.param_bounds[p] for p in self.param_names]
            
            result = differential_evolution(
                lambda x: -self._acquisition_function(x),
                bounds,
                maxiter=100,
                seed=42
            )
            x = result.x
        
        return {name: x[i] for i, name in enumerate(self.param_names)}
    
    def tell(self, params: Dict[str, float], value: float):
        """Report evaluation result."""
        x = np.array([params[p] for p in self.param_names])
        self.X.append(x)
        self.y.append(value)
        
        if value > self.best_y:
            self.best_y = value
            self.best_x = params.copy()
    
    def optimize(
        self,
        objective_func: Callable,
        n_iterations: int = 50
    ) -> Dict:
        """
        Run Bayesian optimization.
        
        Args:
            objective_func: Function to optimize (takes dict, returns float)
            n_iterations: Number of optimization iterations
            
        Returns:
            Dictionary with best parameters and history
        """
        for i in range(n_iterations):
            # Suggest next point
            params = self.suggest_next_point()
            
            # Evaluate
            value = objective_func(params)
            
            # Update
            self.tell(params, value)
            
            if (i + 1) % 10 == 0:
                print(f"Iteration {i+1}/{n_iterations}: Best = {self.best_y:.4f}")
        
        return {
            'best_params': self.best_x,
            'best_value': self.best_y,
            'all_params': [{name: x[i] for i, name in enumerate(self.param_names)} for x in self.X],
            'all_values': self.y
        }


class BayesianModelAveraging:
    """
    Bayesian Model Averaging for combining multiple trading models.
    """
    
    def __init__(self, models: List[str]):
        """
        Initialize BMA.
        
        Args:
            models: List of model names
        """
        self.models = models
        self.n_models = len(models)
        
        # Initialize with uniform prior
        self.model_probs = np.ones(self.n_models) / self.n_models
        self.model_evidence = np.zeros(self.n_models)
    
    def update_model_probabilities(
        self,
        predictions: Dict[str, float],
        actual: float,
        model_uncertainties: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        Update model probabilities based on prediction accuracy.
        
        Args:
            predictions: Dictionary of model predictions
            actual: Actual outcome
            model_uncertainties: Dictionary of model prediction uncertainties
            
        Returns:
            Updated model probabilities
        """
        # Calculate likelihood for each model
        likelihoods = np.zeros(self.n_models)
        
        for i, model in enumerate(self.models):
            pred = predictions.get(model, 0)
            
            if model_uncertainties and model in model_uncertainties:
                sigma = model_uncertainties[model]
            else:
                sigma = 0.1  # Default uncertainty
            
            # Gaussian likelihood
            likelihoods[i] = stats.norm.pdf(actual, pred, sigma)
        
        # Update posterior probabilities
        unnormalized = self.model_probs * likelihoods
        self.model_probs = unnormalized / np.sum(unnormalized)
        
        return {
            'model_probabilities': dict(zip(self.models, self.model_probs)),
            'best_model': self.models[np.argmax(self.model_probs)],
            'model_entropy': -np.sum(self.model_probs * np.log(self.model_probs + 1e-10))
        }
    
    def weighted_prediction(self, predictions: Dict[str, float]) -> float:
        """
        Get weighted average prediction.
        
        Args:
            predictions: Dictionary of model predictions
            
        Returns:
            Weighted prediction
        """
        weighted_sum = 0
        for i, model in enumerate(self.models):
            weighted_sum += self.model_probs[i] * predictions.get(model, 0)
        return weighted_sum


# PyMC3/Pyro integration patterns
class ProbabilisticModel:
    """
    Wrapper for probabilistic programming models (PyMC3/Pyro).
    """
    
    def __init__(self, backend: str = 'numpy'):
        """
        Initialize probabilistic model.
        
        Args:
            backend: Backend to use ('numpy', 'pymc', 'pyro')
        """
        self.backend = backend
        
        if backend == 'pymc' and not PYMC_AVAILABLE:
            warnings.warn("PyMC not available, falling back to numpy")
            self.backend = 'numpy'
        
        if backend == 'pyro' and not PYRO_AVAILABLE:
            warnings.warn("Pyro not available, falling back to numpy")
            self.backend = 'numpy'
    
    def estimate_returns_distribution(
        self,
        returns: np.ndarray,
        n_samples: int = 5000
    ) -> Dict:
        """
        Estimate posterior distribution of returns.
        
        Args:
            returns: Historical returns
            n_samples: Number of posterior samples
            
        Returns:
            Dictionary with posterior samples and statistics
        """
        returns = np.array(returns)
        
        if self.backend == 'pymc':
            return self._estimate_pymc(returns, n_samples)
        elif self.backend == 'pyro':
            return self._estimate_pyro(returns, n_samples)
        else:
            return self._estimate_numpy(returns, n_samples)
    
    def _estimate_numpy(self, returns: np.ndarray, n_samples: int) -> Dict:
        """Estimate using conjugate priors (numpy)."""
        n = len(returns)
        sample_mean = np.mean(returns)
        sample_var = np.var(returns, ddof=1)
        
        # Normal-inverse-gamma posterior
        # Prior parameters
        mu0 = 0
        kappa0 = 1
        alpha0 = 1
        beta0 = 1
        
        # Posterior parameters
        kappa_n = kappa0 + n
        mu_n = (kappa0 * mu0 + n * sample_mean) / kappa_n
        alpha_n = alpha0 + n / 2
        beta_n = beta0 + 0.5 * n * sample_var + 0.5 * (kappa0 * n * (sample_mean - mu0)**2) / kappa_n
        
        # Sample from posterior
        sigma2_samples = 1 / np.random.gamma(alpha_n, 1/beta_n, n_samples)
        mu_samples = np.random.normal(mu_n, np.sqrt(sigma2_samples / kappa_n))
        
        return {
            'mu_samples': mu_samples,
            'sigma_samples': np.sqrt(sigma2_samples),
            'mu_mean': np.mean(mu_samples),
            'mu_std': np.std(mu_samples),
            'sigma_mean': np.mean(np.sqrt(sigma2_samples)),
            'mu_ci': (np.percentile(mu_samples, 2.5), np.percentile(mu_samples, 97.5)),
            'sigma_ci': (np.percentile(np.sqrt(sigma2_samples), 2.5), 
                        np.percentile(np.sqrt(sigma2_samples), 97.5))
        }
    
    def _estimate_pymc(self, returns: np.ndarray, n_samples: int) -> Dict:
        """Estimate using PyMC."""
        with pm.Model() as model:
            # Priors
            mu = pm.Normal('mu', mu=0, sigma=0.1)
            sigma = pm.HalfNormal('sigma', sigma=0.1)
            
            # Likelihood
            obs = pm.Normal('obs', mu=mu, sigma=sigma, observed=returns)
            
            # Sample
            trace = pm.sample(n_samples, tune=1000, chains=4, progressbar=False)
        
        return {
            'mu_samples': trace.posterior['mu'].values.flatten(),
            'sigma_samples': trace.posterior['sigma'].values.flatten(),
            'mu_mean': np.mean(trace.posterior['mu']),
            'mu_std': np.std(trace.posterior['mu']),
            'sigma_mean': np.mean(trace.posterior['sigma']),
            'trace': trace
        }
    
    def _estimate_pyro(self, returns: np.ndarray, n_samples: int) -> Dict:
        """Estimate using Pyro."""
        if not PYRO_AVAILABLE:
            return self._estimate_numpy(returns, n_samples)
        
        returns_tensor = torch.tensor(returns, dtype=torch.float32)
        
        def model(data):
            mu = pyro.sample('mu', dist.Normal(0, 0.1))
            sigma = pyro.sample('sigma', dist.HalfNormal(0.1))
            with pyro.plate('data', len(data)):
                pyro.sample('obs', dist.Normal(mu, sigma), obs=data)
        
        def guide(data):
            mu_loc = pyro.param('mu_loc', torch.tensor(0.0))
            mu_scale = pyro.param('mu_scale', torch.tensor(0.1), constraint=dist.constraints.positive)
            sigma_loc = pyro.param('sigma_loc', torch.tensor(0.1), constraint=dist.constraints.positive)
            
            pyro.sample('mu', dist.Normal(mu_loc, mu_scale))
            pyro.sample('sigma', dist.HalfNormal(sigma_loc))
        
        # Run SVI
        optimizer = Adam({"lr": 0.01})
        svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
        
        for _ in range(1000):
            svi.step(returns_tensor)
        
        # Sample from posterior
        mu_loc = pyro.param('mu_loc').item()
        mu_scale = pyro.param('mu_scale').item()
        sigma_loc = pyro.param('sigma_loc').item()
        
        mu_samples = np.random.normal(mu_loc, mu_scale, n_samples)
        sigma_samples = np.random.exponential(sigma_loc, n_samples)
        
        return {
            'mu_samples': mu_samples,
            'sigma_samples': sigma_samples,
            'mu_mean': mu_loc,
            'mu_std': mu_scale,
            'sigma_mean': sigma_loc
        }


# Convenience functions
def quick_bayesian_update(
    prior_mean: float,
    prior_std: float,
    data: np.ndarray
) -> Dict:
    """
    Quick Bayesian update for normal-normal model.
    
    Args:
        prior_mean: Prior mean
        prior_std: Prior standard deviation
        data: Observed data
        
    Returns:
        Posterior statistics
    """
    updater = BayesianProbabilityUpdater()
    return updater.update_normal_normal(
        prior_mean, prior_std**2, data
    )


def estimate_signal_confidence(
    signal_history: List[Tuple[float, float]],
    current_signal: float
) -> Dict:
    """
    Quick signal confidence estimation.
    
    Args:
        signal_history: List of (signal, outcome) pairs
        current_signal: Current signal value
        
    Returns:
        Confidence metrics
    """
    estimator = SignalConfidenceEstimator()
    return estimator.estimate_signal_confidence(signal_history, current_signal)


if __name__ == "__main__":
    # Example usage
    print("Bayesian Inference Engine Demo")
    print("=" * 50)
    
    # Beta-Binomial example (win rate)
    updater = BayesianProbabilityUpdater()
    result = updater.update_beta_binomial(1, 1, 35, 50)  # 35 wins out of 50
    print(f"\nWin Rate Estimate:")
    print(f"  Mean: {result['mean']:.2%}")
    print(f"  95% CI: [{result['credibility_interval'][0]:.2%}, {result['credibility_interval'][1]:.2%}]")
    
    # Normal-Normal example (returns)
    returns = np.random.normal(0.001, 0.02, 100)
    result = updater.update_normal_normal(0, 0.01, returns)
    print(f"\nReturn Estimate:")
    print(f"  Mean: {result['posterior_mean']:.4f}")
    print(f"  95% CI: [{result['credibility_interval'][0]:.4f}, {result['credibility_interval'][1]:.4f}]")
    
    # Signal confidence
    signal_history = [(0.5, 0.02), (0.3, 0.01), (-0.2, -0.01), (0.6, 0.03)] * 20
    estimator = SignalConfidenceEstimator()
    conf = estimator.estimate_signal_confidence(signal_history, 0.4)
    print(f"\nSignal Confidence:")
    print(f"  Confidence: {conf['confidence']:.2%}")
    print(f"  Reliability: {conf['reliability']}")
