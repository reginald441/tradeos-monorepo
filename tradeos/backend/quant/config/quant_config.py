"""
Quant Configuration for TradeOS
===============================

Centralized configuration for all quantitative models.

Author: TradeOS Quant Team
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import json
import os


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulations."""
    n_simulations: int = 10000
    confidence_level: float = 0.95
    random_seed: Optional[int] = None
    parallel: bool = True
    n_workers: int = -1
    block_bootstrap: bool = False
    block_size: int = 10
    use_antithetic: bool = False
    
    def to_dict(self) -> Dict:
        return {
            'n_simulations': self.n_simulations,
            'confidence_level': self.confidence_level,
            'random_seed': self.random_seed,
            'parallel': self.parallel,
            'n_workers': self.n_workers,
            'block_bootstrap': self.block_bootstrap,
            'block_size': self.block_size,
            'use_antithetic': self.use_antithetic
        }


@dataclass
class PortfolioConfig:
    """Configuration for portfolio optimization."""
    objective: str = "max_sharpe"  # 'max_sharpe', 'min_variance', 'risk_parity'
    risk_free_rate: float = 0.02
    frequency: int = 252
    min_weight: float = 0.0
    max_weight: float = 1.0
    allow_short: bool = False
    target_return: Optional[float] = None
    target_risk: Optional[float] = None
    transaction_cost: float = 0.001
    
    def to_dict(self) -> Dict:
        return {
            'objective': self.objective,
            'risk_free_rate': self.risk_free_rate,
            'frequency': self.frequency,
            'min_weight': self.min_weight,
            'max_weight': self.max_weight,
            'allow_short': self.allow_short,
            'target_return': self.target_return,
            'target_risk': self.target_risk,
            'transaction_cost': self.transaction_cost
        }


@dataclass
class RLConfig:
    """Configuration for Reinforcement Learning agents."""
    # Network architecture
    state_dim: int = 64
    action_dim: int = 3
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    
    # Training parameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    
    # PPO specific
    ppo_clip: float = 0.2
    ppo_epochs: int = 10
    batch_size: int = 64
    
    # DQN specific
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    target_update_freq: int = 100
    replay_buffer_size: int = 100000
    
    # General
    device: str = "auto"
    seed: Optional[int] = None
    
    def to_dict(self) -> Dict:
        return {
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'hidden_dims': self.hidden_dims,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda,
            'ppo_clip': self.ppo_clip,
            'ppo_epochs': self.ppo_epochs,
            'batch_size': self.batch_size,
            'epsilon_start': self.epsilon_start,
            'epsilon_end': self.epsilon_end,
            'epsilon_decay': self.epsilon_decay,
            'target_update_freq': self.target_update_freq,
            'replay_buffer_size': self.replay_buffer_size,
            'device': self.device,
            'seed': self.seed
        }


@dataclass
class BayesianConfig:
    """Configuration for Bayesian inference."""
    prior_strength: float = 1.0
    credibility_interval: float = 0.95
    n_samples: int = 10000
    n_chains: int = 4
    random_seed: Optional[int] = None
    use_mcmc: bool = True
    use_vi: bool = False
    backend: str = "numpy"  # 'numpy', 'pymc', 'pyro'
    
    def to_dict(self) -> Dict:
        return {
            'prior_strength': self.prior_strength,
            'credibility_interval': self.credibility_interval,
            'n_samples': self.n_samples,
            'n_chains': self.n_chains,
            'random_seed': self.random_seed,
            'use_mcmc': self.use_mcmc,
            'use_vi': self.use_vi,
            'backend': self.backend
        }


@dataclass
class CovarianceConfig:
    """Configuration for covariance modeling."""
    method: str = "ewm"  # 'ewm', 'garch', 'dcc', 'realized'
    decay_factor: float = 0.94
    window_size: int = 252
    min_periods: int = 30
    annualize: bool = True
    frequency: int = 252
    garch_p: int = 1
    garch_q: int = 1
    
    def to_dict(self) -> Dict:
        return {
            'method': self.method,
            'decay_factor': self.decay_factor,
            'window_size': self.window_size,
            'min_periods': self.min_periods,
            'annualize': self.annualize,
            'frequency': self.frequency,
            'garch_p': self.garch_p,
            'garch_q': self.garch_q
        }


@dataclass
class HMMConfig:
    """Configuration for Hidden Markov Models."""
    n_components: int = 3
    covariance_type: str = "full"
    n_iter: int = 100
    tol: float = 1e-6
    random_state: Optional[int] = None
    init_params: str = "stmc"
    params: str = "stmc"
    
    def to_dict(self) -> Dict:
        return {
            'n_components': self.n_components,
            'covariance_type': self.covariance_type,
            'n_iter': self.n_iter,
            'tol': self.tol,
            'random_state': self.random_state,
            'init_params': self.init_params,
            'params': self.params
        }


@dataclass
class MetricsConfig:
    """Configuration for performance metrics."""
    risk_free_rate: float = 0.0
    frequency: str = 'daily'
    annualize: bool = True
    var_confidence: float = 0.95
    benchmark: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'risk_free_rate': self.risk_free_rate,
            'frequency': self.frequency,
            'annualize': self.annualize,
            'var_confidence': self.var_confidence,
            'benchmark': self.benchmark
        }


@dataclass
class BacktestConfig:
    """Configuration for backtest reporting."""
    initial_capital: float = 100000.0
    commission: float = 0.001
    slippage: float = 0.0005
    risk_free_rate: float = 0.02
    frequency: str = 'daily'
    currency: str = 'USD'
    
    def to_dict(self) -> Dict:
        return {
            'initial_capital': self.initial_capital,
            'commission': self.commission,
            'slippage': self.slippage,
            'risk_free_rate': self.risk_free_rate,
            'frequency': self.frequency,
            'currency': self.currency
        }


@dataclass
class QuantEngineConfig:
    """Main configuration for the Quant Engine."""
    # Sub-configurations
    monte_carlo: MonteCarloConfig = field(default_factory=MonteCarloConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    bayesian: BayesianConfig = field(default_factory=BayesianConfig)
    covariance: CovarianceConfig = field(default_factory=CovarianceConfig)
    hmm: HMMConfig = field(default_factory=HMMConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    
    # Global settings
    verbose: bool = True
    cache_results: bool = True
    parallel_processing: bool = True
    n_workers: int = -1
    random_seed: Optional[int] = None
    
    def to_dict(self) -> Dict:
        return {
            'monte_carlo': self.monte_carlo.to_dict(),
            'portfolio': self.portfolio.to_dict(),
            'rl': self.rl.to_dict(),
            'bayesian': self.bayesian.to_dict(),
            'covariance': self.covariance.to_dict(),
            'hmm': self.hmm.to_dict(),
            'metrics': self.metrics.to_dict(),
            'backtest': self.backtest.to_dict(),
            'verbose': self.verbose,
            'cache_results': self.cache_results,
            'parallel_processing': self.parallel_processing,
            'n_workers': self.n_workers,
            'random_seed': self.random_seed
        }
    
    def to_json(self, filepath: Optional[str] = None) -> str:
        """Export configuration to JSON."""
        json_str = json.dumps(self.to_dict(), indent=2)
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)
        
        return json_str
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'QuantEngineConfig':
        """Create configuration from dictionary."""
        return cls(
            monte_carlo=MonteCarloConfig(**config_dict.get('monte_carlo', {})),
            portfolio=PortfolioConfig(**config_dict.get('portfolio', {})),
            rl=RLConfig(**config_dict.get('rl', {})),
            bayesian=BayesianConfig(**config_dict.get('bayesian', {})),
            covariance=CovarianceConfig(**config_dict.get('covariance', {})),
            hmm=HMMConfig(**config_dict.get('hmm', {})),
            metrics=MetricsConfig(**config_dict.get('metrics', {})),
            backtest=BacktestConfig(**config_dict.get('backtest', {})),
            verbose=config_dict.get('verbose', True),
            cache_results=config_dict.get('cache_results', True),
            parallel_processing=config_dict.get('parallel_processing', True),
            n_workers=config_dict.get('n_workers', -1),
            random_seed=config_dict.get('random_seed', None)
        )
    
    @classmethod
    def from_json(cls, filepath: str) -> 'QuantEngineConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


# Predefined configurations
class PresetConfigs:
    """Predefined configuration presets."""
    
    @staticmethod
    def conservative() -> QuantEngineConfig:
        """Conservative configuration for low-risk strategies."""
        config = QuantEngineConfig()
        
        # Portfolio optimization
        config.portfolio.objective = "min_variance"
        config.portfolio.min_weight = 0.0
        config.portfolio.max_weight = 0.2  # Max 20% per asset
        
        # Monte Carlo
        config.monte_carlo.n_simulations = 50000
        config.monte_carlo.confidence_level = 0.99
        
        # HMM
        config.hmm.n_components = 4  # More granular regimes
        
        return config
    
    @staticmethod
    def aggressive() -> QuantEngineConfig:
        """Aggressive configuration for high-risk strategies."""
        config = QuantEngineConfig()
        
        # Portfolio optimization
        config.portfolio.objective = "max_sharpe"
        config.portfolio.allow_short = True
        config.portfolio.max_weight = 2.0  # Allow leverage
        
        # Monte Carlo
        config.monte_carlo.n_simulations = 10000
        
        # RL
        config.rl.learning_rate = 1e-3
        
        return config
    
    @staticmethod
    def high_frequency() -> QuantEngineConfig:
        """Configuration for high-frequency trading."""
        config = QuantEngineConfig()
        
        # Covariance
        config.covariance.method = "realized"
        config.covariance.window_size = 50
        
        # Metrics
        config.metrics.frequency = 'hourly'
        
        # Backtest
        config.backtest.commission = 0.0005  # Lower commission for HFT
        
        return config
    
    @staticmethod
    def research() -> QuantEngineConfig:
        """Configuration for research and backtesting."""
        config = QuantEngineConfig()
        
        # Monte Carlo
        config.monte_carlo.n_simulations = 100000
        config.monte_carlo.parallel = True
        
        # Bayesian
        config.bayesian.n_samples = 50000
        config.bayesian.n_chains = 8
        
        # Verbose output
        config.verbose = True
        config.cache_results = True
        
        return config


# Environment-based configuration
def load_config_from_env() -> QuantEngineConfig:
    """Load configuration from environment variables."""
    config = QuantEngineConfig()
    
    # Monte Carlo
    if os.getenv('MC_SIMULATIONS'):
        config.monte_carlo.n_simulations = int(os.getenv('MC_SIMULATIONS'))
    
    if os.getenv('MC_CONFIDENCE'):
        config.monte_carlo.confidence_level = float(os.getenv('MC_CONFIDENCE'))
    
    # Portfolio
    if os.getenv('PORTFOLIO_OBJECTIVE'):
        config.portfolio.objective = os.getenv('PORTFOLIO_OBJECTIVE')
    
    if os.getenv('RISK_FREE_RATE'):
        config.portfolio.risk_free_rate = float(os.getenv('RISK_FREE_RATE'))
    
    # RL
    if os.getenv('RL_LEARNING_RATE'):
        config.rl.learning_rate = float(os.getenv('RL_LEARNING_RATE'))
    
    if os.getenv('RL_DEVICE'):
        config.rl.device = os.getenv('RL_DEVICE')
    
    # Global
    if os.getenv('QUANT_VERBOSE'):
        config.verbose = os.getenv('QUANT_VERBOSE').lower() == 'true'
    
    if os.getenv('QUANT_PARALLEL'):
        config.parallel_processing = os.getenv('QUANT_PARALLEL').lower() == 'true'
    
    if os.getenv('QUANT_N_WORKERS'):
        config.n_workers = int(os.getenv('QUANT_N_WORKERS'))
    
    if os.getenv('QUANT_RANDOM_SEED'):
        config.random_seed = int(os.getenv('QUANT_RANDOM_SEED'))
    
    return config


# Default configuration instance
default_config = QuantEngineConfig()


if __name__ == "__main__":
    # Example usage
    print("Quant Configuration Demo")
    print("=" * 50)
    
    # Default configuration
    print("\n1. Default Configuration:")
    config = QuantEngineConfig()
    print(f"   Monte Carlo simulations: {config.monte_carlo.n_simulations}")
    print(f"   Portfolio objective: {config.portfolio.objective}")
    print(f"   RL learning rate: {config.rl.learning_rate}")
    
    # Preset configurations
    print("\n2. Conservative Configuration:")
    conservative = PresetConfigs.conservative()
    print(f"   Portfolio objective: {conservative.portfolio.objective}")
    print(f"   Max weight: {conservative.portfolio.max_weight}")
    
    print("\n3. Aggressive Configuration:")
    aggressive = PresetConfigs.aggressive()
    print(f"   Portfolio objective: {aggressive.portfolio.objective}")
    print(f"   Allow short: {aggressive.portfolio.allow_short}")
    
    # Export to JSON
    print("\n4. Export to JSON:")
    json_str = config.to_json()
    print(json_str[:200] + "...")
