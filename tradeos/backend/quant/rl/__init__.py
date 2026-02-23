"""
Reinforcement Learning Module
=============================

Provides RL agents for trading strategy adaptation:
- DQN (Deep Q-Network)
- PPO (Proximal Policy Optimization)
- Market environment wrapper
- Strategy parameter adaptation
"""

from .agent import (
    DQNAgent,
    PPOAgent,
    TradingEnvironment,
    RLConfig,
    ReplayBuffer,
    StrategyParameterAdapter,
    RegimeBasedAgent,
    train_rl_agent,
    evaluate_agent
)

__all__ = [
    'DQNAgent',
    'PPOAgent',
    'TradingEnvironment',
    'RLConfig',
    'ReplayBuffer',
    'StrategyParameterAdapter',
    'RegimeBasedAgent',
    'train_rl_agent',
    'evaluate_agent'
]
