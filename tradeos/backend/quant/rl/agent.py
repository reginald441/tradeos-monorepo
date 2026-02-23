"""
Reinforcement Learning Agents for TradeOS
=========================================

Implements RL agents for trading strategy adaptation:
- PPO (Proximal Policy Optimization) agent
- DQN (Deep Q-Network) agent
- Market environment wrapper
- Strategy parameter adaptation
- Regime-based switching

Author: TradeOS Quant Team
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Callable, Union, Any
from dataclasses import dataclass
from collections import deque
import random
import warnings
from abc import ABC, abstractmethod

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.distributions import Categorical, Normal
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. RL agents will not function.")


@dataclass
class RLConfig:
    """Configuration for RL agents."""
    # Network architecture
    state_dim: int = 64
    action_dim: int = 3
    hidden_dims: List[int] = None
    
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
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 256]
        
        if self.device == "auto":
            if TORCH_AVAILABLE and torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"


class TradingEnvironment:
    """
    Trading environment wrapper for reinforcement learning.
    
    Implements OpenAI Gym-style interface for trading:
    - State: Market features, position info, account status
    - Actions: Buy, Sell, Hold, position sizing
    - Rewards: PnL-based, risk-adjusted, or custom
    """
    
    def __init__(
        self,
        price_data: pd.DataFrame,
        features: Optional[np.ndarray] = None,
        initial_capital: float = 100000.0,
        commission: float = 0.001,
        reward_type: str = "returns",
        window_size: int = 20,
        max_position: float = 1.0
    ):
        """
        Initialize trading environment.
        
        Args:
            price_data: DataFrame with OHLCV data
            features: Pre-computed features (optional)
            initial_capital: Starting capital
            commission: Commission per trade (as fraction)
            reward_type: Type of reward function ('returns', 'sharpe', 'sortino')
            window_size: Observation window size
            max_position: Maximum position size (as fraction of capital)
        """
        self.price_data = price_data.reset_index(drop=True)
        self.n_steps = len(price_data)
        
        if features is not None:
            self.features = features
        else:
            self.features = self._compute_features()
        
        self.initial_capital = initial_capital
        self.commission = commission
        self.reward_type = reward_type
        self.window_size = window_size
        self.max_position = max_position
        
        # State dimensions
        self.feature_dim = self.features.shape[1]
        self.state_dim = self.feature_dim + 3  # features + position + cash + equity
        
        # Action space: 0=Hold, 1=Buy, 2=Sell, 3=Increase, 4=Decrease
        self.action_dim = 5
        
        self.reset()
    
    def _compute_features(self) -> np.ndarray:
        """Compute technical features from price data."""
        features = []
        
        # Returns
        returns = self.price_data['close'].pct_change().fillna(0)
        features.append(returns)
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            ma = self.price_data['close'].rolling(window).mean()
            features.append((self.price_data['close'] - ma) / ma)
        
        # Volatility
        for window in [10, 20]:
            vol = returns.rolling(window).std()
            features.append(vol)
        
        # RSI
        delta = self.price_data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        features.append(rsi / 100)
        
        # MACD
        ema12 = self.price_data['close'].ewm(span=12).mean()
        ema26 = self.price_data['close'].ewm(span=26).mean()
        macd = ema12 - ema26
        features.append(macd / self.price_data['close'])
        
        # Bollinger Bands
        ma20 = self.price_data['close'].rolling(20).mean()
        std20 = self.price_data['close'].rolling(20).std()
        bb_upper = ma20 + 2 * std20
        bb_lower = ma20 - 2 * std20
        features.append((self.price_data['close'] - bb_lower) / (bb_upper - bb_lower))
        
        # Volume features
        if 'volume' in self.price_data.columns:
            vol_ma = self.price_data['volume'].rolling(20).mean()
            features.append(self.price_data['volume'] / vol_ma)
        
        # Combine features
        feature_df = pd.concat(features, axis=1)
        feature_df = feature_df.fillna(0).replace([np.inf, -np.inf], 0)
        
        return feature_df.values
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.current_step = self.window_size
        self.position = 0.0  # Current position (0 to max_position)
        self.cash = self.initial_capital
        self.equity = self.initial_capital
        self.trades = []
        self.equity_curve = [self.initial_capital]
        self.returns_history = deque(maxlen=50)
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current state observation."""
        # Feature window
        feature_window = self.features[
            max(0, self.current_step - self.window_size):self.current_step
        ]
        
        # Pad if necessary
        if len(feature_window) < self.window_size:
            padding = np.zeros((self.window_size - len(feature_window), self.feature_dim))
            feature_window = np.vstack([padding, feature_window])
        
        # Flatten features
        feature_flat = feature_window.flatten()
        
        # Add account info
        account_info = np.array([
            self.position / self.max_position,  # Normalized position
            self.cash / self.initial_capital,   # Normalized cash
            self.equity / self.initial_capital  # Normalized equity
        ])
        
        return np.concatenate([feature_flat, account_info])
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one trading step.
        
        Args:
            action: Action to take (0=Hold, 1=Buy, 2=Sell, 3=Increase, 4=Decrease)
            
        Returns:
            (next_state, reward, done, info)
        """
        current_price = self.price_data['close'].iloc[self.current_step]
        prev_equity = self.equity
        
        # Execute action
        if action == 1:  # Buy
            target_position = self.max_position
        elif action == 2:  # Sell
            target_position = -self.max_position
        elif action == 3:  # Increase position
            target_position = min(self.position + 0.25, self.max_position)
        elif action == 4:  # Decrease position
            target_position = max(self.position - 0.25, -self.max_position)
        else:  # Hold
            target_position = self.position
        
        # Calculate trade
        position_change = target_position - self.position
        shares_to_trade = position_change * self.equity / current_price
        
        # Apply commission
        trade_value = abs(shares_to_trade) * current_price
        commission_cost = trade_value * self.commission
        
        # Update position and cash
        self.position = target_position
        self.cash -= shares_to_trade * current_price + commission_cost
        
        # Update equity
        position_value = self.position * self.equity  # Simplified
        self.equity = self.cash + position_value
        
        # Record trade
        if abs(position_change) > 0.001:
            self.trades.append({
                'step': self.current_step,
                'action': action,
                'price': current_price,
                'position': self.position,
                'commission': commission_cost
            })
        
        # Calculate reward
        reward = self._calculate_reward(prev_equity)
        
        # Update history
        self.equity_curve.append(self.equity)
        if len(self.equity_curve) > 1:
            ret = (self.equity_curve[-1] - self.equity_curve[-2]) / self.equity_curve[-2]
            self.returns_history.append(ret)
        
        # Advance
        self.current_step += 1
        done = self.current_step >= self.n_steps - 1
        
        # Get next state
        next_state = self._get_state() if not done else np.zeros(self.state_dim)
        
        info = {
            'equity': self.equity,
            'position': self.position,
            'step': self.current_step
        }
        
        return next_state, reward, done, info
    
    def _calculate_reward(self, prev_equity: float) -> float:
        """Calculate reward based on reward type."""
        if self.reward_type == "returns":
            return (self.equity - prev_equity) / prev_equity
        
        elif self.reward_type == "log_returns":
            return np.log(self.equity / prev_equity)
        
        elif self.reward_type == "sharpe":
            if len(self.returns_history) < 10:
                return (self.equity - prev_equity) / prev_equity
            returns = np.array(self.returns_history)
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8)
            return sharpe
        
        elif self.reward_type == "sortino":
            if len(self.returns_history) < 10:
                return (self.equity - prev_equity) / prev_equity
            returns = np.array(self.returns_history)
            downside = returns[returns < 0]
            downside_std = np.std(downside) if len(downside) > 0 else 1e-8
            sortino = np.mean(returns) / downside_std
            return sortino
        
        elif self.reward_type == "calmar":
            if len(self.equity_curve) < 10:
                return (self.equity - prev_equity) / prev_equity
            returns = np.array(self.returns_history)
            equity_array = np.array(self.equity_curve)
            running_max = np.maximum.accumulate(equity_array)
            drawdown = (equity_array - running_max) / running_max
            max_dd = abs(np.min(drawdown))
            calmar = np.mean(returns) * 252 / (max_dd + 1e-8)
            return calmar
        
        else:
            return (self.equity - prev_equity) / prev_equity


class ReplayBuffer:
    """Experience replay buffer for DQN."""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """Sample batch of experiences."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.buffer)


if TORCH_AVAILABLE:
    class QNetwork(nn.Module):
        """Q-Network for DQN."""
        
        def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int]):
            super().__init__()
            
            layers = []
            prev_dim = state_dim
            
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                prev_dim = hidden_dim
            
            layers.append(nn.Linear(prev_dim, action_dim))
            
            self.network = nn.Sequential(*layers)
        
        def forward(self, x):
            return self.network(x)
    
    
    class PolicyNetwork(nn.Module):
        """Policy network for PPO."""
        
        def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_dims: List[int],
            continuous: bool = False
        ):
            super().__init__()
            
            self.continuous = continuous
            
            # Shared layers
            layers = []
            prev_dim = state_dim
            
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                prev_dim = hidden_dim
            
            self.shared = nn.Sequential(*layers)
            
            # Policy head
            if continuous:
                self.mean = nn.Linear(prev_dim, action_dim)
                self.log_std = nn.Parameter(torch.zeros(action_dim))
            else:
                self.action_logits = nn.Linear(prev_dim, action_dim)
            
            # Value head
            self.value = nn.Linear(prev_dim, 1)
        
        def forward(self, x):
            features = self.shared(x)
            value = self.value(features)
            
            if self.continuous:
                mean = self.mean(features)
                std = torch.exp(self.log_std)
                return mean, std, value
            else:
                action_probs = F.softmax(self.action_logits(features), dim=-1)
                return action_probs, value
        
        def get_action(self, state):
            """Sample action from policy."""
            with torch.no_grad():
                if self.continuous:
                    mean, std, _ = self.forward(state)
                    dist = Normal(mean, std)
                    action = dist.sample()
                    log_prob = dist.log_prob(action).sum(-1)
                    return action, log_prob
                else:
                    action_probs, _ = self.forward(state)
                    dist = Categorical(action_probs)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                    return action, log_prob
        
        def evaluate(self, state, action):
            """Evaluate action for PPO update."""
            if self.continuous:
                mean, std, value = self.forward(state)
                dist = Normal(mean, std)
                log_prob = dist.log_prob(action).sum(-1)
                entropy = dist.entropy().sum(-1)
            else:
                action_probs, value = self.forward(state)
                dist = Categorical(action_probs)
                log_prob = dist.log_prob(action)
                entropy = dist.entropy()
            
            return log_prob, value.squeeze(-1), entropy


class DQNAgent:
    """
    Deep Q-Network agent for trading.
    
    Implements DQN with:
    - Experience replay
    - Target network
    - Epsilon-greedy exploration
    """
    
    def __init__(self, config: RLConfig):
        """Initialize DQN agent."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for DQN agent")
        
        self.config = config
        
        if config.seed is not None:
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)
            random.seed(config.seed)
        
        self.device = torch.device(config.device)
        
        # Networks
        self.q_network = QNetwork(
            config.state_dim,
            config.action_dim,
            config.hidden_dims
        ).to(self.device)
        
        self.target_network = QNetwork(
            config.state_dim,
            config.action_dim,
            config.hidden_dims
        ).to(self.device)
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=config.learning_rate
        )
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(config.replay_buffer_size)
        
        # Exploration
        self.epsilon = config.epsilon_start
        
        # Training step counter
        self.train_step = 0
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randrange(self.config.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train(self) -> Optional[float]:
        """Train the agent."""
        if len(self.replay_buffer) < self.config.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.config.batch_size
        )
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q values
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + self.config.gamma * next_q * (1 - dones)
        
        # Loss
        loss = F.mse_loss(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.train_step += 1
        if self.train_step % self.config.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(
            self.config.epsilon_end,
            self.epsilon * self.config.epsilon_decay
        )
        
        return loss.item()
    
    def save(self, path: str):
        """Save model."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_step': self.train_step
        }, path)
    
    def load(self, path: str):
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.train_step = checkpoint['train_step']


class PPOAgent:
    """
    Proximal Policy Optimization agent for trading.
    
    Implements PPO with:
    - Clipped surrogate objective
    - Generalized Advantage Estimation (GAE)
    - Mini-batch updates
    """
    
    def __init__(self, config: RLConfig, continuous: bool = False):
        """Initialize PPO agent."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for PPO agent")
        
        self.config = config
        self.continuous = continuous
        
        if config.seed is not None:
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)
            random.seed(config.seed)
        
        self.device = torch.device(config.device)
        
        # Policy network
        self.policy = PolicyNetwork(
            config.state_dim,
            config.action_dim,
            config.hidden_dims,
            continuous
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=config.learning_rate
        )
        
        # Memory for rollout
        self.reset_memory()
    
    def reset_memory(self):
        """Reset rollout memory."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def select_action(self, state: np.ndarray) -> Tuple[Union[int, np.ndarray], Dict]:
        """Select action from policy."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            if self.continuous:
                action, log_prob = self.policy.get_action(state_tensor)
                action = action.cpu().numpy()[0]
                log_prob = log_prob.item()
                _, _, value = self.policy(state_tensor)
            else:
                action, log_prob = self.policy.get_action(state_tensor)
                action = action.item()
                log_prob = log_prob.item()
                _, value = self.policy(state_tensor)
            
            value = value.item()
        
        return action, {
            'log_prob': log_prob,
            'value': value
        }
    
    def store_transition(self, state, action, reward, done, info):
        """Store transition."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(info['value'])
        self.log_probs.append(info['log_prob'])
        self.dones.append(done)
    
    def compute_gae(self, next_value: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation."""
        rewards = np.array(self.rewards)
        values = np.array(self.values + [next_value])
        dones = np.array(self.dones)
        
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.config.gamma * next_value - values[t]
            advantages[t] = last_gae = delta + self.config.gamma * self.config.gae_lambda * last_gae
        
        returns = advantages + values[:-1]
        
        return advantages, returns
    
    def train(self, next_state: np.ndarray) -> Dict:
        """Train policy using collected rollout."""
        # Get next value for GAE
        with torch.no_grad():
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            if self.continuous:
                _, _, next_value = self.policy(next_state_tensor)
            else:
                _, next_value = self.policy(next_state_tensor)
            next_value = next_value.item()
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(next_value)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device) if self.continuous else torch.LongTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # PPO update
        total_policy_loss = 0
        total_value_loss = 0
        
        for _ in range(self.config.ppo_epochs):
            # Evaluate actions
            log_probs, values, entropy = self.policy.evaluate(states, actions)
            
            # Policy loss (clipped)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config.ppo_clip, 1 + self.config.ppo_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values, returns)
            
            # Total loss
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
        
        # Reset memory
        self.reset_memory()
        
        return {
            'policy_loss': total_policy_loss / self.config.ppo_epochs,
            'value_loss': total_value_loss / self.config.ppo_epochs
        }
    
    def save(self, path: str):
        """Save model."""
        torch.save({
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)
    
    def load(self, path: str):
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])


class StrategyParameterAdapter:
    """
    RL-based strategy parameter adaptation.
    
    Adapts trading strategy parameters based on market conditions.
    """
    
    def __init__(
        self,
        param_ranges: Dict[str, Tuple[float, float]],
        config: RLConfig
    ):
        """
        Initialize parameter adapter.
        
        Args:
            param_ranges: Dictionary of parameter names to (min, max) ranges
            config: RL configuration
        """
        self.param_ranges = param_ranges
        self.param_names = list(param_ranges.keys())
        self.n_params = len(param_names)
        
        # Create continuous action space for parameters
        config.action_dim = self.n_params
        self.agent = PPOAgent(config, continuous=True)
    
    def adapt_parameters(
        self,
        market_state: np.ndarray,
        current_params: Dict[str, float],
        performance: float
    ) -> Dict[str, float]:
        """
        Adapt strategy parameters based on market state.
        
        Args:
            market_state: Current market state features
            current_params: Current parameter values
            performance: Recent strategy performance
            
        Returns:
            Updated parameters
        """
        # Select action (parameter adjustments)
        action, info = self.agent.select_action(market_state)
        
        # Convert action to parameter adjustments (-1 to 1)
        adjustments = np.tanh(action)
        
        # Apply adjustments
        new_params = {}
        for i, param_name in enumerate(self.param_names):
            min_val, max_val = self.param_ranges[param_name]
            current_val = current_params.get(param_name, (min_val + max_val) / 2)
            
            # Adjust parameter
            range_size = max_val - min_val
            adjustment = adjustments[i] * range_size * 0.1  # 10% adjustment max
            new_val = np.clip(current_val + adjustment, min_val, max_val)
            new_params[param_name] = new_val
        
        return new_params
    
    def update(self, reward: float):
        """Update the adapter with performance reward."""
        # Store transition and train if needed
        pass  # Implementation depends on specific use case


class RegimeBasedAgent:
    """
    RL agent that switches strategies based on market regime.
    """
    
    def __init__(
        self,
        regime_detector: Any,
        agents: Dict[str, Any],
        config: RLConfig
    ):
        """
        Initialize regime-based agent.
        
        Args:
            regime_detector: Regime detection model
            agents: Dictionary mapping regime names to agents
            config: RL configuration
        """
        self.regime_detector = regime_detector
        self.agents = agents
        self.current_regime = None
        self.current_agent = None
    
    def detect_regime(self, market_data: np.ndarray) -> str:
        """Detect current market regime."""
        regime = self.regime_detector.predict(market_data)
        return regime
    
    def select_action(self, state: np.ndarray, market_data: np.ndarray) -> int:
        """Select action based on current regime."""
        # Detect regime
        regime = self.detect_regime(market_data)
        
        # Switch agent if regime changed
        if regime != self.current_regime:
            self.current_regime = regime
            self.current_agent = self.agents.get(regime, self.agents.get('default'))
        
        # Select action from current agent
        if hasattr(self.current_agent, 'select_action'):
            return self.current_agent.select_action(state)
        else:
            return self.current_agent
    
    def train(self, *args, **kwargs):
        """Train all agents."""
        for regime, agent in self.agents.items():
            if hasattr(agent, 'train'):
                agent.train(*args, **kwargs)


# Training utilities
def train_rl_agent(
    agent: Union[DQNAgent, PPOAgent],
    env: TradingEnvironment,
    n_episodes: int = 1000,
    eval_freq: int = 100,
    save_path: Optional[str] = None
) -> Dict:
    """
    Train an RL agent.
    
    Args:
        agent: RL agent to train
        env: Trading environment
        n_episodes: Number of training episodes
        eval_freq: Evaluation frequency
        save_path: Path to save best model
        
    Returns:
        Training history
    """
    history = {
        'episode_returns': [],
        'episode_lengths': [],
        'eval_returns': [],
        'losses': []
    }
    
    best_eval_return = -np.inf
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_return = 0
        episode_length = 0
        
        done = False
        while not done:
            # Select action
            if isinstance(agent, DQNAgent):
                action = agent.select_action(state, training=True)
                next_state, reward, done, info = env.step(action)
                agent.store_transition(state, action, reward, next_state, done)
                
                # Train
                loss = agent.train()
                if loss is not None:
                    history['losses'].append(loss)
            
            elif isinstance(agent, PPOAgent):
                action, action_info = agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                agent.store_transition(state, action, reward, done, action_info)
                
                # Train at end of episode or when memory is full
                if done or len(agent.states) >= 2048:
                    train_info = agent.train(next_state)
            
            episode_return += reward
            episode_length += 1
            state = next_state
        
        history['episode_returns'].append(episode_return)
        history['episode_lengths'].append(episode_length)
        
        # Evaluation
        if (episode + 1) % eval_freq == 0:
            eval_return = evaluate_agent(agent, env)
            history['eval_returns'].append(eval_return)
            
            print(f"Episode {episode + 1}/{n_episodes} - "
                  f"Train Return: {episode_return:.4f}, "
                  f"Eval Return: {eval_return:.4f}")
            
            # Save best model
            if save_path and eval_return > best_eval_return:
                best_eval_return = eval_return
                agent.save(save_path)
    
    return history


def evaluate_agent(
    agent: Union[DQNAgent, PPOAgent],
    env: TradingEnvironment,
    n_episodes: int = 10
) -> float:
    """
    Evaluate agent performance.
    
    Args:
        agent: RL agent
        env: Trading environment
        n_episodes: Number of evaluation episodes
        
    Returns:
        Average return
    """
    returns = []
    
    for _ in range(n_episodes):
        state = env.reset()
        episode_return = 0
        done = False
        
        while not done:
            if isinstance(agent, DQNAgent):
                action = agent.select_action(state, training=False)
            else:
                action, _ = agent.select_action(state)
            
            state, reward, done, _ = env.step(action)
            episode_return += reward
        
        returns.append(episode_return)
    
    return np.mean(returns)


if __name__ == "__main__":
    # Example usage
    if TORCH_AVAILABLE:
        print("PyTorch available - RL agents ready")
        print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    else:
        print("PyTorch not available - RL agents disabled")
