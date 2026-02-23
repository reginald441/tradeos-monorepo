"""
TradeOS Strategy Configuration Module
=====================================
Configuration schemas and validation for trading strategies.

Author: TradeOS Team
Version: 1.0.0
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Union, Callable
from enum import Enum
import json
import yaml
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Strategy type classification."""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    STATISTICAL_ARBITRAGE = "statistical_arbitrage"
    LIQUIDITY = "liquidity"
    MULTI_FACTOR = "multi_factor"
    CUSTOM = "custom"


class PositionSizingMethod(Enum):
    """Position sizing methods."""
    FIXED = "fixed"
    PERCENT_OF_EQUITY = "percent_of_equity"
    KELLY = "kelly"
    ATR_BASED = "atr_based"
    RISK_PARITY = "risk_parity"
    VOLATILITY_TARGET = "volatility_target"


class RiskManagementMethod(Enum):
    """Risk management methods."""
    FIXED_STOP = "fixed_stop"
    ATR_STOP = "atr_stop"
    TRAILING_STOP = "trailing_stop"
    TIME_STOP = "time_stop"
    PORTFOLIO_STOP = "portfolio_stop"


@dataclass
class PositionSizingConfig:
    """Position sizing configuration."""
    method: PositionSizingMethod = PositionSizingMethod.PERCENT_OF_EQUITY
    fixed_amount: float = 10000.0
    percent_of_equity: float = 0.1
    risk_per_trade: float = 0.02
    atr_period: int = 14
    atr_multiple: float = 2.0
    kelly_win_rate: float = 0.5
    kelly_avg_win: float = 1.0
    kelly_avg_loss: float = 1.0
    max_position_pct: float = 0.25
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'method': self.method.value,
            'fixed_amount': self.fixed_amount,
            'percent_of_equity': self.percent_of_equity,
            'risk_per_trade': self.risk_per_trade,
            'atr_period': self.atr_period,
            'atr_multiple': self.atr_multiple,
            'kelly_win_rate': self.kelly_win_rate,
            'kelly_avg_win': self.kelly_avg_win,
            'kelly_avg_loss': self.kelly_avg_loss,
            'max_position_pct': self.max_position_pct
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PositionSizingConfig':
        if 'method' in data and isinstance(data['method'], str):
            data['method'] = PositionSizingMethod(data['method'])
        return cls(**data)


@dataclass
class RiskManagementConfig:
    """Risk management configuration."""
    method: RiskManagementMethod = RiskManagementMethod.ATR_STOP
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04
    trailing_stop_pct: float = 0.015
    atr_period: int = 14
    atr_stop_multiple: float = 2.0
    max_holding_periods: int = 20
    max_daily_loss_pct: float = 0.05
    max_position_drawdown_pct: float = 0.1
    use_position_sizing: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'method': self.method.value,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'trailing_stop_pct': self.trailing_stop_pct,
            'atr_period': self.atr_period,
            'atr_stop_multiple': self.atr_stop_multiple,
            'max_holding_periods': max_holding_periods,
            'max_daily_loss_pct': self.max_daily_loss_pct,
            'max_position_drawdown_pct': self.max_position_drawdown_pct,
            'use_position_sizing': self.use_position_sizing
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RiskManagementConfig':
        if 'method' in data and isinstance(data['method'], str):
            data['method'] = RiskManagementMethod(data['method'])
        return cls(**data)


@dataclass
class IndicatorConfig:
    """Technical indicator configuration."""
    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'params': self.params,
            'enabled': self.enabled
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IndicatorConfig':
        return cls(**data)


@dataclass
class FilterConfig:
    """Signal filter configuration."""
    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    required: bool = False  # If True, filter must pass for signal
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'params': self.params,
            'enabled': self.enabled,
            'required': self.required
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FilterConfig':
        return cls(**data)


@dataclass
class StrategyParameters:
    """Strategy-specific parameters."""
    params: Dict[str, Any] = field(default_factory=dict)
    
    def get(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        self.params[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        return self.params.copy()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyParameters':
        return cls(params=data)


@dataclass
class StrategyConfig:
    """Complete strategy configuration."""
    name: str
    strategy_type: StrategyType
    description: str = ""
    version: str = "1.0.0"
    
    # Trading parameters
    symbols: List[str] = field(default_factory=list)
    timeframes: List[str] = field(default_factory=lambda: ['1h'])
    min_bars: int = 50
    
    # Component configurations
    parameters: StrategyParameters = field(default_factory=StrategyParameters)
    position_sizing: PositionSizingConfig = field(default_factory=PositionSizingConfig)
    risk_management: RiskManagementConfig = field(default_factory=RiskManagementConfig)
    indicators: List[IndicatorConfig] = field(default_factory=list)
    filters: List[FilterConfig] = field(default_factory=list)
    
    # Strategy settings
    enabled: bool = True
    weight: float = 1.0
    priority: int = 0
    
    # Regime settings
    allowed_regimes: List[str] = field(default_factory=lambda: ['all'])
    disallowed_regimes: List[str] = field(default_factory=list)
    
    # Backtest settings
    backtest_start: Optional[str] = None
    backtest_end: Optional[str] = None
    initial_capital: float = 100000.0
    commission: float = 0.001
    slippage: float = 0.0
    
    # Metadata
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    author: str = ""
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if isinstance(self.strategy_type, str):
            self.strategy_type = StrategyType(self.strategy_type)
        if isinstance(self.parameters, dict):
            self.parameters = StrategyParameters.from_dict(self.parameters)
        if isinstance(self.position_sizing, dict):
            self.position_sizing = PositionSizingConfig.from_dict(self.position_sizing)
        if isinstance(self.risk_management, dict):
            self.risk_management = RiskManagementConfig.from_dict(self.risk_management)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'strategy_type': self.strategy_type.value,
            'description': self.description,
            'version': self.version,
            'symbols': self.symbols,
            'timeframes': self.timeframes,
            'min_bars': self.min_bars,
            'parameters': self.parameters.to_dict(),
            'position_sizing': self.position_sizing.to_dict(),
            'risk_management': self.risk_management.to_dict(),
            'indicators': [i.to_dict() for i in self.indicators],
            'filters': [f.to_dict() for f in self.filters],
            'enabled': self.enabled,
            'weight': self.weight,
            'priority': self.priority,
            'allowed_regimes': self.allowed_regimes,
            'disallowed_regimes': self.disallowed_regimes,
            'backtest_start': self.backtest_start,
            'backtest_end': self.backtest_end,
            'initial_capital': self.initial_capital,
            'commission': self.commission,
            'slippage': self.slippage,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'author': self.author,
            'tags': self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyConfig':
        # Convert nested objects
        if 'parameters' in data and isinstance(data['parameters'], dict):
            data['parameters'] = StrategyParameters.from_dict(data['parameters'])
        if 'position_sizing' in data and isinstance(data['position_sizing'], dict):
            data['position_sizing'] = PositionSizingConfig.from_dict(data['position_sizing'])
        if 'risk_management' in data and isinstance(data['risk_management'], dict):
            data['risk_management'] = RiskManagementConfig.from_dict(data['risk_management'])
        if 'indicators' in data and isinstance(data['indicators'], list):
            data['indicators'] = [IndicatorConfig.from_dict(i) for i in data['indicators']]
        if 'filters' in data and isinstance(data['filters'], list):
            data['filters'] = [FilterConfig.from_dict(f) for f in data['filters']]
        
        return cls(**data)
    
    def to_json(self, filepath: Optional[str] = None) -> str:
        """Export to JSON string or file."""
        json_str = json.dumps(self.to_dict(), indent=2)
        
        if filepath:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w') as f:
                f.write(json_str)
            logger.info(f"Strategy config saved to {filepath}")
        
        return json_str
    
    def to_yaml(self, filepath: Optional[str] = None) -> str:
        """Export to YAML string or file."""
        yaml_str = yaml.dump(self.to_dict(), default_flow_style=False)
        
        if filepath:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w') as f:
                f.write(yaml_str)
            logger.info(f"Strategy config saved to {filepath}")
        
        return yaml_str
    
    @classmethod
    def from_json(cls, filepath: str) -> 'StrategyConfig':
        """Load from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Strategy config loaded from {filepath}")
        return cls.from_dict(data)
    
    @classmethod
    def from_yaml(cls, filepath: str) -> 'StrategyConfig':
        """Load from YAML file."""
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        
        logger.info(f"Strategy config loaded from {filepath}")
        return cls.from_dict(data)


@dataclass
class EngineConfig:
    """Strategy engine configuration."""
    name: str = "TradeOS Engine"
    mode: str = "backtest"
    initial_capital: float = 100000.0
    
    # Execution settings
    signal_aggregator: str = "weighted_confidence"
    use_regime_filter: bool = True
    max_open_positions: int = 10
    max_correlated_positions: int = 3
    
    # Risk settings
    max_daily_loss_pct: float = 0.05
    max_position_size_pct: float = 0.25
    portfolio_heat_pct: float = 0.5
    
    # Strategy settings
    strategies: List[StrategyConfig] = field(default_factory=list)
    
    # Data settings
    data_provider: str = ""
    data_symbols: List[str] = field(default_factory=list)
    data_timeframes: List[str] = field(default_factory=lambda: ['1h'])
    
    # Logging settings
    log_level: str = "INFO"
    log_to_file: bool = True
    log_file_path: str = "logs/engine.log"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'mode': self.mode,
            'initial_capital': self.initial_capital,
            'signal_aggregator': self.signal_aggregator,
            'use_regime_filter': self.use_regime_filter,
            'max_open_positions': self.max_open_positions,
            'max_correlated_positions': self.max_correlated_positions,
            'max_daily_loss_pct': self.max_daily_loss_pct,
            'max_position_size_pct': self.max_position_size_pct,
            'portfolio_heat_pct': self.portfolio_heat_pct,
            'strategies': [s.to_dict() for s in self.strategies],
            'data_provider': self.data_provider,
            'data_symbols': self.data_symbols,
            'data_timeframes': self.data_timeframes,
            'log_level': self.log_level,
            'log_to_file': self.log_to_file,
            'log_file_path': self.log_file_path
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EngineConfig':
        if 'strategies' in data:
            data['strategies'] = [StrategyConfig.from_dict(s) for s in data['strategies']]
        return cls(**data)
    
    def to_json(self, filepath: Optional[str] = None) -> str:
        """Export to JSON."""
        json_str = json.dumps(self.to_dict(), indent=2)
        
        if filepath:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w') as f:
                f.write(json_str)
        
        return json_str
    
    def to_yaml(self, filepath: Optional[str] = None) -> str:
        """Export to YAML."""
        yaml_str = yaml.dump(self.to_dict(), default_flow_style=False)
        
        if filepath:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w') as f:
                f.write(yaml_str)
        
        return yaml_str
    
    @classmethod
    def from_json(cls, filepath: str) -> 'EngineConfig':
        """Load from JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_yaml(cls, filepath: str) -> 'EngineConfig':
        """Load from YAML."""
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)


# Predefined strategy configurations
def create_ema_crossover_config(
    name: str = "EMA_Crossover",
    fast_period: int = 12,
    slow_period: int = 26,
    symbols: List[str] = None
) -> StrategyConfig:
    """Create EMA Crossover strategy configuration."""
    return StrategyConfig(
        name=name,
        strategy_type=StrategyType.TREND_FOLLOWING,
        description="EMA Crossover trend following strategy",
        symbols=symbols or ['BTCUSDT'],
        timeframes=['1h'],
        parameters=StrategyParameters.from_dict({
            'fast_period': fast_period,
            'slow_period': slow_period,
            'use_volume': True,
            'volume_threshold': 1.2
        }),
        position_sizing=PositionSizingConfig(
            method=PositionSizingMethod.ATR_BASED,
            risk_per_trade=0.02,
            atr_multiple=2.0
        ),
        risk_management=RiskManagementConfig(
            method=RiskManagementMethod.ATR_STOP,
            atr_stop_multiple=2.0
        ),
        indicators=[
            IndicatorConfig(name='ema', params={'period': fast_period}),
            IndicatorConfig(name='ema', params={'period': slow_period}),
            IndicatorConfig(name='atr', params={'period': 14})
        ]
    )


def create_rsi_mean_reversion_config(
    name: str = "RSI_MeanReversion",
    rsi_period: int = 14,
    oversold: int = 30,
    overbought: int = 70,
    symbols: List[str] = None
) -> StrategyConfig:
    """Create RSI Mean Reversion strategy configuration."""
    return StrategyConfig(
        name=name,
        strategy_type=StrategyType.MEAN_REVERSION,
        description="RSI Mean Reversion strategy",
        symbols=symbols or ['BTCUSDT'],
        timeframes=['1h'],
        parameters=StrategyParameters.from_dict({
            'rsi_period': rsi_period,
            'oversold': oversold,
            'overbought': overbought,
            'exit_at_middle': True
        }),
        position_sizing=PositionSizingConfig(
            method=PositionSizingMethod.PERCENT_OF_EQUITY,
            percent_of_equity=0.1
        ),
        risk_management=RiskManagementConfig(
            method=RiskManagementMethod.FIXED_STOP,
            stop_loss_pct=0.02
        ),
        indicators=[
            IndicatorConfig(name='rsi', params={'period': rsi_period})
        ],
        allowed_regimes=['ranging', 'weak_uptrend', 'weak_downtrend']
    )


def create_breakout_config(
    name: str = "Breakout",
    lookback_period: int = 20,
    symbols: List[str] = None
) -> StrategyConfig:
    """Create Breakout strategy configuration."""
    return StrategyConfig(
        name=name,
        strategy_type=StrategyType.BREAKOUT,
        description="Breakout strategy with volume confirmation",
        symbols=symbols or ['BTCUSDT'],
        timeframes=['4h'],
        parameters=StrategyParameters.from_dict({
            'lookback_period': lookback_period,
            'volume_confirm': True,
            'volume_mult': 1.5,
            'breakout_threshold': 0.01
        }),
        position_sizing=PositionSizingConfig(
            method=PositionSizingMethod.ATR_BASED,
            risk_per_trade=0.02
        ),
        risk_management=RiskManagementConfig(
            method=RiskManagementMethod.ATR_STOP,
            atr_stop_multiple=2.0
        ),
        indicators=[
            IndicatorConfig(name='donchian_channels', params={'period': lookback_period}),
            IndicatorConfig(name='atr', params={'period': 14})
        ]
    )


# Export all classes and functions
__all__ = [
    'StrategyType',
    'PositionSizingMethod',
    'RiskManagementMethod',
    'PositionSizingConfig',
    'RiskManagementConfig',
    'IndicatorConfig',
    'FilterConfig',
    'StrategyParameters',
    'StrategyConfig',
    'EngineConfig',
    'create_ema_crossover_config',
    'create_rsi_mean_reversion_config',
    'create_breakout_config'
]
