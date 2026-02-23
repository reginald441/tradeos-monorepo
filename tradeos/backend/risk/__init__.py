"""
TradeOS Risk Engine - Capital Protection Core

A comprehensive risk management system for algorithmic trading.

Modules:
    position_sizing: Position sizing methods (Kelly, ATR, Risk-per-trade, etc.)
    drawdown_control: Drawdown monitoring and circuit breakers
    exposure_manager: Portfolio exposure tracking and limits
    correlation_monitor: Cross-asset correlation and portfolio heat
    kill_switch: Emergency trading halt controls
    var_calculator: Value at Risk calculations
    risk_validator: Pre-trade validation gatekeeper
    engine: Main risk engine coordinator

Usage:
    >>> from tradeos.backend.risk import create_risk_engine
    >>> engine = create_risk_engine(tier="pro")
    >>> engine.initialize()
    >>>
    >>> # Validate a trade
    >>> result = engine.validate_trade(trade_request, portfolio_state)
    >>> if result.is_valid:
    ...     execute_trade(trade_request)
    ... else:
    ...     print(f"Trade rejected: {result.rejection_reason}")
"""

__version__ = "1.0.0"
__author__ = "TradeOS"

# Import main components for easy access
from .engine.risk_engine import RiskEngine, create_risk_engine
from .risk_validator import RiskValidator, validate_trade
from .position_sizing import PositionSizer
from .drawdown_control import DrawdownController
from .exposure_manager import ExposureManager
from .correlation_monitor import CorrelationMonitor
from .kill_switch import KillSwitch, KillSwitchTrigger, CircuitBreakerPanel
from .var_calculator import VaRCalculator

# Import models
from .models.risk_profile import (
    TradeRequest,
    ValidationResult,
    PortfolioState,
    Position,
    RiskReport,
    RiskLevel,
    RiskLimits,
    PositionSizingParams,
    PositionSizingMethod,
    SizingRecommendation,
    SubscriptionTier,
    VaRMethod,
    VaRResult,
    KillSwitchState,
    DrawdownState,
    ExposureByAsset,
    ExposureBySector,
    CorrelationMatrix,
)

# Import config
from .config.risk_limits import (
    get_risk_limits,
    get_tier_from_string,
    DEFAULT_RISK_LIMITS,
    FREE_TIER_LIMITS,
    BASIC_TIER_LIMITS,
    PRO_TIER_LIMITS,
    ENTERPRISE_TIER_LIMITS,
    INSTITUTIONAL_TIER_LIMITS,
)

__all__ = [
    # Main classes
    "RiskEngine",
    "create_risk_engine",
    "RiskValidator",
    "validate_trade",
    "PositionSizer",
    "DrawdownController",
    "ExposureManager",
    "CorrelationMonitor",
    "KillSwitch",
    "KillSwitchTrigger",
    "CircuitBreakerPanel",
    "VaRCalculator",
    
    # Models
    "TradeRequest",
    "ValidationResult",
    "PortfolioState",
    "Position",
    "RiskReport",
    "RiskLevel",
    "RiskLimits",
    "PositionSizingParams",
    "PositionSizingMethod",
    "SizingRecommendation",
    "SubscriptionTier",
    "VaRMethod",
    "VaRResult",
    "KillSwitchState",
    "DrawdownState",
    "ExposureByAsset",
    "ExposureBySector",
    "CorrelationMatrix",
    
    # Config
    "get_risk_limits",
    "get_tier_from_string",
    "DEFAULT_RISK_LIMITS",
    "FREE_TIER_LIMITS",
    "BASIC_TIER_LIMITS",
    "PRO_TIER_LIMITS",
    "ENTERPRISE_TIER_LIMITS",
    "INSTITUTIONAL_TIER_LIMITS",
]
