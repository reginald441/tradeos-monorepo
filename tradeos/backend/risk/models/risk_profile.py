"""
Risk Profile Data Models for TradeOS Risk Engine

Defines all data structures for risk management including:
- Risk profiles and limits
- Position sizing parameters
- Drawdown controls
- Exposure tracking
- Validation results
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Tuple
import numpy as np


class RiskLevel(Enum):
    """Risk severity levels"""
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()
    EMERGENCY = auto()


class SubscriptionTier(Enum):
    """Subscription tiers with different risk limits"""
    FREE = "free"
    BASIC = "basic"
    PRO = "pro"
    ENTERPRISE = "enterprise"
    INSTITUTIONAL = "institutional"


class PositionSizingMethod(Enum):
    """Available position sizing methods"""
    FIXED_FRACTIONAL = "fixed_fractional"
    KELLY_CRITERION = "kelly_criterion"
    VOLATILITY_BASED = "volatility_based"
    RISK_PER_TRADE = "risk_per_trade"
    OPTIMAL_F = "optimal_f"


class VaRMethod(Enum):
    """Value at Risk calculation methods"""
    HISTORICAL = "historical"
    PARAMETRIC = "parametric"
    MONTE_CARLO = "monte_carlo"


@dataclass
class RiskLimits:
    """Comprehensive risk limits configuration"""
    # Position sizing limits
    max_position_size_pct: Decimal = Decimal("0.10")  # Max 10% per position
    max_risk_per_trade_pct: Decimal = Decimal("0.02")  # Max 2% risk per trade
    min_position_size: Decimal = Decimal("10.0")  # Minimum position value
    
    # Portfolio exposure limits
    max_total_exposure_pct: Decimal = Decimal("1.0")  # Max 100% total exposure
    max_single_asset_exposure_pct: Decimal = Decimal("0.20")  # Max 20% per asset
    max_sector_exposure_pct: Decimal = Decimal("0.40")  # Max 40% per sector
    max_leverage: Decimal = Decimal("3.0")  # Max 3x leverage
    max_margin_usage_pct: Decimal = Decimal("0.80")  # Max 80% margin usage
    
    # Drawdown limits
    max_daily_loss_pct: Decimal = Decimal("0.05")  # Max 5% daily loss
    max_weekly_loss_pct: Decimal = Decimal("0.10")  # Max 10% weekly loss
    max_drawdown_pct: Decimal = Decimal("0.20")  # Max 20% drawdown
    circuit_breaker_drawdown_pct: Decimal = Decimal("0.25")  # Hard stop at 25%
    
    # Correlation limits
    max_correlation_exposure: Decimal = Decimal("0.70")  # Max correlation
    max_portfolio_heat: Decimal = Decimal("0.50")  # Max portfolio heat
    
    # VaR limits
    max_daily_var_pct: Decimal = Decimal("0.05")  # Max 5% daily VaR
    var_confidence_level: Decimal = Decimal("0.95")  # 95% confidence
    
    # Kill switch triggers
    kill_switch_daily_loss_pct: Decimal = Decimal("0.10")  # Kill at 10% daily loss
    kill_switch_drawdown_pct: Decimal = Decimal("0.30")  # Kill at 30% drawdown
    
    # Trading halts
    trading_halt_cooldown_minutes: int = 30
    auto_resume_after_halt: bool = False


@dataclass
class PositionSizingParams:
    """Parameters for position sizing calculations"""
    method: PositionSizingMethod = PositionSizingMethod.RISK_PER_TRADE
    
    # Fixed fractional parameters
    fixed_fraction_pct: Decimal = Decimal("0.02")
    
    # Kelly criterion parameters
    kelly_fraction: Decimal = Decimal("0.25")  # Half-Kelly for safety
    win_rate: Optional[Decimal] = None
    avg_win_loss_ratio: Optional[Decimal] = None
    
    # Volatility-based parameters
    atr_period: int = 14
    atr_multiplier: Decimal = Decimal("2.0")
    risk_atr_multiple: Decimal = Decimal("1.0")
    
    # Risk-per-trade parameters
    risk_per_trade_pct: Decimal = Decimal("0.01")  # 1% risk per trade
    
    # Optimal f parameters
    optimal_f_lookback: int = 50
    optimal_f_fraction: Decimal = Decimal("0.50")  # Conservative optimal f


@dataclass
class TradeRequest:
    """Trade request for risk validation"""
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: Decimal
    price: Optional[Decimal] = None  # None for market orders
    order_type: str = "market"
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    strategy_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ValidationResult:
    """Result of risk validation"""
    is_valid: bool
    rejection_reason: Optional[str] = None
    risk_level: RiskLevel = RiskLevel.LOW
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __bool__(self):
        return self.is_valid


@dataclass
class Position:
    """Position data for risk calculations"""
    symbol: str
    quantity: Decimal
    avg_entry_price: Decimal
    current_price: Decimal
    side: str  # 'long' or 'short'
    unrealized_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    margin_used: Decimal = Decimal("0")
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def market_value(self) -> Decimal:
        """Calculate current market value"""
        return abs(self.quantity) * self.current_price
    
    @property
    def position_pnl_pct(self) -> Decimal:
        """Calculate position P&L percentage"""
        if self.avg_entry_price == 0:
            return Decimal("0")
        return (self.current_price - self.avg_entry_price) / self.avg_entry_price


@dataclass
class PortfolioState:
    """Current portfolio state for risk monitoring"""
    total_equity: Decimal
    cash_balance: Decimal
    buying_power: Decimal
    margin_used: Decimal
    margin_available: Decimal
    positions: Dict[str, Position] = field(default_factory=dict)
    daily_pnl: Decimal = Decimal("0")
    daily_pnl_pct: Decimal = Decimal("0")
    weekly_pnl: Decimal = Decimal("0")
    weekly_pnl_pct: Decimal = Decimal("0")
    total_pnl: Decimal = Decimal("0")
    total_pnl_pct: Decimal = Decimal("0")
    peak_equity: Decimal = Decimal("0")
    current_drawdown_pct: Decimal = Decimal("0")
    max_drawdown_pct: Decimal = Decimal("0")
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def total_exposure(self) -> Decimal:
        """Calculate total portfolio exposure"""
        return sum(pos.market_value for pos in self.positions.values())
    
    @property
    def total_exposure_pct(self) -> Decimal:
        """Calculate total exposure as percentage of equity"""
        if self.total_equity == 0:
            return Decimal("0")
        return self.total_exposure / self.total_equity
    
    @property
    def leverage(self) -> Decimal:
        """Calculate current leverage"""
        if self.total_equity == 0:
            return Decimal("0")
        return self.total_exposure / self.total_equity
    
    @property
    def margin_usage_pct(self) -> Decimal:
        """Calculate margin usage percentage"""
        total_margin = self.margin_used + self.margin_available
        if total_margin == 0:
            return Decimal("0")
        return self.margin_used / total_margin


@dataclass
class ExposureByAsset:
    """Exposure breakdown by asset"""
    symbol: str
    market_value: Decimal
    exposure_pct: Decimal
    position_count: int
    unrealized_pnl: Decimal


@dataclass
class ExposureBySector:
    """Exposure breakdown by sector"""
    sector: str
    market_value: Decimal
    exposure_pct: Decimal
    symbols: List[str] = field(default_factory=list)


@dataclass
class CorrelationMatrix:
    """Correlation matrix data"""
    symbols: List[str]
    correlation_matrix: np.ndarray
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def get_correlation(self, symbol1: str, symbol2: str) -> float:
        """Get correlation between two symbols"""
        if symbol1 not in self.symbols or symbol2 not in self.symbols:
            return 0.0
        idx1 = self.symbols.index(symbol1)
        idx2 = self.symbols.index(symbol2)
        return float(self.correlation_matrix[idx1, idx2])


@dataclass
class VaRResult:
    """Value at Risk calculation result"""
    var_value: Decimal
    var_pct: Decimal
    cvar_value: Decimal  # Conditional VaR (Expected Shortfall)
    cvar_pct: Decimal
    confidence_level: Decimal
    method: VaRMethod
    holding_period_days: int
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class KillSwitchState:
    """Kill switch state tracking"""
    is_active: bool = False
    triggered_at: Optional[datetime] = None
    triggered_by: Optional[str] = None
    reason: Optional[str] = None
    can_resume_at: Optional[datetime] = None
    manual_override: bool = False
    trigger_count_today: int = 0
    last_trigger_date: Optional[datetime] = None


@dataclass
class DrawdownState:
    """Drawdown tracking state"""
    peak_equity: Decimal = Decimal("0")
    trough_equity: Decimal = Decimal("0")
    current_drawdown_pct: Decimal = Decimal("0")
    max_drawdown_pct: Decimal = Decimal("0")
    max_drawdown_start: Optional[datetime] = None
    max_drawdown_end: Optional[datetime] = None
    in_drawdown: bool = False
    drawdown_start: Optional[datetime] = None
    daily_loss_pct: Decimal = Decimal("0")
    weekly_loss_pct: Decimal = Decimal("0")
    trading_halted: bool = False
    halt_reason: Optional[str] = None
    halt_until: Optional[datetime] = None


@dataclass
class RiskReport:
    """Comprehensive risk report"""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    portfolio_state: Optional[PortfolioState] = None
    drawdown_state: Optional[DrawdownState] = None
    kill_switch_state: Optional[KillSwitchState] = None
    var_result: Optional[VaRResult] = None
    exposure_by_asset: Dict[str, ExposureByAsset] = field(default_factory=dict)
    exposure_by_sector: Dict[str, ExposureBySector] = field(default_factory=dict)
    correlation_matrix: Optional[CorrelationMatrix] = None
    portfolio_heat: Decimal = Decimal("0")
    risk_level: RiskLevel = RiskLevel.LOW
    alerts: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class SizingRecommendation:
    """Position sizing recommendation"""
    recommended_quantity: Decimal
    recommended_value: Decimal
    risk_amount: Decimal
    risk_pct: Decimal
    method: PositionSizingMethod
    confidence_score: Decimal  # 0-1 confidence in recommendation
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
