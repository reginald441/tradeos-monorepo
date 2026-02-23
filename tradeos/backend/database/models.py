"""
TradeOS Database Models
All SQLAlchemy 2.0 async models for the TradeOS platform.
"""

import enum
import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.connection import Base


# ============================================================================
# Enums
# ============================================================================

class UserRole(str, enum.Enum):
    """User role enumeration."""
    ADMIN = "admin"
    USER = "user"
    ANALYST = "analyst"


class SubscriptionTier(str, enum.Enum):
    """Subscription tier enumeration."""
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class SubscriptionStatus(str, enum.Enum):
    """Subscription status enumeration."""
    ACTIVE = "active"
    CANCELED = "canceled"
    PAST_DUE = "past_due"
    UNPAID = "unpaid"
    TRIALING = "trialing"


class StrategyType(str, enum.Enum):
    """Trading strategy type enumeration."""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    SCALPING = "scalping"
    ARBITRAGE = "arbitrage"
    CUSTOM = "custom"


class TradeSide(str, enum.Enum):
    """Trade side enumeration."""
    BUY = "buy"
    SELL = "sell"


class TradeStatus(str, enum.Enum):
    """Trade status enumeration."""
    PENDING = "pending"
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"
    ERROR = "error"


class OrderType(str, enum.Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderStatus(str, enum.Enum):
    """Order status enumeration."""
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class PositionSide(str, enum.Enum):
    """Position side enumeration."""
    LONG = "long"
    SHORT = "short"


class TimeFrame(str, enum.Enum):
    """Candle timeframe enumeration."""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"
    M = "1M"


class ApiKeyPermission(str, enum.Enum):
    """API key permission enumeration."""
    READ = "read"
    TRADE = "trade"
    WITHDRAW = "withdraw"
    ADMIN = "admin"


# ============================================================================
# Models
# ============================================================================

class User(Base):
    """
    User model representing platform users.
    """
    __tablename__ = "users"
    
    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    
    # Authentication fields
    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True
    )
    hashed_password: Mapped[str] = mapped_column(
        String(255),
        nullable=False
    )
    
    # Profile fields
    first_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    last_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    phone: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    
    # Role and subscription
    role: Mapped[UserRole] = mapped_column(
        Enum(UserRole, name="user_role"),
        default=UserRole.USER,
        nullable=False
    )
    subscription_tier: Mapped[SubscriptionTier] = mapped_column(
        Enum(SubscriptionTier, name="subscription_tier"),
        default=SubscriptionTier.FREE,
        nullable=False
    )
    
    # Status fields
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    email_verified_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    last_login_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Security fields
    failed_login_attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    locked_until: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    two_factor_secret: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    two_factor_enabled: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=func.now(),
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=func.now(),
        onupdate=func.now(),
        nullable=False
    )
    
    # Relationships
    strategies: Mapped[List["Strategy"]] = relationship(
        "Strategy",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin"
    )
    trades: Mapped[List["Trade"]] = relationship(
        "Trade",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin"
    )
    positions: Mapped[List["Position"]] = relationship(
        "Position",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin"
    )
    orders: Mapped[List["Order"]] = relationship(
        "Order",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin"
    )
    risk_profile: Mapped[Optional["RiskProfile"]] = relationship(
        "RiskProfile",
        back_populates="user",
        uselist=False,
        cascade="all, delete-orphan",
        lazy="selectin"
    )
    subscription: Mapped[Optional["Subscription"]] = relationship(
        "Subscription",
        back_populates="user",
        uselist=False,
        cascade="all, delete-orphan",
        lazy="selectin"
    )
    api_keys: Mapped[List["ApiKey"]] = relationship(
        "ApiKey",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin"
    )
    backtest_results: Mapped[List["BacktestResult"]] = relationship(
        "BacktestResult",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin"
    )
    
    # Indexes
    __table_args__ = (
        Index("ix_users_email_active", "email", "is_active"),
        Index("ix_users_subscription", "subscription_tier", "is_active"),
        Index("ix_users_created_at", "created_at"),
    )
    
    def __repr__(self) -> str:
        return f"<User(id={self.id}, email={self.email}, role={self.role})>"


class Strategy(Base):
    """
    Trading strategy model.
    """
    __tablename__ = "strategies"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # Strategy info
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    type: Mapped[StrategyType] = mapped_column(
        Enum(StrategyType, name="strategy_type"),
        nullable=False
    )
    
    # Configuration
    config: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    symbols: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)
    timeframes: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)
    
    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_public: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    
    # Performance tracking
    total_trades: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    winning_trades: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_pnl: Mapped[Decimal] = mapped_column(
        Numeric(19, 8),
        default=Decimal("0"),
        nullable=False
    )
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=func.now(),
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=func.now(),
        onupdate=func.now(),
        nullable=False
    )
    last_run_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="strategies")
    trades: Mapped[List["Trade"]] = relationship("Trade", back_populates="strategy")
    backtest_results: Mapped[List["BacktestResult"]] = relationship(
        "BacktestResult",
        back_populates="strategy"
    )
    
    # Indexes
    __table_args__ = (
        Index("ix_strategies_user_active", "user_id", "is_active"),
        Index("ix_strategies_type", "type", "is_active"),
        Index("ix_strategies_created_at", "created_at"),
    )
    
    def __repr__(self) -> str:
        return f"<Strategy(id={self.id}, name={self.name}, type={self.type})>"
    
    @property
    def win_rate(self) -> float:
        """Calculate win rate percentage."""
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100


class Trade(Base):
    """
    Trade model representing executed trades.
    """
    __tablename__ = "trades"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    strategy_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("strategies.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )
    
    # Trade details
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    side: Mapped[TradeSide] = mapped_column(
        Enum(TradeSide, name="trade_side"),
        nullable=False
    )
    
    # Pricing
    entry_price: Mapped[Decimal] = mapped_column(Numeric(19, 8), nullable=False)
    exit_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(19, 8), nullable=True)
    quantity: Mapped[Decimal] = mapped_column(Numeric(19, 8), nullable=False)
    
    # PnL
    realized_pnl: Mapped[Optional[Decimal]] = mapped_column(Numeric(19, 8), nullable=True)
    realized_pnl_pct: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4), nullable=True)
    fees: Mapped[Decimal] = mapped_column(Numeric(19, 8), default=Decimal("0"), nullable=False)
    
    # Status
    status: Mapped[TradeStatus] = mapped_column(
        Enum(TradeStatus, name="trade_status"),
        default=TradeStatus.PENDING,
        nullable=False
    )
    
    # Exchange info
    exchange: Mapped[str] = mapped_column(String(50), default="binance", nullable=False)
    exchange_trade_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    
    # Timestamps
    opened_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    closed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=func.now(),
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=func.now(),
        onupdate=func.now(),
        nullable=False
    )
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="trades")
    strategy: Mapped[Optional["Strategy"]] = relationship("Strategy", back_populates="trades")
    orders: Mapped[List["Order"]] = relationship("Order", back_populates="trade")
    
    # Indexes
    __table_args__ = (
        Index("ix_trades_user_symbol", "user_id", "symbol"),
        Index("ix_trades_user_status", "user_id", "status"),
        Index("ix_trades_strategy", "strategy_id"),
        Index("ix_trades_symbol_time", "symbol", "opened_at"),
        Index("ix_trades_status_time", "status", "opened_at"),
    )
    
    def __repr__(self) -> str:
        return f"<Trade(id={self.id}, symbol={self.symbol}, side={self.side}, status={self.status})>"
    
    @property
    def is_closed(self) -> bool:
        """Check if trade is closed."""
        return self.status == TradeStatus.CLOSED
    
    @property
    def notional_value(self) -> Decimal:
        """Calculate notional value of the trade."""
        return self.entry_price * self.quantity


class Position(Base):
    """
    Position model representing open positions.
    """
    __tablename__ = "positions"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # Position details
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    side: Mapped[PositionSide] = mapped_column(
        Enum(PositionSide, name="position_side"),
        nullable=False
    )
    quantity: Mapped[Decimal] = mapped_column(Numeric(19, 8), nullable=False)
    avg_entry_price: Mapped[Decimal] = mapped_column(Numeric(19, 8), nullable=False)
    
    # Current pricing
    current_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(19, 8), nullable=True)
    mark_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(19, 8), nullable=True)
    
    # PnL
    unrealized_pnl: Mapped[Decimal] = mapped_column(
        Numeric(19, 8),
        default=Decimal("0"),
        nullable=False
    )
    unrealized_pnl_pct: Mapped[Decimal] = mapped_column(
        Numeric(10, 4),
        default=Decimal("0"),
        nullable=False
    )
    realized_pnl: Mapped[Decimal] = mapped_column(
        Numeric(19, 8),
        default=Decimal("0"),
        nullable=False
    )
    
    # Margin info (for leveraged positions)
    leverage: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    margin: Mapped[Optional[Decimal]] = mapped_column(Numeric(19, 8), nullable=True)
    liquidation_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(19, 8), nullable=True)
    
    # Exchange info
    exchange: Mapped[str] = mapped_column(String(50), default="binance", nullable=False)
    
    # Timestamps
    opened_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=func.now(),
        onupdate=func.now(),
        nullable=False
    )
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="positions")
    
    # Indexes
    __table_args__ = (
        Index("ix_positions_user_symbol", "user_id", "symbol", unique=True),
        Index("ix_positions_user_side", "user_id", "side"),
        Index("ix_positions_exchange", "exchange", "symbol"),
    )
    
    def __repr__(self) -> str:
        return f"<Position(id={self.id}, symbol={self.symbol}, side={self.side}, qty={self.quantity})>"
    
    @property
    def notional_value(self) -> Decimal:
        """Calculate notional value of the position."""
        if self.current_price:
            return self.current_price * self.quantity
        return self.avg_entry_price * self.quantity
    
    @property
    is_long(self) -> bool:
        """Check if position is long."""
        return self.side == PositionSide.LONG


class MarketData(Base):
    """
    Market data model for OHLCV candles.
    """
    __tablename__ = "market_data"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    
    # Candle info
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    timeframe: Mapped[TimeFrame] = mapped_column(
        Enum(TimeFrame, name="timeframe"),
        nullable=False
    )
    
    # OHLCV data
    open: Mapped[Decimal] = mapped_column(Numeric(19, 8), nullable=False)
    high: Mapped[Decimal] = mapped_column(Numeric(19, 8), nullable=False)
    low: Mapped[Decimal] = mapped_column(Numeric(19, 8), nullable=False)
    close: Mapped[Decimal] = mapped_column(Numeric(19, 8), nullable=False)
    volume: Mapped[Decimal] = mapped_column(Numeric(19, 8), nullable=False)
    
    # Additional data
    quote_volume: Mapped[Optional[Decimal]] = mapped_column(Numeric(19, 8), nullable=True)
    trades_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    taker_buy_volume: Mapped[Optional[Decimal]] = mapped_column(Numeric(19, 8), nullable=True)
    
    # Timestamp
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=func.now(),
        nullable=False
    )
    
    # Source
    exchange: Mapped[str] = mapped_column(String(50), default="binance", nullable=False)
    
    # Indexes
    __table_args__ = (
        Index("ix_market_data_symbol_tf_time", "symbol", "timeframe", "timestamp", unique=True),
        Index("ix_market_data_time_range", "symbol", "timeframe", "timestamp"),
        Index("ix_market_data_exchange", "exchange", "symbol"),
    )
    
    def __repr__(self) -> str:
        return f"<MarketData(symbol={self.symbol}, tf={self.timeframe}, close={self.close})>"


class Order(Base):
    """
    Order model representing exchange orders.
    """
    __tablename__ = "orders"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    trade_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("trades.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )
    
    # Exchange order info
    exchange_order_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    client_order_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    
    # Order details
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    side: Mapped[TradeSide] = mapped_column(
        Enum(TradeSide, name="order_trade_side"),
        nullable=False
    )
    type: Mapped[OrderType] = mapped_column(
        Enum(OrderType, name="order_type"),
        nullable=False
    )
    
    # Quantity and pricing
    quantity: Mapped[Decimal] = mapped_column(Numeric(19, 8), nullable=False)
    filled_quantity: Mapped[Decimal] = mapped_column(
        Numeric(19, 8),
        default=Decimal("0"),
        nullable=False
    )
    price: Mapped[Optional[Decimal]] = mapped_column(Numeric(19, 8), nullable=True)
    avg_fill_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(19, 8), nullable=True)
    stop_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(19, 8), nullable=True)
    
    # Fees
    fee: Mapped[Decimal] = mapped_column(Numeric(19, 8), default=Decimal("0"), nullable=False)
    fee_asset: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    
    # Status
    status: Mapped[OrderStatus] = mapped_column(
        Enum(OrderStatus, name="order_status"),
        default=OrderStatus.PENDING,
        nullable=False
    )
    
    # Exchange info
    exchange: Mapped[str] = mapped_column(String(50), default="binance", nullable=False)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=func.now(),
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=func.now(),
        onupdate=func.now(),
        nullable=False
    )
    filled_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="orders")
    trade: Mapped[Optional["Trade"]] = relationship("Trade", back_populates="orders")
    
    # Indexes
    __table_args__ = (
        Index("ix_orders_user_status", "user_id", "status"),
        Index("ix_orders_exchange_order", "exchange", "exchange_order_id"),
        Index("ix_orders_symbol_status", "symbol", "status"),
        Index("ix_orders_created_at", "created_at"),
    )
    
    def __repr__(self) -> str:
        return f"<Order(id={self.id}, symbol={self.symbol}, type={self.type}, status={self.status})>"
    
    @property
    def remaining_quantity(self) -> Decimal:
        """Calculate remaining quantity to fill."""
        return self.quantity - self.filled_quantity
    
    @property
    def fill_percentage(self) -> float:
        """Calculate fill percentage."""
        if self.quantity == 0:
            return 0.0
        return float(self.filled_quantity / self.quantity) * 100


class RiskProfile(Base):
    """
    Risk profile model for user risk management settings.
    """
    __tablename__ = "risk_profiles"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True
    )
    
    # Drawdown limits
    max_drawdown_pct: Mapped[Decimal] = mapped_column(
        Numeric(5, 2),
        default=Decimal("10.00"),
        nullable=False
    )
    daily_loss_limit_pct: Mapped[Decimal] = mapped_column(
        Numeric(5, 2),
        default=Decimal("5.00"),
        nullable=False
    )
    
    # Position limits
    risk_per_trade_pct: Mapped[Decimal] = mapped_column(
        Numeric(5, 2),
        default=Decimal("1.00"),
        nullable=False
    )
    max_positions: Mapped[int] = mapped_column(Integer, default=10, nullable=False)
    max_positions_per_symbol: Mapped[int] = mapped_column(Integer, default=2, nullable=False)
    
    # Correlation limits
    correlation_limit: Mapped[Decimal] = mapped_column(
        Numeric(5, 2),
        default=Decimal("0.70"),
        nullable=False
    )
    
    # Leverage limits
    max_leverage: Mapped[int] = mapped_column(Integer, default=5, nullable=False)
    
    # Size limits
    max_position_size_usd: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(19, 2),
        nullable=True
    )
    min_position_size_usd: Mapped[Decimal] = mapped_column(
        Numeric(19, 2),
        default=Decimal("10.00"),
        nullable=False
    )
    
    # Notifications
    notify_on_risk_breach: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    notify_on_position_limit: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=func.now(),
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=func.now(),
        onupdate=func.now(),
        nullable=False
    )
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="risk_profile")
    
    def __repr__(self) -> str:
        return f"<RiskProfile(user_id={self.user_id}, max_drawdown={self.max_drawdown_pct}%)>"


class Subscription(Base):
    """
    Subscription model for user billing/subscription management.
    """
    __tablename__ = "subscriptions"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True
    )
    
    # Subscription details
    tier: Mapped[SubscriptionTier] = mapped_column(
        Enum(SubscriptionTier, name="sub_tier"),
        nullable=False
    )
    status: Mapped[SubscriptionStatus] = mapped_column(
        Enum(SubscriptionStatus, name="sub_status"),
        default=SubscriptionStatus.TRIALING,
        nullable=False
    )
    
    # Stripe integration
    stripe_customer_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    stripe_subscription_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    stripe_price_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    
    # Billing period
    current_period_start: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    current_period_end: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    trial_start: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    trial_end: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Cancellation
    canceled_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    cancel_at_period_end: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    
    # Usage tracking
    api_calls_this_period: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    strategies_limit: Mapped[int] = mapped_column(Integer, default=3, nullable=False)
    backtests_limit: Mapped[int] = mapped_column(Integer, default=10, nullable=False)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=func.now(),
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=func.now(),
        onupdate=func.now(),
        nullable=False
    )
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="subscription")
    
    # Indexes
    __table_args__ = (
        Index("ix_subscriptions_stripe_sub", "stripe_subscription_id"),
        Index("ix_subscriptions_status", "status", "current_period_end"),
    )
    
    def __repr__(self) -> str:
        return f"<Subscription(user_id={self.user_id}, tier={self.tier}, status={self.status})>"
    
    @property
    def is_active(self) -> bool:
        """Check if subscription is active."""
        return self.status in (SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIALING)
    
    @property
    def is_trial(self) -> bool:
        """Check if subscription is in trial period."""
        return self.status == SubscriptionStatus.TRIALING
    
    @property
    def days_until_expiry(self) -> Optional[int]:
        """Calculate days until subscription expires."""
        if self.current_period_end:
            delta = self.current_period_end - datetime.utcnow()
            return max(0, delta.days)
        return None


class ApiKey(Base):
    """
    API key model for user API access.
    """
    __tablename__ = "api_keys"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # Key info
    key_hash: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Permissions
    permissions: Mapped[list] = mapped_column(
        JSONB,
        default=lambda: [ApiKeyPermission.READ.value],
        nullable=False
    )
    
    # Rate limiting (optional override)
    rate_limit_override: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Usage tracking
    request_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    
    # Expiration
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=func.now(),
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=func.now(),
        onupdate=func.now(),
        nullable=False
    )
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="api_keys")
    
    # Indexes
    __table_args__ = (
        Index("ix_api_keys_user_active", "user_id", "is_active"),
        Index("ix_api_keys_expires", "expires_at"),
    )
    
    def __repr__(self) -> str:
        return f"<ApiKey(id={self.id}, name={self.name}, active={self.is_active})>"
    
    @property
    def is_expired(self) -> bool:
        """Check if API key has expired."""
        if self.expires_at:
            return datetime.utcnow() > self.expires_at
        return False
    
    def has_permission(self, permission: ApiKeyPermission) -> bool:
        """Check if API key has specific permission."""
        return permission.value in self.permissions


class BacktestResult(Base):
    """
    Backtest result model for strategy backtesting.
    """
    __tablename__ = "backtest_results"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    strategy_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("strategies.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # Backtest period
    start_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    end_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    
    # Performance metrics
    initial_capital: Mapped[Decimal] = mapped_column(Numeric(19, 2), nullable=False)
    final_capital: Mapped[Decimal] = mapped_column(Numeric(19, 2), nullable=False)
    total_return: Mapped[Decimal] = mapped_column(Numeric(10, 4), nullable=False)
    total_return_pct: Mapped[Decimal] = mapped_column(Numeric(10, 4), nullable=False)
    
    # Risk metrics
    sharpe_ratio: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4), nullable=True)
    sortino_ratio: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4), nullable=True)
    max_drawdown: Mapped[Decimal] = mapped_column(Numeric(10, 4), nullable=False)
    max_drawdown_pct: Mapped[Decimal] = mapped_column(Numeric(10, 4), nullable=False)
    volatility: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4), nullable=True)
    calmar_ratio: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4), nullable=True)
    
    # Trade statistics
    trades_count: Mapped[int] = mapped_column(Integer, nullable=False)
    winning_trades: Mapped[int] = mapped_column(Integer, nullable=False)
    losing_trades: Mapped[int] = mapped_column(Integer, nullable=False)
    win_rate: Mapped[Decimal] = mapped_column(Numeric(5, 2), nullable=False)
    avg_trade_return: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4), nullable=True)
    avg_win: Mapped[Optional[Decimal]] = mapped_column(Numeric(19, 8), nullable=True)
    avg_loss: Mapped[Optional[Decimal]] = mapped_column(Numeric(19, 8), nullable=True)
    profit_factor: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4), nullable=True)
    
    # Configuration
    symbols: Mapped[list] = mapped_column(JSONB, nullable=False)
    timeframes: Mapped[list] = mapped_column(JSONB, nullable=False)
    parameters: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    
    # Results data
    equity_curve: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    trades_log: Mapped[Optional[list]] = mapped_column(JSONB, nullable=True)
    monthly_returns: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    
    # Status
    status: Mapped[str] = mapped_column(String(20), default="completed", nullable=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Timestamps
    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=func.now(),
        nullable=False
    )
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="backtest_results")
    strategy: Mapped["Strategy"] = relationship("Strategy", back_populates="backtest_results")
    
    # Indexes
    __table_args__ = (
        Index("ix_backtests_user_strategy", "user_id", "strategy_id"),
        Index("ix_backtests_date_range", "start_date", "end_date"),
        Index("ix_backtests_return", "total_return_pct"),
        Index("ix_backtests_sharpe", "sharpe_ratio"),
    )
    
    def __repr__(self) -> str:
        return f"<BacktestResult(id={self.id}, strategy_id={self.strategy_id}, return={self.total_return_pct}%)>"
    
    @property
    def duration_days(self) -> int:
        """Calculate backtest duration in days."""
        return (self.end_date - self.start_date).days
    
    @property
    def loss_rate(self) -> Decimal:
        """Calculate loss rate percentage."""
        if self.trades_count == 0:
            return Decimal("0")
        return Decimal(str(self.losing_trades / self.trades_count * 100))


# ============================================================================
# Model exports
# ============================================================================

__all__ = [
    # Base
    "Base",
    # Enums
    "UserRole",
    "SubscriptionTier",
    "SubscriptionStatus",
    "StrategyType",
    "TradeSide",
    "TradeStatus",
    "OrderType",
    "OrderStatus",
    "PositionSide",
    "TimeFrame",
    "ApiKeyPermission",
    # Models
    "User",
    "Strategy",
    "Trade",
    "Position",
    "MarketData",
    "Order",
    "RiskProfile",
    "Subscription",
    "ApiKey",
    "BacktestResult",
]
