"""
TradeOS Order Models
====================
Comprehensive order models for all trading instruments.
Supports crypto, forex, and equity orders with full state management.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Union


class OrderType(Enum):
    """Supported order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    TAKE_PROFIT = "take_profit"
    TAKE_PROFIT_LIMIT = "take_profit_limit"
    ICEBERG = "iceberg"
    FOK = "fill_or_kill"
    IOC = "immediate_or_cancel"


class OrderSide(Enum):
    """Order side - buy or sell."""
    BUY = "buy"
    SELL = "sell"


class OrderState(Enum):
    """
    Order state machine states.
    
    State transitions:
    - PENDING -> OPEN (order accepted by exchange)
    - PENDING -> REJECTED (order rejected)
    - OPEN -> PARTIALLY_FILLED (partial fill)
    - OPEN -> FILLED (complete fill)
    - OPEN -> CANCELLED (user cancellation)
    - OPEN -> EXPIRED (time limit reached)
    - PARTIALLY_FILLED -> FILLED (remaining quantity filled)
    - PARTIALLY_FILLED -> CANCELLED (cancel remaining)
    """
    PENDING = "pending"           # Order submitted, awaiting exchange confirmation
    OPEN = "open"                 # Order confirmed and active on exchange
    PARTIALLY_FILLED = "partially_filled"  # Some quantity filled
    FILLED = "filled"             # Complete fill
    CANCELLED = "cancelled"       # User cancelled
    REJECTED = "rejected"         # Exchange rejected
    EXPIRED = "expired"           # Time in force expired
    PENDING_CANCEL = "pending_cancel"  # Cancellation requested
    ERROR = "error"               # Error state


class TimeInForce(Enum):
    """Time in force options."""
    GTC = "good_till_cancelled"   # Good till cancelled
    IOC = "immediate_or_cancel"   # Immediate or cancel
    FOK = "fill_or_kill"          # Fill or kill
    GTD = "good_till_date"        # Good till date
    DAY = "day"                   # Day order
    OPG = "at_opening"            # At opening
    CLS = "at_close"              # At close


class PositionSide(Enum):
    """Position side for derivatives."""
    LONG = "long"
    SHORT = "short"
    BOTH = "both"  # Hedge mode


@dataclass
class OrderFill:
    """Represents a single fill/execution of an order."""
    fill_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: Decimal
    price: Decimal
    fee: Decimal
    fee_currency: str
    timestamp: datetime
    exchange_fill_id: Optional[str] = None
    is_maker: bool = False
    
    def __post_init__(self):
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp.replace('Z', '+00:00'))
    
    @property
    def notional_value(self) -> Decimal:
        """Calculate notional value of this fill."""
        return self.quantity * self.price
    
    @property
    def net_value(self) -> Decimal:
        """Calculate net value after fees."""
        if self.side == OrderSide.BUY:
            return self.notional_value + self.fee
        return self.notional_value - self.fee


@dataclass
class Order:
    """
    Comprehensive order model for all trading types.
    
    Supports:
    - Spot trading (crypto)
    - Margin trading
    - Futures/Perpetuals
    - Options
    - Forex
    """
    # Core identifiers
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    client_order_id: Optional[str] = None
    exchange_order_id: Optional[str] = None
    
    # Order details
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    quantity: Decimal = field(default_factory=lambda: Decimal("0"))
    
    # Price levels
    price: Optional[Decimal] = None           # Limit price
    stop_price: Optional[Decimal] = None      # Stop/trigger price
    trailing_delta: Optional[Decimal] = None  # Trailing stop delta
    
    # Execution details
    time_in_force: TimeInForce = TimeInForce.GTC
    expire_time: Optional[datetime] = None
    
    # Fill tracking
    filled_quantity: Decimal = field(default_factory=lambda: Decimal("0"))
    remaining_quantity: Decimal = field(default_factory=lambda: Decimal("0"))
    average_fill_price: Optional[Decimal] = None
    fills: List[OrderFill] = field(default_factory=list)
    
    # State
    state: OrderState = OrderState.PENDING
    state_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Fees
    total_fee: Decimal = field(default_factory=lambda: Decimal("0"))
    fee_currency: str = ""
    
    # Derivatives specific
    position_side: Optional[PositionSide] = None
    leverage: Optional[Decimal] = None
    reduce_only: bool = False
    post_only: bool = False
    
    # Exchange info
    exchange: str = ""
    account_id: Optional[str] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    
    # Additional properties
    tags: Dict[str, Any] = field(default_factory=dict)
    strategy_id: Optional[str] = None
    parent_order_id: Optional[str] = None  # For OCO, bracket orders
    child_order_ids: List[str] = field(default_factory=list)
    
    # Slippage tracking
    expected_price: Optional[Decimal] = None
    slippage_bps: Optional[Decimal] = None  # Slippage in basis points
    
    def __post_init__(self):
        """Initialize computed fields."""
        if isinstance(self.quantity, (int, float, str)):
            self.quantity = Decimal(str(self.quantity))
        if isinstance(self.filled_quantity, (int, float, str)):
            self.filled_quantity = Decimal(str(self.filled_quantity))
        if isinstance(self.remaining_quantity, (int, float, str)):
            self.remaining_quantity = Decimal(str(self.remaining_quantity))
        if isinstance(self.total_fee, (int, float, str)):
            self.total_fee = Decimal(str(self.total_fee))
            
        # Set remaining quantity if not set
        if self.remaining_quantity == 0 and self.quantity > 0:
            self.remaining_quantity = self.quantity - self.filled_quantity
            
        # Record initial state
        if not self.state_history:
            self._record_state_change(self.state, "Order created")
    
    def _record_state_change(self, new_state: OrderState, reason: str = ""):
        """Record a state change in history."""
        self.state_history.append({
            "state": new_state.value,
            "timestamp": datetime.utcnow().isoformat(),
            "reason": reason
        })
    
    def update_state(self, new_state: OrderState, reason: str = ""):
        """Update order state with history tracking."""
        old_state = self.state
        self.state = new_state
        self.updated_at = datetime.utcnow()
        self._record_state_change(new_state, reason)
        
        # Update timestamps
        if new_state == OrderState.OPEN and not self.submitted_at:
            self.submitted_at = datetime.utcnow()
        elif new_state == OrderState.FILLED and not self.filled_at:
            self.filled_at = datetime.utcnow()
        elif new_state == OrderState.CANCELLED and not self.cancelled_at:
            self.cancelled_at = datetime.utcnow()
    
    def add_fill(self, fill: OrderFill):
        """Add a fill to this order."""
        self.fills.append(fill)
        self.filled_quantity += fill.quantity
        self.remaining_quantity = self.quantity - self.filled_quantity
        self.total_fee += fill.fee
        
        # Update average fill price
        if self.filled_quantity > 0:
            total_value = sum(f.quantity * f.price for f in self.fills)
            self.average_fill_price = total_value / self.filled_quantity
        
        # Calculate slippage
        if self.expected_price and self.average_fill_price:
            price_diff = abs(self.average_fill_price - self.expected_price)
            self.slippage_bps = (price_diff / self.expected_price) * Decimal("10000")
        
        # Update state based on fill
        if self.remaining_quantity <= 0:
            self.update_state(OrderState.FILLED, "Order completely filled")
        else:
            self.update_state(OrderState.PARTIALLY_FILLED, f"Partial fill: {fill.quantity}")
    
    @property
    def is_active(self) -> bool:
        """Check if order is still active."""
        return self.state in (OrderState.PENDING, OrderState.OPEN, OrderState.PARTIALLY_FILLED)
    
    @property
    def is_complete(self) -> bool:
        """Check if order is complete (filled, cancelled, rejected, expired)."""
        return self.state in (OrderState.FILLED, OrderState.CANCELLED, 
                             OrderState.REJECTED, OrderState.EXPIRED)
    
    @property
    def fill_percentage(self) -> Decimal:
        """Calculate fill percentage."""
        if self.quantity == 0:
            return Decimal("0")
        return (self.filled_quantity / self.quantity) * Decimal("100")
    
    @property
    def notional_value(self) -> Optional[Decimal]:
        """Calculate notional value of the order."""
        if self.average_fill_price:
            return self.filled_quantity * self.average_fill_price
        if self.price:
            return self.quantity * self.price
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary."""
        return {
            "order_id": self.order_id,
            "client_order_id": self.client_order_id,
            "exchange_order_id": self.exchange_order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "quantity": str(self.quantity),
            "price": str(self.price) if self.price else None,
            "stop_price": str(self.stop_price) if self.stop_price else None,
            "time_in_force": self.time_in_force.value,
            "filled_quantity": str(self.filled_quantity),
            "remaining_quantity": str(self.remaining_quantity),
            "average_fill_price": str(self.average_fill_price) if self.average_fill_price else None,
            "state": self.state.value,
            "total_fee": str(self.total_fee),
            "fee_currency": self.fee_currency,
            "exchange": self.exchange,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "fill_percentage": str(self.fill_percentage),
            "slippage_bps": str(self.slippage_bps) if self.slippage_bps else None,
            "is_active": self.is_active,
            "is_complete": self.is_complete,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Order:
        """Create order from dictionary."""
        order = cls(
            order_id=data.get("order_id", str(uuid.uuid4())),
            client_order_id=data.get("client_order_id"),
            exchange_order_id=data.get("exchange_order_id"),
            symbol=data.get("symbol", ""),
            side=OrderSide(data.get("side", "buy")),
            order_type=OrderType(data.get("order_type", "market")),
            quantity=Decimal(data.get("quantity", "0")),
            price=Decimal(data["price"]) if data.get("price") else None,
            stop_price=Decimal(data["stop_price"]) if data.get("stop_price") else None,
            time_in_force=TimeInForce(data.get("time_in_force", "good_till_cancelled")),
            filled_quantity=Decimal(data.get("filled_quantity", "0")),
            remaining_quantity=Decimal(data.get("remaining_quantity", "0")),
            average_fill_price=Decimal(data["average_fill_price"]) if data.get("average_fill_price") else None,
            state=OrderState(data.get("state", "pending")),
            total_fee=Decimal(data.get("total_fee", "0")),
            fee_currency=data.get("fee_currency", ""),
            exchange=data.get("exchange", ""),
            strategy_id=data.get("strategy_id"),
            expected_price=Decimal(data["expected_price"]) if data.get("expected_price") else None,
        )
        
        if "created_at" in data:
            order.created_at = datetime.fromisoformat(data["created_at"].replace('Z', '+00:00'))
        if "updated_at" in data:
            order.updated_at = datetime.fromisoformat(data["updated_at"].replace('Z', '+00:00'))
            
        return order


@dataclass
class OrderRequest:
    """Request to create a new order."""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    time_in_force: TimeInForce = TimeInForce.GTC
    client_order_id: Optional[str] = None
    post_only: bool = False
    reduce_only: bool = False
    leverage: Optional[Decimal] = None
    expire_time: Optional[datetime] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    
    def to_order(self) -> Order:
        """Convert request to Order object."""
        return Order(
            client_order_id=self.client_order_id or str(uuid.uuid4()),
            symbol=self.symbol,
            side=self.side,
            order_type=self.order_type,
            quantity=self.quantity,
            price=self.price,
            stop_price=self.stop_price,
            time_in_force=self.time_in_force,
            post_only=self.post_only,
            reduce_only=self.reduce_only,
            leverage=self.leverage,
            expire_time=self.expire_time,
            tags=self.tags,
            expected_price=self.price,
        )


@dataclass
class OrderBatch:
    """Batch of orders for bulk operations."""
    batch_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    orders: List[Order] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def add_order(self, order: Order):
        """Add order to batch."""
        self.orders.append(order)
    
    @property
    def all_complete(self) -> bool:
        """Check if all orders are complete."""
        return all(o.is_complete for o in self.orders)
    
    @property
    def active_orders(self) -> List[Order]:
        """Get active orders."""
        return [o for o in self.orders if o.is_active]


@dataclass
class BracketOrder:
    """Bracket order with entry, take profit, and stop loss."""
    entry_order: Order
    take_profit_order: Optional[Order] = None
    stop_loss_order: Optional[Order] = None
    bracket_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def __post_init__(self):
        """Link orders together."""
        self.entry_order.parent_order_id = self.bracket_id
        if self.take_profit_order:
            self.take_profit_order.parent_order_id = self.bracket_id
        if self.stop_loss_order:
            self.stop_loss_order.parent_order_id = self.bracket_id


# Event types for order updates
class OrderEventType(Enum):
    """Types of order events."""
    CREATED = "created"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    CANCEL_REJECTED = "cancel_rejected"
    EXPIRED = "expired"
    ERROR = "error"
    MODIFY_REQUESTED = "modify_requested"
    MODIFIED = "modified"


@dataclass
class OrderEvent:
    """Event representing an order update."""
    event_type: OrderEventType
    order_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    data: Dict[str, Any] = field(default_factory=dict)
    exchange: str = ""
    error_message: Optional[str] = None
