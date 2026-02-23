"""
Trading Router

Handles trade execution, position management, and order operations.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from decimal import Decimal
from datetime import datetime

from ..dependencies.auth import get_current_user
from ..database.models import User, Trade, Position, Order
from ..database.connection import get_db_session
from ..execution.order_manager import OrderManager
from ..execution.paper_trading import PaperTradingExchange
from ..risk.engine.risk_engine import create_risk_engine
from ..risk.models.risk_profile import TradeRequest, PortfolioState

router = APIRouter(prefix="/trading", tags=["Trading"])


# Request/Response Models
class PlaceOrderRequest(BaseModel):
    symbol: str
    side: Literal["buy", "sell"]
    order_type: Literal["market", "limit", "stop", "stop_limit"] = "market"
    quantity: Decimal = Field(..., gt=0)
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    time_in_force: Literal["gtc", "ioc", "fok"] = "gtc"
    strategy_id: Optional[int] = None


class OrderResponse(BaseModel):
    id: int
    symbol: str
    side: str
    order_type: str
    quantity: Decimal
    price: Optional[Decimal]
    status: str
    exchange_order_id: Optional[str]
    created_at: datetime


class PositionResponse(BaseModel):
    id: int
    symbol: str
    side: str
    quantity: Decimal
    avg_entry_price: Decimal
    unrealized_pnl: Decimal
    opened_at: datetime


class TradeResponse(BaseModel):
    id: int
    symbol: str
    side: str
    entry_price: Decimal
    exit_price: Optional[Decimal]
    quantity: Decimal
    realized_pnl: Optional[Decimal]
    status: str
    opened_at: datetime
    closed_at: Optional[datetime]


class ClosePositionRequest(BaseModel):
    position_id: int
    quantity: Optional[Decimal] = None  # None = close full position


# Initialize components
order_manager = OrderManager()
risk_engine = create_risk_engine(tier="pro")


@router.post("/orders", response_model=OrderResponse)
async def place_order(
    request: PlaceOrderRequest,
    current_user: User = Depends(get_current_user)
):
    """Place a new trading order with risk validation."""
    
    # Validate order with risk engine
    trade_request = TradeRequest(
        symbol=request.symbol,
        side=request.side,
        quantity=float(request.quantity),
        price=float(request.price) if request.price else None,
        order_type=request.order_type
    )
    
    # Get portfolio state for risk validation
    portfolio_state = await _get_portfolio_state(current_user.id)
    
    # Validate trade
    validation_result = risk_engine.validate_trade(trade_request, portfolio_state)
    
    if not validation_result.is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Risk validation failed: {validation_result.rejection_reason}"
        )
    
    # Place order through order manager
    try:
        order = await order_manager.place_order(
            user_id=current_user.id,
            symbol=request.symbol,
            side=request.side,
            order_type=request.order_type,
            quantity=float(request.quantity),
            price=float(request.price) if request.price else None,
            stop_price=float(request.stop_price) if request.stop_price else None,
            time_in_force=request.time_in_force,
            strategy_id=request.strategy_id
        )
        
        return OrderResponse(
            id=order.id,
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            quantity=Decimal(str(order.quantity)),
            price=Decimal(str(order.price)) if order.price else None,
            status=order.status,
            exchange_order_id=order.exchange_order_id,
            created_at=order.created_at
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Order placement failed: {str(e)}")


@router.get("/orders", response_model=List[OrderResponse])
async def get_orders(
    status: Optional[str] = None,
    symbol: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """Get user's orders with optional filtering."""
    orders = await order_manager.get_user_orders(
        user_id=current_user.id,
        status=status,
        symbol=symbol
    )
    
    return [
        OrderResponse(
            id=order.id,
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            quantity=Decimal(str(order.quantity)),
            price=Decimal(str(order.price)) if order.price else None,
            status=order.status,
            exchange_order_id=order.exchange_order_id,
            created_at=order.created_at
        )
        for order in orders
    ]


@router.delete("/orders/{order_id}")
async def cancel_order(
    order_id: int,
    current_user: User = Depends(get_current_user)
):
    """Cancel an open order."""
    try:
        success = await order_manager.cancel_order(
            user_id=current_user.id,
            order_id=order_id
        )
        
        if success:
            return {"success": True, "message": "Order cancelled successfully"}
        else:
            raise HTTPException(status_code=400, detail="Order could not be cancelled")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cancel failed: {str(e)}")


@router.get("/positions", response_model=List[PositionResponse])
async def get_positions(
    current_user: User = Depends(get_current_user)
):
    """Get all open positions for the user."""
    async with get_db_session() as session:
        from sqlalchemy import select
        result = await session.execute(
            select(Position).where(
                Position.user_id == current_user.id,
                Position.is_open == True
            )
        )
        positions = result.scalars().all()
        
        return [
            PositionResponse(
                id=pos.id,
                symbol=pos.symbol,
                side=pos.side,
                quantity=Decimal(str(pos.quantity)),
                avg_entry_price=Decimal(str(pos.avg_entry_price)),
                unrealized_pnl=Decimal(str(pos.unrealized_pnl or 0)),
                opened_at=pos.opened_at
            )
            for pos in positions
        ]


@router.post("/positions/close")
async def close_position(
    request: ClosePositionRequest,
    current_user: User = Depends(get_current_user)
):
    """Close an open position."""
    try:
        order = await order_manager.close_position(
            user_id=current_user.id,
            position_id=request.position_id,
            quantity=float(request.quantity) if request.quantity else None
        )
        
        return {
            "success": True,
            "message": "Position close order placed",
            "order_id": order.id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Close position failed: {str(e)}")


@router.get("/trades", response_model=List[TradeResponse])
async def get_trades(
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(get_current_user)
):
    """Get trade history for the user."""
    async with get_db_session() as session:
        from sqlalchemy import select
        result = await session.execute(
            select(Trade)
            .where(Trade.user_id == current_user.id)
            .order_by(Trade.opened_at.desc())
            .limit(limit)
            .offset(offset)
        )
        trades = result.scalars().all()
        
        return [
            TradeResponse(
                id=trade.id,
                symbol=trade.symbol,
                side=trade.side,
                entry_price=Decimal(str(trade.entry_price)),
                exit_price=Decimal(str(trade.exit_price)) if trade.exit_price else None,
                quantity=Decimal(str(trade.quantity)),
                realized_pnl=Decimal(str(trade.realized_pnl)) if trade.realized_pnl else None,
                status=trade.status,
                opened_at=trade.opened_at,
                closed_at=trade.closed_at
            )
            for trade in trades
        ]


@router.get("/portfolio")
async def get_portfolio_summary(
    current_user: User = Depends(get_current_user)
):
    """Get portfolio summary including balance, PnL, and exposure."""
    async with get_db_session() as session:
        from sqlalchemy import select, func
        
        # Get positions
        result = await session.execute(
            select(Position).where(
                Position.user_id == current_user.id,
                Position.is_open == True
            )
        )
        positions = result.scalars().all()
        
        # Calculate totals
        total_exposure = sum(pos.quantity * pos.avg_entry_price for pos in positions)
        total_unrealized_pnl = sum(pos.unrealized_pnl or 0 for pos in positions)
        
        # Get today's PnL
        from datetime import date
        today = date.today()
        result = await session.execute(
            select(func.sum(Trade.realized_pnl)).where(
                Trade.user_id == current_user.id,
                func.date(Trade.closed_at) == today
            )
        )
        today_pnl = result.scalar() or 0
        
        return {
            "total_positions": len(positions),
            "total_exposure": float(total_exposure),
            "unrealized_pnl": float(total_unrealized_pnl),
            "today_pnl": float(today_pnl),
            "positions_by_symbol": [
                {
                    "symbol": pos.symbol,
                    "side": pos.side,
                    "quantity": float(pos.quantity),
                    "avg_entry": float(pos.avg_entry_price),
                    "unrealized_pnl": float(pos.unrealized_pnl or 0)
                }
                for pos in positions
            ]
        }


async def _get_portfolio_state(user_id: int) -> PortfolioState:
    """Get current portfolio state for risk validation."""
    async with get_db_session() as session:
        from sqlalchemy import select, func
        
        # Get positions
        result = await session.execute(
            select(Position).where(
                Position.user_id == user_id,
                Position.is_open == True
            )
        )
        positions = result.scalars().all()
        
        # Get account balance (simplified - would come from exchange)
        result = await session.execute(
            select(func.sum(Trade.realized_pnl)).where(Trade.user_id == user_id)
        )
        total_pnl = result.scalar() or 0
        
        # Calculate exposure by symbol
        exposure_by_symbol = {}
        for pos in positions:
            exposure_by_symbol[pos.symbol] = exposure_by_symbol.get(pos.symbol, 0) + \
                float(pos.quantity * pos.avg_entry_price)
        
        return PortfolioState(
            total_equity=100000.0 + float(total_pnl),  # Starting balance + PnL
            available_margin=50000.0,  # Simplified
            open_positions={pos.symbol: {
                "side": pos.side,
                "quantity": float(pos.quantity),
                "entry_price": float(pos.avg_entry_price)
            } for pos in positions},
            exposure_by_symbol=exposure_by_symbol,
            daily_pnl=float(total_pnl),
            max_drawdown=0.0  # Would be calculated from equity curve
        )
