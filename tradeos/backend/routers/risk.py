"""
Risk Router

Handles risk management, exposure monitoring, and risk configuration.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from decimal import Decimal
from datetime import datetime

from ..dependencies.auth import get_current_user
from ..database.models import User, RiskProfile
from ..database.connection import get_db_session
from ..risk.engine.risk_engine import create_risk_engine
from ..risk.models.risk_profile import RiskProfile as RiskProfileModel

router = APIRouter(prefix="/risk", tags=["Risk Management"])


# Request/Response Models
class RiskProfileRequest(BaseModel):
    max_drawdown_pct: float = Field(..., ge=1, le=100)
    risk_per_trade_pct: float = Field(..., ge=0.1, le=100)
    max_positions: int = Field(..., ge=1, le=1000)
    max_correlation: float = Field(..., ge=0, le=1)
    daily_loss_limit_pct: float = Field(..., ge=0.1, le=100)
    weekly_loss_limit_pct: float = Field(..., ge=0.1, le=100)
    position_sizing_method: str = "risk_per_trade"
    leverage_max: float = Field(default=1.0, ge=1, le=100)
    margin_call_threshold: float = Field(default=0.3, ge=0.1, le=1)


class RiskProfileResponse(BaseModel):
    id: int
    max_drawdown_pct: float
    risk_per_trade_pct: float
    max_positions: int
    max_correlation: float
    daily_loss_limit_pct: float
    weekly_loss_limit_pct: float
    position_sizing_method: str
    leverage_max: float
    margin_call_threshold: float
    created_at: datetime
    updated_at: Optional[datetime]


class ExposureResponse(BaseModel):
    symbol: str
    exposure: float
    exposure_pct: float
    side: str
    quantity: float


class RiskMetricsResponse(BaseModel):
    portfolio_value: float
    total_exposure: float
    exposure_pct: float
    available_margin: float
    daily_pnl: float
    daily_pnl_pct: float
    current_drawdown: float
    var_95: float
    var_99: float
    sharpe_ratio: float
    sortino_ratio: float
    positions_count: int


class KillSwitchRequest(BaseModel):
    reason: Optional[str] = None


class KillSwitchResponse(BaseModel):
    activated: bool
    activated_at: datetime
    reason: Optional[str]
    can_recover: bool


# Initialize risk engine
risk_engine = create_risk_engine(tier="pro")


@router.get("/profile", response_model=RiskProfileResponse)
async def get_risk_profile(current_user: User = Depends(get_current_user)):
    """Get user's risk profile configuration."""
    async with get_db_session() as session:
        from sqlalchemy import select
        
        result = await session.execute(
            select(RiskProfile).where(RiskProfile.user_id == current_user.id)
        )
        profile = result.scalar_one_or_none()
        
        if not profile:
            # Create default profile
            profile = RiskProfile(
                user_id=current_user.id,
                max_drawdown_pct=20.0,
                risk_per_trade_pct=2.0,
                max_positions=10,
                max_correlation=0.7,
                daily_loss_limit_pct=5.0,
                weekly_loss_limit_pct=10.0,
                position_sizing_method="risk_per_trade",
                leverage_max=1.0,
                margin_call_threshold=0.3
            )
            session.add(profile)
            await session.commit()
            await session.refresh(profile)
        
        return RiskProfileResponse(
            id=profile.id,
            max_drawdown_pct=profile.max_drawdown_pct,
            risk_per_trade_pct=profile.risk_per_trade_pct,
            max_positions=profile.max_positions,
            max_correlation=profile.max_correlation,
            daily_loss_limit_pct=profile.daily_loss_limit_pct,
            weekly_loss_limit_pct=profile.weekly_loss_limit_pct,
            position_sizing_method=profile.position_sizing_method,
            leverage_max=profile.leverage_max,
            margin_call_threshold=profile.margin_call_threshold,
            created_at=profile.created_at,
            updated_at=profile.updated_at
        )


@router.put("/profile", response_model=RiskProfileResponse)
async def update_risk_profile(
    request: RiskProfileRequest,
    current_user: User = Depends(get_current_user)
):
    """Update user's risk profile configuration."""
    async with get_db_session() as session:
        from sqlalchemy import select
        
        result = await session.execute(
            select(RiskProfile).where(RiskProfile.user_id == current_user.id)
        )
        profile = result.scalar_one_or_none()
        
        if not profile:
            profile = RiskProfile(user_id=current_user.id)
            session.add(profile)
        
        # Update fields
        profile.max_drawdown_pct = request.max_drawdown_pct
        profile.risk_per_trade_pct = request.risk_per_trade_pct
        profile.max_positions = request.max_positions
        profile.max_correlation = request.max_correlation
        profile.daily_loss_limit_pct = request.daily_loss_limit_pct
        profile.weekly_loss_limit_pct = request.weekly_loss_limit_pct
        profile.position_sizing_method = request.position_sizing_method
        profile.leverage_max = request.leverage_max
        profile.margin_call_threshold = request.margin_call_threshold
        profile.updated_at = datetime.utcnow()
        
        await session.commit()
        await session.refresh(profile)
        
        return RiskProfileResponse(
            id=profile.id,
            max_drawdown_pct=profile.max_drawdown_pct,
            risk_per_trade_pct=profile.risk_per_trade_pct,
            max_positions=profile.max_positions,
            max_correlation=profile.max_correlation,
            daily_loss_limit_pct=profile.daily_loss_limit_pct,
            weekly_loss_limit_pct=profile.weekly_loss_limit_pct,
            position_sizing_method=profile.position_sizing_method,
            leverage_max=profile.leverage_max,
            margin_call_threshold=profile.margin_call_threshold,
            created_at=profile.created_at,
            updated_at=profile.updated_at
        )


@router.get("/exposure", response_model=List[ExposureResponse])
async def get_exposure(current_user: User = Depends(get_current_user)):
    """Get current exposure by symbol."""
    async with get_db_session() as session:
        from sqlalchemy import select
        from ..database.models import Position
        
        result = await session.execute(
            select(Position).where(
                Position.user_id == current_user.id,
                Position.is_open == True
            )
        )
        positions = result.scalars().all()
        
        # Calculate total portfolio value for percentage
        total_exposure = sum(pos.quantity * pos.avg_entry_price for pos in positions)
        
        exposure_list = []
        for pos in positions:
            exposure = float(pos.quantity * pos.avg_entry_price)
            exposure_pct = (exposure / total_exposure * 100) if total_exposure > 0 else 0
            
            exposure_list.append(ExposureResponse(
                symbol=pos.symbol,
                exposure=exposure,
                exposure_pct=exposure_pct,
                side=pos.side,
                quantity=float(pos.quantity)
            ))
        
        return sorted(exposure_list, key=lambda x: x.exposure, reverse=True)


@router.get("/metrics", response_model=RiskMetricsResponse)
async def get_risk_metrics(current_user: User = Depends(get_current_user)):
    """Get comprehensive risk metrics."""
    async with get_db_session() as session:
        from sqlalchemy import select, func
        from ..database.models import Position, Trade
        from datetime import date, timedelta
        
        # Get positions
        result = await session.execute(
            select(Position).where(
                Position.user_id == current_user.id,
                Position.is_open == True
            )
        )
        positions = result.scalars().all()
        
        # Get portfolio value (simplified - would use actual balance)
        portfolio_value = 100000.0  # Starting balance
        
        # Calculate exposure
        total_exposure = sum(float(pos.quantity * pos.avg_entry_price) for pos in positions)
        exposure_pct = (total_exposure / portfolio_value * 100) if portfolio_value > 0 else 0
        
        # Get daily PnL
        today = date.today()
        result = await session.execute(
            select(func.sum(Trade.realized_pnl)).where(
                Trade.user_id == current_user.id,
                func.date(Trade.closed_at) == today
            )
        )
        daily_pnl = float(result.scalar() or 0)
        daily_pnl_pct = (daily_pnl / portfolio_value * 100) if portfolio_value > 0 else 0
        
        # Calculate VaR (simplified)
        var_95 = total_exposure * 0.05  # Simplified 5% VaR
        var_99 = total_exposure * 0.01  # Simplified 1% VaR
        
        # Get all trades for Sharpe calculation
        result = await session.execute(
            select(Trade).where(
                Trade.user_id == current_user.id,
                Trade.status == "closed"
            ).order_by(Trade.closed_at.desc()).limit(100)
        )
        trades = result.scalars().all()
        
        # Calculate returns for Sharpe
        returns = [float(t.realized_pnl or 0) / portfolio_value for t in trades]
        
        import numpy as np
        if len(returns) > 1:
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe = (avg_return / std_return * np.sqrt(252)) if std_return > 0 else 0
            
            # Sortino (downside deviation only)
            downside_returns = [r for r in returns if r < 0]
            downside_std = np.std(downside_returns) if downside_returns else 1
            sortino = (avg_return / downside_std * np.sqrt(252)) if downside_std > 0 else 0
        else:
            sharpe = 0
            sortino = 0
        
        # Calculate current drawdown
        result = await session.execute(
            select(func.sum(Trade.realized_pnl)).where(Trade.user_id == current_user.id)
        )
        total_pnl = float(result.scalar() or 0)
        
        # Peak equity would be tracked separately
        peak_equity = portfolio_value + max(0, total_pnl)
        current_equity = portfolio_value + total_pnl
        drawdown = ((peak_equity - current_equity) / peak_equity * 100) if peak_equity > 0 else 0
        
        return RiskMetricsResponse(
            portfolio_value=portfolio_value,
            total_exposure=total_exposure,
            exposure_pct=exposure_pct,
            available_margin=portfolio_value - total_exposure,
            daily_pnl=daily_pnl,
            daily_pnl_pct=daily_pnl_pct,
            current_drawdown=drawdown,
            var_95=var_95,
            var_99=var_99,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            positions_count=len(positions)
        )


@router.get("/correlation-matrix")
async def get_correlation_matrix(current_user: User = Depends(get_current_user)):
    """Get correlation matrix for open positions."""
    async with get_db_session() as session:
        from sqlalchemy import select
        from ..database.models import Position, MarketData
        
        # Get positions
        result = await session.execute(
            select(Position).where(
                Position.user_id == current_user.id,
                Position.is_open == True
            )
        )
        positions = result.scalars().all()
        
        symbols = list(set(pos.symbol for pos in positions))
        
        if len(symbols) < 2:
            return {"symbols": symbols, "matrix": []}
        
        # Get historical data for correlation calculation
        import pandas as pd
        import numpy as np
        
        price_data = {}
        for symbol in symbols:
            result = await session.execute(
                select(MarketData).where(
                    MarketData.symbol == symbol,
                    MarketData.timeframe == "1h"
                ).order_by(MarketData.timestamp.desc()).limit(100)
            )
            data = result.scalars().all()
            
            if data:
                df = pd.DataFrame([{
                    'timestamp': d.timestamp,
                    'close': float(d.close)
                } for d in data])
                df['returns'] = df['close'].pct_change()
                price_data[symbol] = df['returns'].dropna()
        
        # Calculate correlation matrix
        if len(price_data) >= 2:
            returns_df = pd.DataFrame(price_data)
            corr_matrix = returns_df.corr().fillna(0)
            
            return {
                "symbols": symbols,
                "matrix": corr_matrix.values.tolist(),
                "warnings": [
                    f"High correlation: {symbols[i]} - {symbols[j]}"
                    for i in range(len(symbols))
                    for j in range(i+1, len(symbols))
                    if corr_matrix.iloc[i, j] > 0.8
                ]
            }
        
        return {"symbols": symbols, "matrix": [], "warnings": []}


@router.post("/kill-switch")
async def activate_kill_switch(
    request: KillSwitchRequest,
    current_user: User = Depends(get_current_user)
):
    """Activate emergency kill switch to halt all trading."""
    from ..risk.kill_switch import KillSwitch
    
    kill_switch = KillSwitch()
    
    # Activate kill switch
    kill_switch.activate(
        reason=request.reason or "Manual activation",
        user_id=current_user.id
    )
    
    # Close all open positions
    async with get_db_session() as session:
        from sqlalchemy import select
        from ..database.models import Position
        from ..execution.order_manager import OrderManager
        
        order_manager = OrderManager()
        
        result = await session.execute(
            select(Position).where(
                Position.user_id == current_user.id,
                Position.is_open == True
            )
        )
        positions = result.scalars().all()
        
        for pos in positions:
            await order_manager.close_position(
                user_id=current_user.id,
                position_id=pos.id
            )
    
    return KillSwitchResponse(
        activated=True,
        activated_at=datetime.utcnow(),
        reason=request.reason,
        can_recover=True
    )


@router.post("/kill-switch/recover")
async def recover_kill_switch(current_user: User = Depends(get_current_user)):
    """Recover from kill switch activation."""
    from ..risk.kill_switch import KillSwitch
    
    kill_switch = KillSwitch()
    
    success = kill_switch.deactivate(user_id=current_user.id)
    
    if success:
        return {
            "success": True,
            "message": "Kill switch deactivated. Trading resumed.",
            "deactivated_at": datetime.utcnow().isoformat()
        }
    else:
        raise HTTPException(status_code=400, detail="Kill switch could not be deactivated")


@router.get("/limits")
async def get_risk_limits(current_user: User = Depends(get_current_user)):
    """Get current risk limits status."""
    async with get_db_session() as session:
        from sqlalchemy import select, func
        from ..database.models import RiskProfile, Trade, Position
        from datetime import date
        
        # Get risk profile
        result = await session.execute(
            select(RiskProfile).where(RiskProfile.user_id == current_user.id)
        )
        profile = result.scalar_one_or_none()
        
        if not profile:
            profile = RiskProfile(user_id=current_user.id)
        
        # Calculate current usage
        today = date.today()
        
        # Daily loss
        result = await session.execute(
            select(func.sum(Trade.realized_pnl)).where(
                Trade.user_id == current_user.id,
                func.date(Trade.closed_at) == today
            )
        )
        daily_pnl = float(result.scalar() or 0)
        daily_loss = abs(min(0, daily_pnl))
        daily_loss_pct = (daily_loss / 100000 * 100)  # Assuming 100k portfolio
        
        # Position count
        result = await session.execute(
            select(func.count(Position.id)).where(
                Position.user_id == current_user.id,
                Position.is_open == True
            )
        )
        position_count = result.scalar() or 0
        
        return {
            "limits": {
                "max_drawdown": {"limit": profile.max_drawdown_pct, "current": 0, "status": "ok"},
                "daily_loss": {"limit": profile.daily_loss_limit_pct, "current": daily_loss_pct, "status": "ok" if daily_loss_pct < profile.daily_loss_limit_pct else "breached"},
                "max_positions": {"limit": profile.max_positions, "current": position_count, "status": "ok" if position_count < profile.max_positions else "at_limit"},
                "risk_per_trade": {"limit": profile.risk_per_trade_pct, "current": 0, "status": "ok"}
            },
            "all_ok": daily_loss_pct < profile.daily_loss_limit_pct and position_count < profile.max_positions
        }
