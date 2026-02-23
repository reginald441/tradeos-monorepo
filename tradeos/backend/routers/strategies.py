"""
Strategies Router

Handles strategy management, configuration, and performance tracking.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from decimal import Decimal

from ..dependencies.auth import get_current_user
from ..database.models import User, Strategy, BacktestResult
from ..database.connection import get_db_session
from ..strategies.engine.strategy_runner import StrategyRunner
from ..strategies.config.strategy_config import StrategyConfig, StrategyType
from ..saas.subscriptions.tier_manager import TierManager

router = APIRouter(prefix="/strategies", tags=["Strategies"])


# Request/Response Models
class CreateStrategyRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    strategy_type: Literal["trend_following", "mean_reversion", "volatility", "liquidity", "custom"]
    symbol: str
    timeframe: str = "1h"
    config: Dict[str, Any] = Field(default_factory=dict)
    risk_settings: Dict[str, Any] = Field(default_factory=dict)


class StrategyResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    strategy_type: str
    symbol: str
    timeframe: str
    is_active: bool
    config: Dict[str, Any]
    created_at: datetime
    updated_at: Optional[datetime]
    performance_summary: Optional[Dict[str, Any]] = None


class StrategyToggleRequest(BaseModel):
    is_active: bool


class StrategyPerformance(BaseModel):
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    average_trade: float


# Initialize components
strategy_runner = StrategyRunner()
tier_manager = TierManager()


@router.post("", response_model=StrategyResponse)
async def create_strategy(
    request: CreateStrategyRequest,
    current_user: User = Depends(get_current_user)
):
    """Create a new trading strategy."""
    
    # Check subscription limits
    can_create = await tier_manager.can_create_strategy(current_user.id)
    if not can_create:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Strategy limit reached for your subscription tier. Upgrade to create more strategies."
        )
    
    async with get_db_session() as session:
        # Create strategy
        strategy = Strategy(
            user_id=current_user.id,
            name=request.name,
            description=request.description,
            type=request.strategy_type,
            symbol=request.symbol,
            timeframe=request.timeframe,
            is_active=False,  # Start inactive
            config=request.config,
            risk_settings=request.risk_settings
        )
        
        session.add(strategy)
        await session.commit()
        await session.refresh(strategy)
        
        return StrategyResponse(
            id=strategy.id,
            name=strategy.name,
            description=strategy.description,
            strategy_type=strategy.type,
            symbol=strategy.symbol,
            timeframe=strategy.timeframe,
            is_active=strategy.is_active,
            config=strategy.config,
            created_at=strategy.created_at,
            updated_at=strategy.updated_at,
            performance_summary=None
        )


@router.get("", response_model=List[StrategyResponse])
async def get_strategies(
    include_inactive: bool = True,
    current_user: User = Depends(get_current_user)
):
    """Get all strategies for the user."""
    async with get_db_session() as session:
        from sqlalchemy import select
        
        query = select(Strategy).where(Strategy.user_id == current_user.id)
        if not include_inactive:
            query = query.where(Strategy.is_active == True)
        
        result = await session.execute(query.order_by(Strategy.created_at.desc()))
        strategies = result.scalars().all()
        
        return [
            StrategyResponse(
                id=s.id,
                name=s.name,
                description=s.description,
                strategy_type=s.type,
                symbol=s.symbol,
                timeframe=s.timeframe,
                is_active=s.is_active,
                config=s.config,
                created_at=s.created_at,
                updated_at=s.updated_at,
                performance_summary=s.performance_summary
            )
            for s in strategies
        ]


@router.get("/{strategy_id}", response_model=StrategyResponse)
async def get_strategy(
    strategy_id: int,
    current_user: User = Depends(get_current_user)
):
    """Get a specific strategy by ID."""
    async with get_db_session() as session:
        from sqlalchemy import select
        
        result = await session.execute(
            select(Strategy).where(
                Strategy.id == strategy_id,
                Strategy.user_id == current_user.id
            )
        )
        strategy = result.scalar_one_or_none()
        
        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        return StrategyResponse(
            id=strategy.id,
            name=strategy.name,
            description=strategy.description,
            strategy_type=strategy.type,
            symbol=strategy.symbol,
            timeframe=strategy.timeframe,
            is_active=strategy.is_active,
            config=strategy.config,
            created_at=strategy.created_at,
            updated_at=strategy.updated_at,
            performance_summary=strategy.performance_summary
        )


@router.put("/{strategy_id}")
async def update_strategy(
    strategy_id: int,
    request: CreateStrategyRequest,
    current_user: User = Depends(get_current_user)
):
    """Update an existing strategy."""
    async with get_db_session() as session:
        from sqlalchemy import select
        
        result = await session.execute(
            select(Strategy).where(
                Strategy.id == strategy_id,
                Strategy.user_id == current_user.id
            )
        )
        strategy = result.scalar_one_or_none()
        
        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        # Update fields
        strategy.name = request.name
        strategy.description = request.description
        strategy.type = request.strategy_type
        strategy.symbol = request.symbol
        strategy.timeframe = request.timeframe
        strategy.config = request.config
        strategy.risk_settings = request.risk_settings
        strategy.updated_at = datetime.utcnow()
        
        await session.commit()
        await session.refresh(strategy)
        
        return StrategyResponse(
            id=strategy.id,
            name=strategy.name,
            description=strategy.description,
            strategy_type=strategy.type,
            symbol=strategy.symbol,
            timeframe=strategy.timeframe,
            is_active=strategy.is_active,
            config=strategy.config,
            created_at=strategy.created_at,
            updated_at=strategy.updated_at,
            performance_summary=strategy.performance_summary
        )


@router.post("/{strategy_id}/toggle")
async def toggle_strategy(
    strategy_id: int,
    request: StrategyToggleRequest,
    current_user: User = Depends(get_current_user)
):
    """Activate or deactivate a strategy."""
    async with get_db_session() as session:
        from sqlalchemy import select
        
        result = await session.execute(
            select(Strategy).where(
                Strategy.id == strategy_id,
                Strategy.user_id == current_user.id
            )
        )
        strategy = result.scalar_one_or_none()
        
        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        # Check active strategy limits when activating
        if request.is_active:
            can_activate = await tier_manager.can_activate_strategy(current_user.id)
            if not can_activate:
                raise HTTPException(
                    status_code=403,
                    detail="Active strategy limit reached. Upgrade your subscription."
                )
            
            # Start strategy in runner
            await strategy_runner.start_strategy(strategy_id, strategy.config)
        else:
            # Stop strategy in runner
            await strategy_runner.stop_strategy(strategy_id)
        
        strategy.is_active = request.is_active
        strategy.updated_at = datetime.utcnow()
        
        await session.commit()
        
        return {
            "success": True,
            "message": f"Strategy {'activated' if request.is_active else 'deactivated'}",
            "strategy_id": strategy_id,
            "is_active": request.is_active
        }


@router.delete("/{strategy_id}")
async def delete_strategy(
    strategy_id: int,
    current_user: User = Depends(get_current_user)
):
    """Delete a strategy."""
    async with get_db_session() as session:
        from sqlalchemy import select
        
        result = await session.execute(
            select(Strategy).where(
                Strategy.id == strategy_id,
                Strategy.user_id == current_user.id
            )
        )
        strategy = result.scalar_one_or_none()
        
        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        # Stop if active
        if strategy.is_active:
            await strategy_runner.stop_strategy(strategy_id)
        
        await session.delete(strategy)
        await session.commit()
        
        return {"success": True, "message": "Strategy deleted successfully"}


@router.get("/{strategy_id}/performance", response_model=StrategyPerformance)
async def get_strategy_performance(
    strategy_id: int,
    current_user: User = Depends(get_current_user)
):
    """Get performance metrics for a strategy."""
    async with get_db_session() as session:
        from sqlalchemy import select, func
        from ..database.models import Trade
        
        # Verify strategy ownership
        result = await session.execute(
            select(Strategy).where(
                Strategy.id == strategy_id,
                Strategy.user_id == current_user.id
            )
        )
        strategy = result.scalar_one_or_none()
        
        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        # Get trades for this strategy
        result = await session.execute(
            select(Trade).where(Trade.strategy_id == strategy_id)
        )
        trades = result.scalars().all()
        
        if not trades:
            return StrategyPerformance(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                total_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                profit_factor=0.0,
                average_trade=0.0
            )
        
        # Calculate metrics
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.realized_pnl and t.realized_pnl > 0])
        losing_trades = len([t for t in trades if t.realized_pnl and t.realized_pnl <= 0])
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
        
        total_return = sum(t.realized_pnl for t in trades if t.realized_pnl) or 0
        gross_profit = sum(t.realized_pnl for t in trades if t.realized_pnl and t.realized_pnl > 0) or 0
        gross_loss = abs(sum(t.realized_pnl for t in trades if t.realized_pnl and t.realized_pnl < 0)) or 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        average_trade = total_return / total_trades if total_trades > 0 else 0
        
        # Get backtest results for Sharpe and drawdown
        result = await session.execute(
            select(BacktestResult).where(BacktestResult.strategy_id == strategy_id)
        )
        backtest = result.scalar_one_or_none()
        
        sharpe = backtest.sharpe_ratio if backtest else 0.0
        max_dd = backtest.max_drawdown if backtest else 0.0
        
        return StrategyPerformance(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_return=float(total_return),
            sharpe_ratio=float(sharpe),
            max_drawdown=float(max_dd),
            profit_factor=float(profit_factor),
            average_trade=float(average_trade)
        )


@router.get("/presets/list")
async def get_strategy_presets():
    """Get list of available strategy presets."""
    return {
        "trend_following": [
            {
                "id": "ema_crossover",
                "name": "EMA Crossover",
                "description": "Classic EMA crossover strategy",
                "default_params": {
                    "fast_ema": 12,
                    "slow_ema": 26,
                    "signal_ema": 9
                }
            },
            {
                "id": "adx_trend",
                "name": "ADX Trend Follower",
                "description": "Trend following with ADX confirmation",
                "default_params": {
                    "adx_period": 14,
                    "adx_threshold": 25,
                    "ema_period": 20
                }
            }
        ],
        "mean_reversion": [
            {
                "id": "rsi_reversion",
                "name": "RSI Mean Reversion",
                "description": "Buy oversold, sell overbought",
                "default_params": {
                    "rsi_period": 14,
                    "oversold": 30,
                    "overbought": 70
                }
            },
            {
                "id": "bb_reversion",
                "name": "Bollinger Band Reversion",
                "description": "Mean reversion using Bollinger Bands",
                "default_params": {
                    "bb_period": 20,
                    "bb_std": 2.0
                }
            }
        ],
        "volatility": [
            {
                "id": "volatility_breakout",
                "name": "Volatility Breakout",
                "description": "Trade volatility expansions",
                "default_params": {
                    "atr_period": 14,
                    "atr_multiplier": 2.0
                }
            }
        ],
        "liquidity": [
            {
                "id": "liquidity_sweep",
                "name": "Liquidity Sweep",
                "description": "Trade liquidity sweeps and stop hunts",
                "default_params": {
                    "lookback": 20,
                    "sweep_threshold": 0.5
                }
            }
        ]
    }
