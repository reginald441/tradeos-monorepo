"""
Backtest Router

Handles strategy backtesting and performance analysis.
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime, date
from decimal import Decimal

from ..dependencies.auth import get_current_user
from ..database.models import User, Strategy, BacktestResult
from ..database.connection import get_db_session
from ..strategies.engine.strategy_runner import StrategyRunner
from ..quant.monte_carlo.engine import MonteCarloEngine
from ..quant.analytics.metrics import calculate_all_metrics
from ..saas.subscriptions.tier_manager import TierManager

router = APIRouter(prefix="/backtest", tags=["Backtesting"])


# Request/Response Models
class RunBacktestRequest(BaseModel):
    strategy_id: int
    start_date: date
    end_date: date
    initial_capital: float = Field(default=100000.0, gt=0)
    commission_pct: float = Field(default=0.1, ge=0)
    slippage_pct: float = Field(default=0.05, ge=0)
    position_size_pct: float = Field(default=10.0, gt=0, le=100)
    run_monte_carlo: bool = False
    monte_carlo_runs: int = Field(default=1000, ge=100, le=10000)


class BacktestResultResponse(BaseModel):
    id: int
    strategy_id: int
    start_date: date
    end_date: date
    initial_capital: float
    final_capital: float
    total_return: float
    total_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_trade: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    status: str
    created_at: datetime
    monte_carlo_results: Optional[Dict[str, Any]] = None


class BacktestListItem(BaseModel):
    id: int
    strategy_name: str
    start_date: date
    end_date: date
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    status: str
    created_at: datetime


class EquityCurvePoint(BaseModel):
    timestamp: datetime
    equity: float
    drawdown: float
    trades_count: int


# Initialize components
strategy_runner = StrategyRunner()
tier_manager = TierManager()


@router.post("/run", response_model=BacktestResultResponse)
async def run_backtest(
    request: RunBacktestRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """Run a backtest for a strategy."""
    
    # Check subscription limits
    can_run = await tier_manager.can_run_backtest(current_user.id)
    if not can_run:
        raise HTTPException(
            status_code=403,
            detail="Backtest limit reached for your subscription tier. Upgrade for more backtests."
        )
    
    async with get_db_session() as session:
        from sqlalchemy import select
        
        # Verify strategy ownership
        result = await session.execute(
            select(Strategy).where(
                Strategy.id == request.strategy_id,
                Strategy.user_id == current_user.id
            )
        )
        strategy = result.scalar_one_or_none()
        
        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        # Create backtest record
        backtest = BacktestResult(
            user_id=current_user.id,
            strategy_id=request.strategy_id,
            start_date=request.start_date,
            end_date=request.end_date,
            initial_capital=request.initial_capital,
            status="running"
        )
        session.add(backtest)
        await session.commit()
        await session.refresh(backtest)
        
        # Run backtest asynchronously
        background_tasks.add_task(
            _execute_backtest,
            backtest_id=backtest.id,
            strategy_id=request.strategy_id,
            strategy_config=strategy.config,
            start_date=request.start_date,
            end_date=request.end_date,
            initial_capital=request.initial_capital,
            commission_pct=request.commission_pct,
            slippage_pct=request.slippage_pct,
            position_size_pct=request.position_size_pct,
            run_monte_carlo=request.run_monte_carlo,
            monte_carlo_runs=request.monte_carlo_runs
        )
        
        return BacktestResultResponse(
            id=backtest.id,
            strategy_id=backtest.strategy_id,
            start_date=backtest.start_date,
            end_date=backtest.end_date,
            initial_capital=backtest.initial_capital,
            final_capital=backtest.initial_capital,
            total_return=0.0,
            total_return_pct=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            max_drawdown_pct=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            avg_trade=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            largest_win=0.0,
            largest_loss=0.0,
            status="running",
            created_at=backtest.created_at,
            monte_carlo_results=None
        )


async def _execute_backtest(
    backtest_id: int,
    strategy_id: int,
    strategy_config: Dict[str, Any],
    start_date: date,
    end_date: date,
    initial_capital: float,
    commission_pct: float,
    slippage_pct: float,
    position_size_pct: float,
    run_monte_carlo: bool,
    monte_carlo_runs: int
):
    """Execute backtest logic."""
    import pandas as pd
    import numpy as np
    from sqlalchemy import select
    from ..database.models import MarketData, Trade
    
    async with get_db_session() as session:
        try:
            # Get historical data
            result = await session.execute(
                select(MarketData).where(
                    MarketData.symbol == strategy_config.get("symbol", "BTCUSDT"),
                    MarketData.timeframe == strategy_config.get("timeframe", "1h"),
                    MarketData.timestamp >= datetime.combine(start_date, datetime.min.time()),
                    MarketData.timestamp <= datetime.combine(end_date, datetime.max.time())
                ).order_by(MarketData.timestamp)
            )
            market_data = result.scalars().all()
            
            if not market_data:
                # Update backtest with error
                result = await session.execute(
                    select(BacktestResult).where(BacktestResult.id == backtest_id)
                )
                backtest = result.scalar_one()
                backtest.status = "error"
                backtest.error_message = "No market data available for the specified period"
                await session.commit()
                return
            
            # Convert to DataFrame
            df = pd.DataFrame([{
                'timestamp': d.timestamp,
                'open': float(d.open),
                'high': float(d.high),
                'low': float(d.low),
                'close': float(d.close),
                'volume': float(d.volume)
            } for d in market_data])
            
            # Run strategy simulation
            trades = []
            equity = initial_capital
            equity_curve = [equity]
            position = None
            
            # Simple EMA crossover example
            df['ema_fast'] = df['close'].ewm(span=strategy_config.get("fast_ema", 12)).mean()
            df['ema_slow'] = df['close'].ewm(span=strategy_config.get("slow_ema", 26)).mean()
            
            for i in range(1, len(df)):
                current = df.iloc[i]
                prev = df.iloc[i-1]
                
                # Entry logic
                if prev['ema_fast'] <= prev['ema_slow'] and current['ema_fast'] > current['ema_slow']:
                    if position is None:
                        # Buy signal
                        position_size = (equity * position_size_pct / 100) / current['close']
                        position = {
                            'entry_price': current['close'],
                            'size': position_size,
                            'side': 'long',
                            'entry_time': current['timestamp']
                        }
                
                # Exit logic
                elif prev['ema_fast'] >= prev['ema_slow'] and current['ema_fast'] < current['ema_slow']:
                    if position and position['side'] == 'long':
                        # Sell signal
                        exit_price = current['close']
                        pnl = (exit_price - position['entry_price']) * position['size']
                        
                        # Apply commission and slippage
                        commission = (position['entry_price'] * position['size'] + exit_price * position['size']) * commission_pct / 100
                        slippage = (position['entry_price'] * position['size'] + exit_price * position['size']) * slippage_pct / 100
                        pnl -= (commission + slippage)
                        
                        equity += pnl
                        
                        trades.append({
                            'entry_price': position['entry_price'],
                            'exit_price': exit_price,
                            'size': position['size'],
                            'pnl': pnl,
                            'entry_time': position['entry_time'],
                            'exit_time': current['timestamp']
                        })
                        
                        position = None
                
                # Update equity curve
                if position:
                    unrealized = (current['close'] - position['entry_price']) * position['size']
                    equity_curve.append(equity + unrealized)
                else:
                    equity_curve.append(equity)
            
            # Close any open position at the end
            if position:
                exit_price = df.iloc[-1]['close']
                pnl = (exit_price - position['entry_price']) * position['size']
                equity += pnl
                trades.append({
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'size': position['size'],
                    'pnl': pnl,
                    'entry_time': position['entry_time'],
                    'exit_time': df.iloc[-1]['timestamp']
                })
            
            # Calculate metrics
            total_return = equity - initial_capital
            total_return_pct = (total_return / initial_capital) * 100
            
            winning_trades = [t for t in trades if t['pnl'] > 0]
            losing_trades = [t for t in trades if t['pnl'] <= 0]
            
            win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
            
            gross_profit = sum(t['pnl'] for t in winning_trades)
            gross_loss = abs(sum(t['pnl'] for t in losing_trades))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            # Calculate drawdown
            equity_series = pd.Series(equity_curve)
            rolling_max = equity_series.expanding().max()
            drawdown = equity_series - rolling_max
            max_drawdown = abs(drawdown.min())
            max_drawdown_pct = (max_drawdown / rolling_max.max() * 100) if rolling_max.max() > 0 else 0
            
            # Calculate Sharpe ratio
            returns = pd.Series(equity_curve).pct_change().dropna()
            sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
            
            # Calculate Sortino
            downside_returns = returns[returns < 0]
            sortino = (returns.mean() / downside_returns.std() * np.sqrt(252)) if len(downside_returns) > 0 and downside_returns.std() > 0 else 0
            
            # Run Monte Carlo if requested
            monte_carlo_results = None
            if run_monte_carlo and trades:
                mc_engine = MonteCarloEngine()
                trade_returns = [t['pnl'] for t in trades]
                monte_carlo_results = mc_engine.simulate_equity_curves(
                    returns=trade_returns,
                    initial_capital=initial_capital,
                    n_simulations=monte_carlo_runs
                )
                monte_carlo_results = {
                    "wor_case_final": float(monte_carlo_results['worst_case_final']),
                    "best_case_final": float(monte_carlo_results['best_case_final']),
                    "median_final": float(monte_carlo_results['median_final']),
                    "risk_of_ruin": float(monte_carlo_results['risk_of_ruin']),
                    "confidence_interval_95": [
                        float(monte_carlo_results['confidence_interval_95'][0]),
                        float(monte_carlo_results['confidence_interval_95'][1])
                    ]
                }
            
            # Update backtest result
            result = await session.execute(
                select(BacktestResult).where(BacktestResult.id == backtest_id)
            )
            backtest = result.scalar_one()
            
            backtest.status = "completed"
            backtest.final_capital = equity
            backtest.total_return = total_return
            backtest.total_return_pct = total_return_pct
            backtest.sharpe_ratio = sharpe
            backtest.sortino_ratio = sortino
            backtest.max_drawdown = max_drawdown
            backtest.max_drawdown_pct = max_drawdown_pct
            backtest.win_rate = win_rate
            backtest.profit_factor = profit_factor
            backtest.total_trades = len(trades)
            backtest.winning_trades = len(winning_trades)
            backtest.losing_trades = len(losing_trades)
            backtest.avg_trade = total_return / len(trades) if trades else 0
            backtest.avg_win = gross_profit / len(winning_trades) if winning_trades else 0
            backtest.avg_loss = gross_loss / len(losing_trades) if losing_trades else 0
            backtest.largest_win = max([t['pnl'] for t in winning_trades]) if winning_trades else 0
            backtest.largest_loss = min([t['pnl'] for t in losing_trades]) if losing_trades else 0
            backtest.equity_curve = equity_curve
            backtest.monte_carlo_results = monte_carlo_results
            backtest.completed_at = datetime.utcnow()
            
            await session.commit()
            
        except Exception as e:
            # Update backtest with error
            result = await session.execute(
                select(BacktestResult).where(BacktestResult.id == backtest_id)
            )
            backtest = result.scalar_one()
            backtest.status = "error"
            backtest.error_message = str(e)
            await session.commit()


@router.get("/results", response_model=List[BacktestListItem])
async def get_backtest_results(
    strategy_id: Optional[int] = None,
    limit: int = 20,
    offset: int = 0,
    current_user: User = Depends(get_current_user)
):
    """Get backtest results for the user."""
    async with get_db_session() as session:
        from sqlalchemy import select
        
        query = select(BacktestResult, Strategy.name).join(
            Strategy, BacktestResult.strategy_id == Strategy.id
        ).where(BacktestResult.user_id == current_user.id)
        
        if strategy_id:
            query = query.where(BacktestResult.strategy_id == strategy_id)
        
        query = query.order_by(BacktestResult.created_at.desc()).limit(limit).offset(offset)
        
        result = await session.execute(query)
        rows = result.all()
        
        return [
            BacktestListItem(
                id=row.BacktestResult.id,
                strategy_name=row.name,
                start_date=row.BacktestResult.start_date,
                end_date=row.BacktestResult.end_date,
                total_return_pct=row.BacktestResult.total_return_pct or 0,
                sharpe_ratio=row.BacktestResult.sharpe_ratio or 0,
                max_drawdown_pct=row.BacktestResult.max_drawdown_pct or 0,
                status=row.BacktestResult.status,
                created_at=row.BacktestResult.created_at
            )
            for row in rows
        ]


@router.get("/results/{backtest_id}", response_model=BacktestResultResponse)
async def get_backtest_detail(
    backtest_id: int,
    current_user: User = Depends(get_current_user)
):
    """Get detailed backtest results."""
    async with get_db_session() as session:
        from sqlalchemy import select
        
        result = await session.execute(
            select(BacktestResult).where(
                BacktestResult.id == backtest_id,
                BacktestResult.user_id == current_user.id
            )
        )
        backtest = result.scalar_one_or_none()
        
        if not backtest:
            raise HTTPException(status_code=404, detail="Backtest not found")
        
        return BacktestResultResponse(
            id=backtest.id,
            strategy_id=backtest.strategy_id,
            start_date=backtest.start_date,
            end_date=backtest.end_date,
            initial_capital=backtest.initial_capital,
            final_capital=backtest.final_capital or backtest.initial_capital,
            total_return=backtest.total_return or 0,
            total_return_pct=backtest.total_return_pct or 0,
            sharpe_ratio=backtest.sharpe_ratio or 0,
            sortino_ratio=backtest.sortino_ratio or 0,
            max_drawdown=backtest.max_drawdown or 0,
            max_drawdown_pct=backtest.max_drawdown_pct or 0,
            win_rate=backtest.win_rate or 0,
            profit_factor=backtest.profit_factor or 0,
            total_trades=backtest.total_trades or 0,
            winning_trades=backtest.winning_trades or 0,
            losing_trades=backtest.losing_trades or 0,
            avg_trade=backtest.avg_trade or 0,
            avg_win=backtest.avg_win or 0,
            avg_loss=backtest.avg_loss or 0,
            largest_win=backtest.largest_win or 0,
            largest_loss=backtest.largest_loss or 0,
            status=backtest.status,
            created_at=backtest.created_at,
            monte_carlo_results=backtest.monte_carlo_results
        )


@router.get("/results/{backtest_id}/equity-curve", response_model=List[EquityCurvePoint])
async def get_equity_curve(
    backtest_id: int,
    current_user: User = Depends(get_current_user)
):
    """Get equity curve data for a backtest."""
    async with get_db_session() as session:
        from sqlalchemy import select
        
        result = await session.execute(
            select(BacktestResult).where(
                BacktestResult.id == backtest_id,
                BacktestResult.user_id == current_user.id
            )
        )
        backtest = result.scalar_one_or_none()
        
        if not backtest:
            raise HTTPException(status_code=404, detail="Backtest not found")
        
        if not backtest.equity_curve:
            return []
        
        # Generate timestamps for equity curve points
        equity_curve = backtest.equity_curve
        start_time = datetime.combine(backtest.start_date, datetime.min.time())
        end_time = datetime.combine(backtest.end_date, datetime.max.time())
        
        # Create evenly spaced timestamps
        from datetime import timedelta
        duration = end_time - start_time
        interval = duration / len(equity_curve) if len(equity_curve) > 1 else duration
        
        peak = equity_curve[0]
        trades_count = 0
        
        points = []
        for i, equity in enumerate(equity_curve):
            timestamp = start_time + interval * i
            
            # Update peak and calculate drawdown
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak * 100 if peak > 0 else 0
            
            # Estimate trades count (simplified)
            if i > 0 and abs(equity - equity_curve[i-1]) > equity_curve[i-1] * 0.001:
                trades_count += 1
            
            points.append(EquityCurvePoint(
                timestamp=timestamp,
                equity=equity,
                drawdown=drawdown,
                trades_count=trades_count
            ))
        
        return points


@router.delete("/results/{backtest_id}")
async def delete_backtest(
    backtest_id: int,
    current_user: User = Depends(get_current_user)
):
    """Delete a backtest result."""
    async with get_db_session() as session:
        from sqlalchemy import select
        
        result = await session.execute(
            select(BacktestResult).where(
                BacktestResult.id == backtest_id,
                BacktestResult.user_id == current_user.id
            )
        )
        backtest = result.scalar_one_or_none()
        
        if not backtest:
            raise HTTPException(status_code=404, detail="Backtest not found")
        
        await session.delete(backtest)
        await session.commit()
        
        return {"success": True, "message": "Backtest deleted successfully"}
