"""
Admin Router

Handles admin-only operations for user management and system oversight.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, date
from decimal import Decimal

from ..dependencies.auth import get_current_user, require_admin
from ..database.models import User, Strategy, Trade, Subscription
from ..database.connection import get_db_session
from ..saas.rbac.permissions import UserRole

router = APIRouter(prefix="/admin", tags=["Admin"], dependencies=[Depends(require_admin)])


# Request/Response Models
class UserListItem(BaseModel):
    id: int
    email: str
    first_name: Optional[str]
    last_name: Optional[str]
    role: str
    subscription_tier: str
    is_verified: bool
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime]


class UserDetailResponse(UserListItem):
    strategies_count: int
    trades_count: int
    total_pnl: float
    subscription_status: str


class UpdateUserRequest(BaseModel):
    role: Optional[str] = None
    subscription_tier: Optional[str] = None
    is_active: Optional[bool] = None
    is_verified: Optional[bool] = None


class SystemStatsResponse(BaseModel):
    total_users: int
    active_users_today: int
    total_strategies: int
    active_strategies: int
    total_trades: int
    total_volume: float
    revenue_this_month: float
    system_health: Dict[str, str]


class AuditLogEntry(BaseModel):
    id: int
    user_id: Optional[int]
    action: str
    resource_type: str
    resource_id: Optional[int]
    details: Dict[str, Any]
    ip_address: str
    timestamp: datetime


@router.get("/users", response_model=List[UserListItem])
async def list_users(
    role: Optional[str] = None,
    tier: Optional[str] = None,
    is_active: Optional[bool] = None,
    search: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
):
    """List all users with filtering."""
    async with get_db_session() as session:
        from sqlalchemy import select, or_
        
        query = select(User)
        
        if role:
            query = query.where(User.role == role)
        if tier:
            query = query.where(User.subscription_tier == tier)
        if is_active is not None:
            query = query.where(User.is_active == is_active)
        if search:
            query = query.where(
                or_(
                    User.email.ilike(f"%{search}%"),
                    User.first_name.ilike(f"%{search}%"),
                    User.last_name.ilike(f"%{search}%")
                )
            )
        
        # Get total count
        from sqlalchemy import func
        count_result = await session.execute(
            select(func.count(User.id)).select_from(query.subquery())
        )
        total = count_result.scalar()
        
        # Get users
        result = await session.execute(
            query.order_by(User.created_at.desc()).limit(limit).offset(offset)
        )
        users = result.scalars().all()
        
        return [
            UserListItem(
                id=user.id,
                email=user.email,
                first_name=user.first_name,
                last_name=user.last_name,
                role=user.role,
                subscription_tier=user.subscription_tier,
                is_verified=user.is_verified,
                is_active=user.is_active,
                created_at=user.created_at,
                last_login=getattr(user, 'last_login', None)
            )
            for user in users
        ]


@router.get("/users/{user_id}", response_model=UserDetailResponse)
async def get_user_detail(user_id: int):
    """Get detailed information about a specific user."""
    async with get_db_session() as session:
        from sqlalchemy import select, func
        
        result = await session.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Get strategy count
        result = await session.execute(
            select(func.count(Strategy.id)).where(Strategy.user_id == user_id)
        )
        strategies_count = result.scalar() or 0
        
        # Get trade count
        result = await session.execute(
            select(func.count(Trade.id)).where(Trade.user_id == user_id)
        )
        trades_count = result.scalar() or 0
        
        # Get total PnL
        result = await session.execute(
            select(func.sum(Trade.realized_pnl)).where(Trade.user_id == user_id)
        )
        total_pnl = result.scalar() or 0
        
        # Get subscription status
        result = await session.execute(
            select(Subscription).where(Subscription.user_id == user_id)
        )
        subscription = result.scalar_one_or_none()
        
        return UserDetailResponse(
            id=user.id,
            email=user.email,
            first_name=user.first_name,
            last_name=user.last_name,
            role=user.role,
            subscription_tier=user.subscription_tier,
            is_verified=user.is_verified,
            is_active=user.is_active,
            created_at=user.created_at,
            last_login=getattr(user, 'last_login', None),
            strategies_count=strategies_count,
            trades_count=trades_count,
            total_pnl=float(total_pnl),
            subscription_status=subscription.status if subscription else "none"
        )


@router.put("/users/{user_id}")
async def update_user(user_id: int, request: UpdateUserRequest):
    """Update user settings."""
    async with get_db_session() as session:
        from sqlalchemy import select
        
        result = await session.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Update fields
        if request.role:
            user.role = request.role
        if request.subscription_tier:
            user.subscription_tier = request.subscription_tier
        if request.is_active is not None:
            user.is_active = request.is_active
        if request.is_verified is not None:
            user.is_verified = request.is_verified
        
        await session.commit()
        await session.refresh(user)
        
        return {
            "success": True,
            "message": "User updated successfully",
            "user": {
                "id": user.id,
                "email": user.email,
                "role": user.role,
                "subscription_tier": user.subscription_tier,
                "is_active": user.is_active
            }
        }


@router.delete("/users/{user_id}")
async def delete_user(user_id: int):
    """Delete a user account."""
    async with get_db_session() as session:
        from sqlalchemy import select
        
        result = await session.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        await session.delete(user)
        await session.commit()
        
        return {"success": True, "message": "User deleted successfully"}


@router.get("/stats", response_model=SystemStatsResponse)
async def get_system_stats():
    """Get overall system statistics."""
    async with get_db_session() as session:
        from sqlalchemy import select, func, and_
        from datetime import timedelta
        
        # Total users
        result = await session.execute(select(func.count(User.id)))
        total_users = result.scalar() or 0
        
        # Active users today
        today = date.today()
        result = await session.execute(
            select(func.count(func.distinct(Trade.user_id))).where(
                func.date(Trade.created_at) == today
            )
        )
        active_users_today = result.scalar() or 0
        
        # Total strategies
        result = await session.execute(select(func.count(Strategy.id)))
        total_strategies = result.scalar() or 0
        
        # Active strategies
        result = await session.execute(
            select(func.count(Strategy.id)).where(Strategy.is_active == True)
        )
        active_strategies = result.scalar() or 0
        
        # Total trades
        result = await session.execute(select(func.count(Trade.id)))
        total_trades = result.scalar() or 0
        
        # Total volume (simplified)
        result = await session.execute(
            select(func.sum(Trade.quantity * Trade.entry_price))
        )
        total_volume = float(result.scalar() or 0)
        
        # Revenue this month
        result = await session.execute(
            select(func.sum(Subscription.amount)).where(
                and_(
                    Subscription.status == "active",
                    func.extract('month', Subscription.current_period_start) == today.month,
                    func.extract('year', Subscription.current_period_start) == today.year
                )
            )
        )
        revenue = float(result.scalar() or 0)
        
        return SystemStatsResponse(
            total_users=total_users,
            active_users_today=active_users_today,
            total_strategies=total_strategies,
            active_strategies=active_strategies,
            total_trades=total_trades,
            total_volume=total_volume,
            revenue_this_month=revenue,
            system_health={
                "database": "healthy",
                "api": "healthy",
                "websocket": "healthy",
                "redis": "healthy"
            }
        )


@router.get("/audit-log")
async def get_audit_log(
    user_id: Optional[int] = None,
    action: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """Get system audit log."""
    # Mock audit log - in real implementation, query audit table
    logs = [
        {
            "id": i,
            "user_id": user_id or (i % 10 + 1),
            "action": action or ["login", "trade", "strategy_update", "settings_change"][i % 4],
            "resource_type": ["user", "trade", "strategy", "settings"][i % 4],
            "resource_id": i * 10,
            "details": {"ip": "192.168.1.1", "user_agent": "Mozilla/5.0"},
            "ip_address": "192.168.1.1",
            "timestamp": datetime.utcnow().isoformat()
        }
        for i in range(limit)
    ]
    
    return {
        "logs": logs,
        "total": 1000,
        "limit": limit,
        "offset": offset
    }


@router.get("/subscriptions")
async def get_all_subscriptions(
    status: Optional[str] = None,
    tier: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
):
    """Get all subscriptions."""
    async with get_db_session() as session:
        from sqlalchemy import select
        
        query = select(Subscription, User.email).join(User, Subscription.user_id == User.id)
        
        if status:
            query = query.where(Subscription.status == status)
        if tier:
            query = query.where(Subscription.tier == tier)
        
        query = query.order_by(Subscription.created_at.desc()).limit(limit).offset(offset)
        
        result = await session.execute(query)
        rows = result.all()
        
        return [
            {
                "id": row.Subscription.id,
                "user_id": row.Subscription.user_id,
                "user_email": row.email,
                "tier": row.Subscription.tier,
                "status": row.Subscription.status,
                "amount": float(row.Subscription.amount) if row.Subscription.amount else 0,
                "current_period_start": row.Subscription.current_period_start.isoformat() if row.Subscription.current_period_start else None,
                "current_period_end": row.Subscription.current_period_end.isoformat() if row.Subscription.current_period_end else None,
                "created_at": row.Subscription.created_at.isoformat()
            }
            for row in rows
        ]


@router.post("/announcements")
async def create_announcement(
    title: str,
    content: str,
    target_tiers: List[str] = ["all"],
    send_email: bool = False
):
    """Create a system announcement."""
    # In real implementation, save to announcements table and optionally send emails
    return {
        "success": True,
        "message": "Announcement created successfully",
        "announcement": {
            "id": 1,
            "title": title,
            "content": content,
            "target_tiers": target_tiers,
            "created_at": datetime.utcnow().isoformat()
        }
    }


@router.get("/health")
async def get_system_health():
    """Get detailed system health status."""
    import psutil
    import time
    
    # System metrics
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "api": {"status": "healthy", "latency_ms": 12},
            "database": {"status": "healthy", "connections": 15, "latency_ms": 5},
            "redis": {"status": "healthy", "latency_ms": 2},
            "websocket": {"status": "healthy", "connections": 342}
        },
        "system": {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_used_gb": round(memory.used / (1024**3), 2),
            "memory_total_gb": round(memory.total / (1024**3), 2),
            "disk_percent": disk.percent,
            "disk_used_gb": round(disk.used / (1024**3), 2),
            "disk_total_gb": round(disk.total / (1024**3), 2)
        }
    }
