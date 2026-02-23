"""
User Router

Handles user profile management, settings, and API keys.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List
from datetime import datetime

from ..dependencies.auth import get_current_user
from ..database.models import User, ApiKey
from ..database.connection import get_db_session
from ..saas.api_keys.manager import ApiKeyManager

router = APIRouter(prefix="/user", tags=["User"])


# Request/Response Models
class UpdateProfileRequest(BaseModel):
    first_name: Optional[str] = Field(None, max_length=100)
    last_name: Optional[str] = Field(None, max_length=100)
    phone: Optional[str] = Field(None, max_length=20)


class ProfileResponse(BaseModel):
    id: int
    email: str
    first_name: Optional[str]
    last_name: Optional[str]
    phone: Optional[str]
    role: str
    subscription_tier: str
    is_verified: bool
    created_at: datetime
    last_login: Optional[datetime]


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=8)


class ApiKeyCreateRequest(BaseModel):
    name: str = Field(..., max_length=100)
    permissions: List[str] = Field(default=["read"])
    ip_whitelist: Optional[List[str]] = None
    expires_in_days: Optional[int] = 90


class ApiKeyResponse(BaseModel):
    id: int
    name: str
    key_preview: str
    permissions: List[str]
    ip_whitelist: Optional[List[str]]
    last_used_at: Optional[datetime]
    expires_at: Optional[datetime]
    is_active: bool
    created_at: datetime


class ApiKeyCreateResponse(BaseModel):
    id: int
    name: str
    api_key: str  # Only shown once on creation
    permissions: List[str]
    expires_at: Optional[datetime]
    created_at: datetime


# Initialize API key manager
api_key_manager = ApiKeyManager()


@router.get("/profile", response_model=ProfileResponse)
async def get_profile(current_user: User = Depends(get_current_user)):
    """Get current user profile."""
    return ProfileResponse(
        id=current_user.id,
        email=current_user.email,
        first_name=current_user.first_name,
        last_name=current_user.last_name,
        phone=getattr(current_user, 'phone', None),
        role=current_user.role,
        subscription_tier=current_user.subscription_tier,
        is_verified=current_user.is_verified,
        created_at=current_user.created_at,
        last_login=getattr(current_user, 'last_login', None)
    )


@router.put("/profile")
async def update_profile(
    request: UpdateProfileRequest,
    current_user: User = Depends(get_current_user)
):
    """Update user profile information."""
    async with get_db_session() as session:
        # Update fields
        if request.first_name is not None:
            current_user.first_name = request.first_name
        if request.last_name is not None:
            current_user.last_name = request.last_name
        if request.phone is not None:
            current_user.phone = request.phone
        
        await session.commit()
        await session.refresh(current_user)
        
        return ProfileResponse(
            id=current_user.id,
            email=current_user.email,
            first_name=current_user.first_name,
            last_name=current_user.last_name,
            phone=getattr(current_user, 'phone', None),
            role=current_user.role,
            subscription_tier=current_user.subscription_tier,
            is_verified=current_user.is_verified,
            created_at=current_user.created_at,
            last_login=getattr(current_user, 'last_login', None)
        )


@router.post("/change-password")
async def change_password(
    request: ChangePasswordRequest,
    current_user: User = Depends(get_current_user)
):
    """Change user password."""
    from ..saas.users.user_manager import UserManager
    
    user_manager = UserManager()
    
    # Verify current password
    is_valid = await user_manager.verify_password(
        current_user,
        request.current_password
    )
    
    if not is_valid:
        raise HTTPException(status_code=400, detail="Current password is incorrect")
    
    # Update password
    success = await user_manager.update_password(
        current_user.id,
        request.new_password
    )
    
    if success:
        return {"success": True, "message": "Password changed successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to change password")


@router.get("/api-keys", response_model=List[ApiKeyResponse])
async def get_api_keys(current_user: User = Depends(get_current_user)):
    """Get all API keys for the user."""
    async with get_db_session() as session:
        from sqlalchemy import select
        
        result = await session.execute(
            select(ApiKey).where(ApiKey.user_id == current_user.id)
            .order_by(ApiKey.created_at.desc())
        )
        keys = result.scalars().all()
        
        return [
            ApiKeyResponse(
                id=key.id,
                name=key.name,
                key_preview=f"{key.key_prefix}..." if key.key_prefix else "****",
                permissions=key.permissions,
                ip_whitelist=key.ip_whitelist,
                last_used_at=key.last_used_at,
                expires_at=key.expires_at,
                is_active=key.is_active,
                created_at=key.created_at
            )
            for key in keys
        ]


@router.post("/api-keys", response_model=ApiKeyCreateResponse)
async def create_api_key(
    request: ApiKeyCreateRequest,
    current_user: User = Depends(get_current_user)
):
    """Create a new API key."""
    api_key, key_string = await api_key_manager.create_api_key(
        user_id=current_user.id,
        name=request.name,
        permissions=request.permissions,
        ip_whitelist=request.ip_whitelist,
        expires_in_days=request.expires_in_days
    )
    
    return ApiKeyCreateResponse(
        id=api_key.id,
        name=api_key.name,
        api_key=key_string,  # Only shown once!
        permissions=api_key.permissions,
        expires_at=api_key.expires_at,
        created_at=api_key.created_at
    )


@router.delete("/api-keys/{key_id}")
async def revoke_api_key(
    key_id: int,
    current_user: User = Depends(get_current_user)
):
    """Revoke an API key."""
    success = await api_key_manager.revoke_api_key(
        key_id=key_id,
        user_id=current_user.id
    )
    
    if success:
        return {"success": True, "message": "API key revoked successfully"}
    else:
        raise HTTPException(status_code=404, detail="API key not found")


@router.get("/notifications/settings")
async def get_notification_settings(current_user: User = Depends(get_current_user)):
    """Get user notification settings."""
    # Default settings
    return {
        "email_notifications": True,
        "trade_executions": True,
        "trade_closures": True,
        "risk_alerts": True,
        "daily_summary": False,
        "weekly_summary": True,
        "price_alerts": True,
        "marketing_emails": False
    }


@router.put("/notifications/settings")
async def update_notification_settings(
    settings: dict,
    current_user: User = Depends(get_current_user)
):
    """Update user notification settings."""
    # In a real implementation, save to database
    return {
        "success": True,
        "message": "Notification settings updated",
        "settings": settings
    }


@router.get("/activity")
async def get_user_activity(
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(get_current_user)
):
    """Get user activity log."""
    # Mock activity log - in real implementation, query activity table
    activities = [
        {
            "id": i,
            "action": action,
            "details": details,
            "ip_address": "192.168.1.1",
            "timestamp": datetime.utcnow().isoformat()
        }
        for i, (action, details) in enumerate([
            ("login", "Successful login from Chrome on Windows"),
            ("strategy_created", "Created strategy 'EMA Crossover'"),
            ("trade_executed", "Bought 0.5 BTC at $65,000"),
            ("backtest_run", "Ran backtest for strategy #123"),
            ("api_key_created", "Created API key 'Trading Bot'"),
            ("password_changed", "Password changed successfully"),
            ("settings_updated", "Updated notification settings"),
        ])
    ]
    
    return {
        "activities": activities[:limit],
        "total": len(activities),
        "limit": limit,
        "offset": offset
    }


@router.get("/stats")
async def get_user_stats(current_user: User = Depends(get_current_user)):
    """Get user trading statistics."""
    async with get_db_session() as session:
        from sqlalchemy import select, func
        from ..database.models import Trade, Strategy, BacktestResult
        
        # Count strategies
        result = await session.execute(
            select(func.count(Strategy.id)).where(Strategy.user_id == current_user.id)
        )
        strategy_count = result.scalar() or 0
        
        # Count active strategies
        result = await session.execute(
            select(func.count(Strategy.id)).where(
                Strategy.user_id == current_user.id,
                Strategy.is_active == True
            )
        )
        active_strategies = result.scalar() or 0
        
        # Count trades
        result = await session.execute(
            select(func.count(Trade.id)).where(Trade.user_id == current_user.id)
        )
        trade_count = result.scalar() or 0
        
        # Count backtests
        result = await session.execute(
            select(func.count(BacktestResult.id)).where(BacktestResult.user_id == current_user.id)
        )
        backtest_count = result.scalar() or 0
        
        # Total PnL
        result = await session.execute(
            select(func.sum(Trade.realized_pnl)).where(Trade.user_id == current_user.id)
        )
        total_pnl = result.scalar() or 0
        
        return {
            "strategies": {
                "total": strategy_count,
                "active": active_strategies
            },
            "trades": {
                "total": trade_count
            },
            "backtests": {
                "total": backtest_count
            },
            "performance": {
                "total_pnl": float(total_pnl)
            },
            "account": {
                "tier": current_user.subscription_tier,
                "member_since": current_user.created_at.isoformat()
            }
        }
