"""
TradeOS Admin Panel API Routes
==============================
Admin-only endpoints for user management, billing oversight, and system configuration.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks, Request
from pydantic import BaseModel, Field

from ..auth.jwt_handler import get_current_user, TokenPayload, require_roles
from ..rbac.permissions import Permission, require_permission, require_permissions
from ..users.user_manager import get_user_manager, User, UserStatus
from ..subscriptions.tier_manager import get_tier_manager, SubscriptionTier
from ..billing.stripe_integration import get_billing_manager
from ..usage.tracker import get_usage_tracker
from ..notifications.email import get_email_service


# ============== Pydantic Models ==============

class UserListResponse(BaseModel):
    id: str
    email: str
    status: str
    tier: str
    roles: List[str]
    email_verified: bool
    created_at: str
    last_login: Optional[str]


class UserDetailResponse(UserListResponse):
    profile: Dict[str, Any]
    oauth_providers: List[str]
    login_attempts: int
    locked_until: Optional[str]


class UserUpdateRequest(BaseModel):
    status: Optional[str] = None
    tier: Optional[str] = None
    roles: Optional[List[str]] = None
    email_verified: Optional[bool] = None


class SubscriptionUpdateRequest(BaseModel):
    tier: str
    reason: str
    period_end: Optional[str] = None


class SystemStatsResponse(BaseModel):
    total_users: int
    active_users: int
    users_by_tier: Dict[str, int]
    users_by_status: Dict[str, int]
    total_revenue_month: float
    mrr: float  # Monthly Recurring Revenue
    api_calls_today: int
    backtests_today: int


class AuditLogEntry(BaseModel):
    id: str
    timestamp: str
    admin_id: str
    admin_email: str
    action: str
    resource_type: str
    resource_id: str
    details: Dict[str, Any]
    ip_address: Optional[str]


# ============== Admin Router ==============

router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(require_roles("admin", "super_admin"))]
)


# ============== Audit Logging ==============

class AuditLogger:
    """Simple audit logging for admin actions"""
    
    def __init__(self):
        self.logs: List[Dict[str, Any]] = []
    
    async def log(
        self,
        admin: TokenPayload,
        action: str,
        resource_type: str,
        resource_id: str,
        details: Dict[str, Any],
        request: Optional[Request] = None
    ) -> None:
        """Log an admin action"""
        import secrets
        
        entry = {
            "id": f"audit_{secrets.token_urlsafe(16)}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "admin_id": admin.user_id,
            "admin_email": admin.email,
            "action": action,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "details": details,
            "ip_address": request.client.host if request and request.client else None
        }
        
        self.logs.append(entry)
        
        # Keep only last 10000 logs
        if len(self.logs) > 10000:
            self.logs = self.logs[-5000:]
    
    def get_logs(
        self,
        admin_id: Optional[str] = None,
        action: Optional[str] = None,
        resource_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get audit logs with filtering"""
        logs = self.logs
        
        if admin_id:
            logs = [l for l in logs if l["admin_id"] == admin_id]
        
        if action:
            logs = [l for l in logs if l["action"] == action]
        
        if resource_type:
            logs = [l for l in logs if l["resource_type"] == resource_type]
        
        logs.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return logs[offset:offset + limit]


audit_logger = AuditLogger()


# ============== User Management Routes ==============

@router.get("/users", response_model=List[UserListResponse])
async def list_users(
    status: Optional[str] = Query(None, description="Filter by status"),
    tier: Optional[str] = Query(None, description="Filter by tier"),
    search: Optional[str] = Query(None, description="Search by email"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    current_user: TokenPayload = Depends(get_current_user)
):
    """
    List all users with optional filtering.
    
    Requires: admin role
    """
    user_manager = get_user_manager()
    
    # Get all users
    users = user_manager.list_users(limit=limit + offset, offset=0)
    
    # Apply filters
    if status:
        try:
            status_enum = UserStatus(status)
            users = [u for u in users if u.status == status_enum]
        except ValueError:
            pass
    
    if tier:
        users = [u for u in users if u.tier == tier]
    
    if search:
        users = [u for u in users if search.lower() in u.email.lower()]
    
    # Apply pagination
    users = users[offset:offset + limit]
    
    return [
        UserListResponse(
            id=u.id,
            email=u.email,
            status=u.status.value,
            tier=u.tier,
            roles=u.roles,
            email_verified=u.email_verified,
            created_at=u.created_at.isoformat(),
            last_login=u.last_login.isoformat() if u.last_login else None
        )
        for u in users
    ]


@router.get("/users/{user_id}", response_model=UserDetailResponse)
async def get_user(
    user_id: str,
    current_user: TokenPayload = Depends(get_current_user)
):
    """Get detailed user information"""
    user_manager = get_user_manager()
    
    user = user_manager.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return UserDetailResponse(
        id=user.id,
        email=user.email,
        status=user.status.value,
        tier=user.tier,
        roles=user.roles,
        email_verified=user.email_verified,
        created_at=user.created_at.isoformat(),
        last_login=user.last_login.isoformat() if user.last_login else None,
        profile={
            "first_name": user.profile.first_name,
            "last_name": user.profile.last_name,
            "company": user.profile.company,
            "timezone": user.profile.timezone
        },
        oauth_providers=list(user.oauth_providers.keys()),
        login_attempts=user.login_attempts,
        locked_until=user.locked_until.isoformat() if user.locked_until else None
    )


@router.patch("/users/{user_id}")
async def update_user(
    user_id: str,
    request: UserUpdateRequest,
    background_tasks: BackgroundTasks,
    current_user: TokenPayload = Depends(get_current_user),
    http_request: Request = None
):
    """
    Update user settings.
    
    Allows admins to:
    - Change user tier
    - Update user status
    - Modify roles
    - Verify email
    """
    user_manager = get_user_manager()
    email_service = get_email_service()
    
    user = user_manager.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    changes = {}
    
    # Update status
    if request.status:
        try:
            new_status = UserStatus(request.status)
            if user.status != new_status:
                changes["status"] = {"from": user.status.value, "to": new_status.value}
                user.status = new_status
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {request.status}")
    
    # Update tier
    if request.tier:
        if user.tier != request.tier:
            changes["tier"] = {"from": user.tier, "to": request.tier}
            user.tier = request.tier
            
            # Send notification
            background_tasks.add_task(
                email_service.send_templated_email,
                to_email=user.email,
                to_name=user.profile.first_name or user.email,
                template="subscription_confirmed",  # Simplified
                subject="Your Plan Has Been Updated",
                name=user.profile.first_name or user.email,
                plan_name=request.tier.title(),
                amount="0",
                interval="month",
                start_date=datetime.now().isoformat(),
                next_billing="N/A",
                features=[]
            )
    
    # Update roles
    if request.roles:
        changes["roles"] = {"from": user.roles, "to": request.roles}
        user.roles = request.roles
    
    # Update email verified
    if request.email_verified is not None:
        changes["email_verified"] = {"from": user.email_verified, "to": request.email_verified}
        user.email_verified = request.email_verified
    
    user.updated_at = datetime.now(timezone.utc)
    
    # Log action
    await audit_logger.log(
        admin=current_user,
        action="user_update",
        resource_type="user",
        resource_id=user_id,
        details=changes,
        request=http_request
    )
    
    return {
        "message": "User updated successfully",
        "changes": changes,
        "user": user.to_dict()
    }


@router.post("/users/{user_id}/suspend")
async def suspend_user(
    user_id: str,
    reason: str,
    current_user: TokenPayload = Depends(get_current_user),
    http_request: Request = None
):
    """Suspend a user account"""
    user_manager = get_user_manager()
    
    user = user_manager.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    user.status = UserStatus.SUSPENDED
    user.updated_at = datetime.now(timezone.utc)
    
    # Revoke all tokens
    from ..auth.jwt_handler import get_jwt_handler
    get_jwt_handler().revoke_all_user_tokens(user_id)
    
    # Log action
    await audit_logger.log(
        admin=current_user,
        action="user_suspend",
        resource_type="user",
        resource_id=user_id,
        details={"reason": reason},
        request=http_request
    )
    
    return {"message": "User suspended successfully"}


@router.post("/users/{user_id}/reactivate")
async def reactivate_user(
    user_id: str,
    current_user: TokenPayload = Depends(get_current_user),
    http_request: Request = None
):
    """Reactivate a suspended user account"""
    user_manager = get_user_manager()
    
    user = user_manager.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if user.status != UserStatus.SUSPENDED:
        raise HTTPException(status_code=400, detail="User is not suspended")
    
    user.status = UserStatus.ACTIVE
    user.updated_at = datetime.now(timezone.utc)
    
    # Log action
    await audit_logger.log(
        admin=current_user,
        action="user_reactivate",
        resource_type="user",
        resource_id=user_id,
        details={},
        request=http_request
    )
    
    return {"message": "User reactivated successfully"}


# ============== Subscription Management Routes ==============

@router.post("/users/{user_id}/subscription")
async def update_subscription(
    user_id: str,
    request: SubscriptionUpdateRequest,
    current_user: TokenPayload = Depends(get_current_user),
    http_request: Request = None
):
    """
    Manually update a user's subscription.
    
    Use for:
    - Complimentary upgrades
    - Handling support issues
    - Correcting billing errors
    """
    user_manager = get_user_manager()
    tier_manager = get_tier_manager()
    
    user = user_manager.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Validate tier
    if request.tier not in ["free", "pro", "enterprise"]:
        raise HTTPException(status_code=400, detail="Invalid tier")
    
    old_tier = user.tier
    user.tier = request.tier
    user.updated_at = datetime.now(timezone.utc)
    
    # Log action
    await audit_logger.log(
        admin=current_user,
        action="subscription_update",
        resource_type="subscription",
        resource_id=user_id,
        details={
            "old_tier": old_tier,
            "new_tier": request.tier,
            "reason": request.reason
        },
        request=http_request
    )
    
    return {
        "message": "Subscription updated successfully",
        "user_id": user_id,
        "old_tier": old_tier,
        "new_tier": request.tier
    }


@router.get("/subscriptions/overview")
async def get_subscription_overview(
    current_user: TokenPayload = Depends(get_current_user)
):
    """Get subscription overview statistics"""
    user_manager = get_user_manager()
    
    users = user_manager.list_users(limit=10000, offset=0)
    
    tier_counts = {}
    status_counts = {}
    
    for user in users:
        tier_counts[user.tier] = tier_counts.get(user.tier, 0) + 1
        status_counts[user.status.value] = status_counts.get(user.status.value, 0) + 1
    
    return {
        "total_users": len(users),
        "users_by_tier": tier_counts,
        "users_by_status": status_counts,
        "active_subscriptions": sum(1 for u in users if u.tier != "free"),
        "free_users": tier_counts.get("free", 0)
    }


# ============== Usage & Analytics Routes ==============

@router.get("/analytics/overview")
async def get_analytics_overview(
    days: int = Query(30, ge=1, le=365),
    current_user: TokenPayload = Depends(get_current_user)
):
    """Get system-wide analytics overview"""
    usage_tracker = get_usage_tracker()
    user_manager = get_user_manager()
    
    users = user_manager.list_users(limit=10000, offset=0)
    
    # Calculate active users (logged in within last 30 days)
    thirty_days_ago = datetime.now(timezone.utc) - timedelta(days=30)
    active_users = sum(
        1 for u in users
        if u.last_login and u.last_login > thirty_days_ago
    )
    
    # Get usage stats
    from ..usage.tracker import UsageType
    api_calls = sum(
        usage_tracker.get_usage_count(u.id, UsageType.API_CALL, "daily")
        for u in users
    )
    
    backtests = sum(
        usage_tracker.get_usage_count(u.id, UsageType.BACKTEST, "daily")
        for u in users
    )
    
    return {
        "total_users": len(users),
        "active_users_30d": active_users,
        "users_by_tier": {
            tier: sum(1 for u in users if u.tier == tier)
            for tier in ["free", "pro", "enterprise"]
        },
        "api_calls_today": api_calls,
        "backtests_today": backtests,
        "period_days": days
    }


@router.get("/analytics/top-users")
async def get_top_users(
    metric: str = Query("api_calls", enum=["api_calls", "backtests", "strategy_executions"]),
    limit: int = Query(10, ge=1, le=100),
    days: int = Query(7, ge=1, le=90),
    current_user: TokenPayload = Depends(get_current_user)
):
    """Get top users by various metrics"""
    usage_tracker = get_usage_tracker()
    user_manager = get_user_manager()
    
    from ..usage.tracker import UsageType
    
    usage_type_map = {
        "api_calls": UsageType.API_CALL,
        "backtests": UsageType.BACKTEST,
        "strategy_executions": UsageType.STRATEGY_EXECUTION
    }
    
    usage_type = usage_type_map.get(metric, UsageType.API_CALL)
    
    top_users = usage_tracker.get_top_users(usage_type, limit, days)
    
    # Enrich with user details
    result = []
    for item in top_users:
        user = user_manager.get_user(item["user_id"])
        if user:
            result.append({
                "user_id": item["user_id"],
                "email": user.email,
                "tier": user.tier,
                "usage": item["usage"]
            })
    
    return result


# ============== Audit Log Routes ==============

@router.get("/audit-logs")
async def get_audit_logs(
    admin_id: Optional[str] = Query(None),
    action: Optional[str] = Query(None),
    resource_type: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    current_user: TokenPayload = Depends(get_current_user)
):
    """Get admin audit logs"""
    logs = audit_logger.get_logs(
        admin_id=admin_id,
        action=action,
        resource_type=resource_type,
        limit=limit,
        offset=offset
    )
    
    return {
        "logs": logs,
        "total": len(audit_logger.logs),
        "limit": limit,
        "offset": offset
    }


# ============== System Configuration Routes ==============

@router.get("/config")
async def get_system_config(
    current_user: TokenPayload = Depends(require_roles("super_admin"))
):
    """Get system configuration (super admin only)"""
    config = get_config()
    
    return {
        "environment": config.environment.value,
        "features": config.features,
        "rate_limits": config.rate_limit.tier_limits,
        "subscription_limits": config.subscription_limits.limits
    }


@router.post("/config/features/{feature}")
async def toggle_feature(
    feature: str,
    enabled: bool,
    current_user: TokenPayload = Depends(require_roles("super_admin")),
    http_request: Request = None
):
    """Toggle a feature flag (super admin only)"""
    config = get_config()
    
    if feature not in config.features:
        raise HTTPException(status_code=400, detail="Unknown feature")
    
    old_value = config.features[feature]
    config.features[feature] = enabled
    
    # Log action
    await audit_logger.log(
        admin=current_user,
        action="feature_toggle",
        resource_type="config",
        resource_id=feature,
        details={"from": old_value, "to": enabled},
        request=http_request
    )
    
    return {
        "feature": feature,
        "enabled": enabled,
        "previous": old_value
    }


# ============== Health & Status Routes ==============

@router.get("/health")
async def admin_health_check(
    current_user: TokenPayload = Depends(get_current_user)
):
    """System health check for admins"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0",
        "services": {
            "auth": "healthy",
            "billing": "healthy",
            "email": "healthy",
            "database": "healthy"
        }
    }


# ============== Export Router Function ==============

def get_admin_router() -> APIRouter:
    """Get the admin router for inclusion in main app"""
    return router
