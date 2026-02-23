"""
TradeOS Role-Based Access Control (RBAC)
=========================================
Permission system with roles, resources, and decorators.
"""

from enum import Enum, auto
from functools import wraps
from typing import Dict, List, Optional, Set, Callable, Any
from dataclasses import dataclass, field
import inspect

from fastapi import HTTPException, Depends

from ..auth.jwt_handler import TokenPayload, get_current_user


class Permission(Enum):
    """System-wide permissions"""
    # User management
    USER_READ = "user:read"
    USER_UPDATE = "user:update"
    USER_DELETE = "user:delete"
    
    # Strategy permissions
    STRATEGY_CREATE = "strategy:create"
    STRATEGY_READ = "strategy:read"
    STRATEGY_UPDATE = "strategy:update"
    STRATEGY_DELETE = "strategy:delete"
    STRATEGY_EXECUTE = "strategy:execute"
    STRATEGY_SHARE = "strategy:share"
    
    # Backtest permissions
    BACKTEST_CREATE = "backtest:create"
    BACKTEST_READ = "backtest:read"
    BACKTEST_DELETE = "backtest:delete"
    
    # Trading permissions
    TRADE_EXECUTE = "trade:execute"
    TRADE_READ = "trade:read"
    TRADE_CANCEL = "trade:cancel"
    
    # Portfolio permissions
    PORTFOLIO_READ = "portfolio:read"
    PORTFOLIO_MANAGE = "portfolio:manage"
    
    # API key permissions
    API_KEY_CREATE = "api_key:create"
    API_KEY_READ = "api_key:read"
    API_KEY_DELETE = "api_key:delete"
    
    # Webhook permissions
    WEBHOOK_CREATE = "webhook:create"
    WEBHOOK_READ = "webhook:read"
    WEBHOOK_DELETE = "webhook:delete"
    
    # Billing permissions
    BILLING_READ = "billing:read"
    BILLING_MANAGE = "billing:manage"
    
    # Admin permissions
    ADMIN_USERS = "admin:users"
    ADMIN_STRATEGIES = "admin:strategies"
    ADMIN_BILLING = "admin:billing"
    ADMIN_SYSTEM = "admin:system"
    ADMIN_SETTINGS = "admin:settings"


class Role(Enum):
    """User roles with associated permissions"""
    
    # Regular user roles
    VIEWER = "viewer"           # Can view strategies and results
    TRADER = "trader"           # Can create and execute strategies
    ANALYST = "analyst"         # Can create backtests and analyze
    
    # Team roles
    TEAM_MEMBER = "team_member"
    TEAM_ADMIN = "team_admin"
    
    # Admin roles
    SUPPORT = "support"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"


# Role to permissions mapping
ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
    Role.VIEWER: {
        Permission.USER_READ,
        Permission.STRATEGY_READ,
        Permission.BACKTEST_READ,
        Permission.TRADE_READ,
        Permission.PORTFOLIO_READ,
    },
    
    Role.TRADER: {
        Permission.USER_READ,
        Permission.USER_UPDATE,
        Permission.STRATEGY_CREATE,
        Permission.STRATEGY_READ,
        Permission.STRATEGY_UPDATE,
        Permission.STRATEGY_DELETE,
        Permission.STRATEGY_EXECUTE,
        Permission.BACKTEST_CREATE,
        Permission.BACKTEST_READ,
        Permission.BACKTEST_DELETE,
        Permission.TRADE_EXECUTE,
        Permission.TRADE_READ,
        Permission.TRADE_CANCEL,
        Permission.PORTFOLIO_READ,
        Permission.PORTFOLIO_MANAGE,
        Permission.API_KEY_CREATE,
        Permission.API_KEY_READ,
        Permission.API_KEY_DELETE,
        Permission.WEBHOOK_CREATE,
        Permission.WEBHOOK_READ,
        Permission.WEBHOOK_DELETE,
        Permission.BILLING_READ,
        Permission.BILLING_MANAGE,
    },
    
    Role.ANALYST: {
        Permission.USER_READ,
        Permission.STRATEGY_READ,
        Permission.STRATEGY_CREATE,
        Permission.STRATEGY_UPDATE,
        Permission.BACKTEST_CREATE,
        Permission.BACKTEST_READ,
        Permission.BACKTEST_DELETE,
        Permission.PORTFOLIO_READ,
        Permission.TRADE_READ,
    },
    
    Role.TEAM_MEMBER: {
        Permission.USER_READ,
        Permission.STRATEGY_READ,
        Permission.STRATEGY_CREATE,
        Permission.STRATEGY_EXECUTE,
        Permission.BACKTEST_CREATE,
        Permission.BACKTEST_READ,
        Permission.TRADE_READ,
        Permission.PORTFOLIO_READ,
    },
    
    Role.TEAM_ADMIN: {
        # Inherits TEAM_MEMBER + additional permissions
        Permission.USER_READ,
        Permission.USER_UPDATE,
        Permission.STRATEGY_CREATE,
        Permission.STRATEGY_READ,
        Permission.STRATEGY_UPDATE,
        Permission.STRATEGY_DELETE,
        Permission.STRATEGY_SHARE,
        Permission.STRATEGY_EXECUTE,
        Permission.BACKTEST_CREATE,
        Permission.BACKTEST_READ,
        Permission.BACKTEST_DELETE,
        Permission.TRADE_EXECUTE,
        Permission.TRADE_READ,
        Permission.TRADE_CANCEL,
        Permission.PORTFOLIO_READ,
        Permission.PORTFOLIO_MANAGE,
        Permission.API_KEY_CREATE,
        Permission.API_KEY_READ,
        Permission.API_KEY_DELETE,
        Permission.WEBHOOK_CREATE,
        Permission.WEBHOOK_READ,
        Permission.WEBHOOK_DELETE,
        Permission.BILLING_READ,
        Permission.BILLING_MANAGE,
    },
    
    Role.SUPPORT: {
        Permission.USER_READ,
        Permission.STRATEGY_READ,
        Permission.BACKTEST_READ,
        Permission.TRADE_READ,
        Permission.PORTFOLIO_READ,
        Permission.ADMIN_USERS,
    },
    
    Role.ADMIN: {
        Permission.USER_READ,
        Permission.USER_UPDATE,
        Permission.USER_DELETE,
        Permission.STRATEGY_CREATE,
        Permission.STRATEGY_READ,
        Permission.STRATEGY_UPDATE,
        Permission.STRATEGY_DELETE,
        Permission.BACKTEST_CREATE,
        Permission.BACKTEST_READ,
        Permission.BACKTEST_DELETE,
        Permission.TRADE_READ,
        Permission.PORTFOLIO_READ,
        Permission.API_KEY_READ,
        Permission.API_KEY_DELETE,
        Permission.WEBHOOK_READ,
        Permission.WEBHOOK_DELETE,
        Permission.BILLING_READ,
        Permission.BILLING_MANAGE,
        Permission.ADMIN_USERS,
        Permission.ADMIN_STRATEGIES,
        Permission.ADMIN_BILLING,
        Permission.ADMIN_SETTINGS,
    },
    
    Role.SUPER_ADMIN: set(Permission),  # All permissions
}


# Tier-based permission overrides
TIER_PERMISSIONS: Dict[str, Set[Permission]] = {
    "free": {
        Permission.STRATEGY_CREATE,
        Permission.STRATEGY_READ,
        Permission.STRATEGY_UPDATE,
        Permission.STRATEGY_DELETE,
        Permission.STRATEGY_EXECUTE,
        Permission.BACKTEST_CREATE,
        Permission.BACKTEST_READ,
        Permission.BACKTEST_DELETE,
        Permission.TRADE_READ,
        Permission.PORTFOLIO_READ,
    },
    "pro": {
        Permission.STRATEGY_CREATE,
        Permission.STRATEGY_READ,
        Permission.STRATEGY_UPDATE,
        Permission.STRATEGY_DELETE,
        Permission.STRATEGY_EXECUTE,
        Permission.STRATEGY_SHARE,
        Permission.BACKTEST_CREATE,
        Permission.BACKTEST_READ,
        Permission.BACKTEST_DELETE,
        Permission.TRADE_EXECUTE,
        Permission.TRADE_READ,
        Permission.TRADE_CANCEL,
        Permission.PORTFOLIO_READ,
        Permission.PORTFOLIO_MANAGE,
        Permission.API_KEY_CREATE,
        Permission.API_KEY_READ,
        Permission.API_KEY_DELETE,
        Permission.WEBHOOK_CREATE,
        Permission.WEBHOOK_READ,
        Permission.WEBHOOK_DELETE,
        Permission.BILLING_READ,
        Permission.BILLING_MANAGE,
    },
    "enterprise": set(Permission),  # All permissions
}


@dataclass
class ResourcePermission:
    """Permission check for a specific resource"""
    permission: Permission
    resource_type: str  # e.g., "strategy", "portfolio"
    resource_id_field: str  # Field name in the request/path
    owner_field: Optional[str] = None  # Field to check ownership


class PermissionChecker:
    """
    Permission checking system for TradeOS.
    
    Supports:
    - Role-based permissions
    - Tier-based restrictions
    - Resource-level permissions (ownership)
    """
    
    def __init__(self):
        self.role_permissions = ROLE_PERMISSIONS
        self.tier_permissions = TIER_PERMISSIONS
    
    def get_role_permissions(self, role: Role) -> Set[Permission]:
        """Get all permissions for a role"""
        return self.role_permissions.get(role, set())
    
    def get_user_permissions(
        self,
        roles: List[str],
        tier: str
    ) -> Set[Permission]:
        """
        Get effective permissions for a user.
        
        Combines role permissions with tier restrictions.
        """
        # Get all role permissions
        user_perms: Set[Permission] = set()
        for role_str in roles:
            try:
                role = Role(role_str)
                user_perms.update(self.get_role_permissions(role))
            except ValueError:
                continue
        
        # Apply tier restrictions (intersection)
        tier_perms = self.tier_permissions.get(tier, set())
        
        return user_perms & tier_perms
    
    def has_permission(
        self,
        user: TokenPayload,
        permission: Permission
    ) -> bool:
        """Check if user has a specific permission"""
        user_perms = self.get_user_permissions(user.roles, user.tier)
        return permission in user_perms
    
    def has_any_permission(
        self,
        user: TokenPayload,
        permissions: List[Permission]
    ) -> bool:
        """Check if user has any of the specified permissions"""
        user_perms = self.get_user_permissions(user.roles, user.tier)
        return any(p in user_perms for p in permissions)
    
    def has_all_permissions(
        self,
        user: TokenPayload,
        permissions: List[Permission]
    ) -> bool:
        """Check if user has all specified permissions"""
        user_perms = self.get_user_permissions(user.roles, user.tier)
        return all(p in user_perms for p in permissions)
    
    def check_permission(
        self,
        user: TokenPayload,
        permission: Permission,
        raise_exception: bool = True
    ) -> bool:
        """
        Check permission and optionally raise exception.
        
        Args:
            user: Current user
            permission: Required permission
            raise_exception: Whether to raise HTTPException on failure
            
        Returns:
            True if has permission
            
        Raises:
            HTTPException: If no permission and raise_exception=True
        """
        if self.has_permission(user, permission):
            return True
        
        if raise_exception:
            raise HTTPException(
                status_code=403,
                detail=f"Permission denied: {permission.value}"
            )
        return False
    
    def is_owner(
        self,
        user_id: str,
        resource_owner_id: str
    ) -> bool:
        """Check if user is the resource owner"""
        return user_id == resource_owner_id
    
    def is_admin(self, user: TokenPayload) -> bool:
        """Check if user has admin role"""
        return Role.ADMIN.value in user.roles or Role.SUPER_ADMIN.value in user.roles


# Global permission checker instance
_permission_checker: Optional[PermissionChecker] = None


def get_permission_checker() -> PermissionChecker:
    """Get or create global permission checker"""
    global _permission_checker
    if _permission_checker is None:
        _permission_checker = PermissionChecker()
    return _permission_checker


# FastAPI Dependencies

def require_permission(permission: Permission):
    """
    FastAPI dependency to require a specific permission.
    
    Usage:
        @app.post("/strategies")
        async def create_strategy(
            user: TokenPayload = Depends(require_permission(Permission.STRATEGY_CREATE))
        ):
            pass
    """
    async def checker(user: TokenPayload = Depends(get_current_user)) -> TokenPayload:
        checker = get_permission_checker()
        checker.check_permission(user, permission)
        return user
    return checker


def require_permissions(*permissions: Permission, require_all: bool = False):
    """
    FastAPI dependency to require multiple permissions.
    
    Args:
        *permissions: Required permissions
        require_all: If True, require all permissions. If False, require any.
        
    Usage:
        @app.delete("/strategies/{id}")
        async def delete_strategy(
            user: TokenPayload = Depends(require_permissions(
                Permission.STRATEGY_DELETE,
                Permission.ADMIN_STRATEGIES,
                require_all=False
            ))
        ):
            pass
    """
    async def checker(user: TokenPayload = Depends(get_current_user)) -> TokenPayload:
        perm_checker = get_permission_checker()
        
        if require_all:
            has_perm = perm_checker.has_all_permissions(user, list(permissions))
        else:
            has_perm = perm_checker.has_any_permission(user, list(permissions))
        
        if not has_perm:
            perm_names = ", ".join(p.value for p in permissions)
            raise HTTPException(
                status_code=403,
                detail=f"Permission denied. Required: {perm_names}"
            )
        
        return user
    return checker


def require_ownership_or_permission(
    permission: Permission,
    resource_owner_id_getter: Callable
):
    """
    Require either ownership of a resource or a specific permission.
    
    Usage:
        @app.put("/strategies/{strategy_id}")
        async def update_strategy(
            strategy_id: str,
            user: TokenPayload = Depends(
                require_ownership_or_permission(
                    Permission.ADMIN_STRATEGIES,
                    lambda: get_strategy_owner(strategy_id)
                )
            )
        ):
            pass
    """
    async def checker(user: TokenPayload = Depends(get_current_user)) -> TokenPayload:
        perm_checker = get_permission_checker()
        
        # Check admin permission first
        if perm_checker.has_permission(user, permission):
            return user
        
        # Check ownership
        owner_id = resource_owner_id_getter()
        if perm_checker.is_owner(user.user_id, owner_id):
            return user
        
        raise HTTPException(
            status_code=403,
            detail="You don't have permission to access this resource"
        )
    return checker


# Decorators for non-FastAPI usage

def permission_required(permission: Permission):
    """
    Decorator to require a permission.
    
    Usage:
        @permission_required(Permission.STRATEGY_EXECUTE)
        def execute_strategy(user: TokenPayload, strategy_id: str):
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Find user in args or kwargs
            user = kwargs.get('user')
            if not user:
                for arg in args:
                    if isinstance(arg, TokenPayload):
                        user = arg
                        break
            
            if not user:
                raise HTTPException(status_code=401, detail="User not found")
            
            checker = get_permission_checker()
            checker.check_permission(user, permission)
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def admin_required(func: Callable) -> Callable:
    """Decorator to require admin role"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        user = kwargs.get('user')
        if not user:
            for arg in args:
                if isinstance(arg, TokenPayload):
                    user = arg
                    break
        
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        
        checker = get_permission_checker()
        if not checker.is_admin(user):
            raise HTTPException(status_code=403, detail="Admin access required")
        
        return func(*args, **kwargs)
    return wrapper


# Permission utilities

def list_user_permissions(user: TokenPayload) -> List[str]:
    """List all permissions for a user"""
    checker = get_permission_checker()
    perms = checker.get_user_permissions(user.roles, user.tier)
    return [p.value for p in perms]


def can_access_feature(user: TokenPayload, feature: str) -> bool:
    """Check if user can access a feature by name"""
    feature_permissions = {
        "live_trading": Permission.TRADE_EXECUTE,
        "api_keys": Permission.API_KEY_CREATE,
        "webhooks": Permission.WEBHOOK_CREATE,
        "team_sharing": Permission.STRATEGY_SHARE,
        "advanced_backtest": Permission.BACKTEST_CREATE,
    }
    
    perm = feature_permissions.get(feature)
    if not perm:
        return True  # Unknown features allowed by default
    
    checker = get_permission_checker()
    return checker.has_permission(user, perm)
