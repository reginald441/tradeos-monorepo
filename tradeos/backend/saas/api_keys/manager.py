"""
TradeOS API Key Management
==========================
Secure API key generation, storage, and validation.
"""

import secrets
import hashlib
import hmac
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum

from fastapi import HTTPException, Request, Depends
from fastapi.security import APIKeyHeader

from ..config.saas_config import get_config, SecurityConfig


class APIKeyPermission(Enum):
    """Permissions that can be granted to API keys"""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    DELETE = "delete"
    ADMIN = "admin"


class APIKeyStatus(Enum):
    """API key status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    REVOKED = "revoked"
    EXPIRED = "expired"


@dataclass
class APIKey:
    """API key entity"""
    id: str
    user_id: str
    name: str
    # Store only the hash, never the full key
    key_hash: str
    key_prefix: str  # First 8 chars for identification
    permissions: List[str]
    status: APIKeyStatus
    created_at: datetime
    expires_at: Optional[datetime]
    last_used_at: Optional[datetime] = None
    usage_count: int = 0
    rate_limit: Optional[int] = None  # Requests per minute
    allowed_ips: List[str] = field(default_factory=list)
    allowed_endpoints: List[str] = field(default_factory=list)  # Empty = all
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Convert to dictionary (safe for API response)"""
        data = {
            "id": self.id,
            "name": self.name,
            "prefix": self.key_prefix,
            "permissions": self.permissions,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "usage_count": self.usage_count,
            "rate_limit": self.rate_limit,
            "allowed_ips": self.allowed_ips if self.allowed_ips else None,
        }
        
        if include_sensitive:
            data["key_hash"] = self.key_hash[:16] + "..."  # Truncated
        
        return data
    
    def is_active(self) -> bool:
        """Check if key is active and not expired"""
        if self.status != APIKeyStatus.ACTIVE:
            return False
        
        if self.expires_at and self.expires_at < datetime.now(timezone.utc):
            return False
        
        return True
    
    def has_permission(self, permission: str) -> bool:
        """Check if key has a specific permission"""
        return permission in self.permissions or APIKeyPermission.ADMIN.value in self.permissions


@dataclass
class APIKeyUsage:
    """API key usage record"""
    key_id: str
    timestamp: datetime
    endpoint: str
    method: str
    status_code: int
    response_time_ms: int
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


# API Key header security scheme
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class APIKeyManager:
    """
    API Key Manager for TradeOS.
    
    Handles:
    - Secure key generation using cryptographically secure random
    - Key hashing for storage (never store raw keys)
    - Key validation middleware
    - Permission scoping per key
    - Usage tracking per key
    - IP allowlisting
    - Rate limiting per key
    """
    
    def __init__(self, security_config: Optional[SecurityConfig] = None):
        self.config = security_config or get_config().security
        
        # In-memory storage (use database in production)
        self._keys: Dict[str, APIKey] = {}  # key_id -> APIKey
        self._key_index: Dict[str, str] = {}  # key_hash -> key_id (for lookup)
        self._user_keys: Dict[str, List[str]] = {}  # user_id -> [key_ids]
        self._usage_logs: List[APIKeyUsage] = []  # Recent usage logs
        self._rate_limit_windows: Dict[str, List[datetime]] = {}  # key_id -> timestamps
    
    def _generate_key(self) -> str:
        """
        Generate a cryptographically secure API key.
        
        Format: tr_<64 random chars>
        
        Returns:
            Full API key (shown only once to user)
        """
        random_part = secrets.token_urlsafe(self.config.api_key_length)
        return f"{self.config.api_key_prefix}{random_part}"
    
    def _hash_key(self, key: str) -> str:
        """
        Hash an API key for storage.
        
        Uses SHA-256 for one-way hashing.
        
        Args:
            key: Raw API key
            
        Returns:
            Hashed key
        """
        return hashlib.sha256(key.encode()).hexdigest()
    
    def _get_key_prefix(self, key: str) -> str:
        """Get prefix for key identification"""
        # Get chars after prefix, take first 8
        key_part = key[len(self.config.api_key_prefix):]
        return key_part[:8]
    
    def create_key(
        self,
        user_id: str,
        name: str,
        permissions: Optional[List[str]] = None,
        expires_in_days: Optional[int] = None,
        rate_limit: Optional[int] = None,
        allowed_ips: Optional[List[str]] = None,
        allowed_endpoints: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> tuple[APIKey, str]:
        """
        Create a new API key.
        
        Args:
            user_id: Owner user ID
            name: Key name/description
            permissions: List of permissions (default: ["read"])
            expires_in_days: Days until expiration (None = no expiration)
            rate_limit: Requests per minute limit
            allowed_ips: Allowed IP addresses (empty = all)
            allowed_endpoints: Allowed API endpoints (empty = all)
            metadata: Additional metadata
            
        Returns:
            Tuple of (APIKey object, raw key string)
            
        Note:
            The raw key is shown ONLY ONCE. Store it securely.
        """
        # Generate key
        raw_key = self._generate_key()
        key_hash = self._hash_key(raw_key)
        key_prefix = self._get_key_prefix(raw_key)
        
        # Calculate expiration
        expires_at = None
        if expires_in_days:
            expires_at = datetime.now(timezone.utc) + timedelta(days=expires_in_days)
        
        # Create key entity
        key_id = f"key_{secrets.token_urlsafe(16)}"
        
        api_key = APIKey(
            id=key_id,
            user_id=user_id,
            name=name,
            key_hash=key_hash,
            key_prefix=key_prefix,
            permissions=permissions or [APIKeyPermission.READ.value],
            status=APIKeyStatus.ACTIVE,
            created_at=datetime.now(timezone.utc),
            expires_at=expires_at,
            rate_limit=rate_limit,
            allowed_ips=allowed_ips or [],
            allowed_endpoints=allowed_endpoints or [],
            metadata=metadata or {}
        )
        
        # Store
        self._keys[key_id] = api_key
        self._key_index[key_hash] = key_id
        
        if user_id not in self._user_keys:
            self._user_keys[user_id] = []
        self._user_keys[user_id].append(key_id)
        
        return api_key, raw_key
    
    def get_key(self, key_id: str) -> Optional[APIKey]:
        """Get API key by ID"""
        return self._keys.get(key_id)
    
    def get_key_by_hash(self, key_hash: str) -> Optional[APIKey]:
        """Get API key by hash"""
        key_id = self._key_index.get(key_hash)
        if key_id:
            return self._keys.get(key_id)
        return None
    
    def validate_key(self, raw_key: str) -> Optional[APIKey]:
        """
        Validate an API key.
        
        Args:
            raw_key: Raw API key from request
            
        Returns:
            APIKey if valid, None otherwise
        """
        # Check prefix
        if not raw_key.startswith(self.config.api_key_prefix):
            return None
        
        # Hash and lookup
        key_hash = self._hash_key(raw_key)
        api_key = self.get_key_by_hash(key_hash)
        
        if not api_key:
            return None
        
        # Check status
        if not api_key.is_active():
            return None
        
        return api_key
    
    def revoke_key(self, key_id: str, user_id: Optional[str] = None) -> bool:
        """
        Revoke an API key.
        
        Args:
            key_id: Key ID to revoke
            user_id: Optional user ID for ownership verification
            
        Returns:
            True if revoked
        """
        api_key = self._keys.get(key_id)
        if not api_key:
            return False
        
        # Verify ownership if user_id provided
        if user_id and api_key.user_id != user_id:
            raise HTTPException(status_code=403, detail="Not authorized to revoke this key")
        
        api_key.status = APIKeyStatus.REVOKED
        
        # Remove from index
        if api_key.key_hash in self._key_index:
            del self._key_index[api_key.key_hash]
        
        return True
    
    def delete_key(self, key_id: str, user_id: Optional[str] = None) -> bool:
        """Permanently delete an API key"""
        api_key = self._keys.get(key_id)
        if not api_key:
            return False
        
        if user_id and api_key.user_id != user_id:
            raise HTTPException(status_code=403, detail="Not authorized to delete this key")
        
        # Remove from all indexes
        del self._keys[key_id]
        
        if api_key.key_hash in self._key_index:
            del self._key_index[api_key.key_hash]
        
        if api_key.user_id in self._user_keys:
            self._user_keys[api_key.user_id] = [
                k for k in self._user_keys[api_key.user_id] if k != key_id
            ]
        
        return True
    
    def list_keys(
        self,
        user_id: str,
        include_revoked: bool = False
    ) -> List[APIKey]:
        """List all API keys for a user"""
        key_ids = self._user_keys.get(user_id, [])
        keys = [self._keys[k] for k in key_ids if k in self._keys]
        
        if not include_revoked:
            keys = [k for k in keys if k.status != APIKeyStatus.REVOKED]
        
        return sorted(keys, key=lambda k: k.created_at, reverse=True)
    
    def update_key(
        self,
        key_id: str,
        user_id: str,
        **updates
    ) -> APIKey:
        """
        Update API key properties.
        
        Args:
            key_id: Key ID
            user_id: User ID (for ownership verification)
            **updates: Fields to update
            
        Returns:
            Updated APIKey
        """
        api_key = self._keys.get(key_id)
        if not api_key:
            raise HTTPException(status_code=404, detail="API key not found")
        
        if api_key.user_id != user_id:
            raise HTTPException(status_code=403, detail="Not authorized")
        
        # Update allowed fields
        if "name" in updates:
            api_key.name = updates["name"]
        
        if "permissions" in updates:
            api_key.permissions = updates["permissions"]
        
        if "rate_limit" in updates:
            api_key.rate_limit = updates["rate_limit"]
        
        if "allowed_ips" in updates:
            api_key.allowed_ips = updates["allowed_ips"]
        
        if "allowed_endpoints" in updates:
            api_key.allowed_endpoints = updates["allowed_endpoints"]
        
        if "status" in updates:
            new_status = updates["status"]
            if new_status in [s.value for s in APIKeyStatus]:
                api_key.status = APIKeyStatus(new_status)
        
        return api_key
    
    def record_usage(
        self,
        key_id: str,
        endpoint: str,
        method: str,
        status_code: int,
        response_time_ms: int,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> None:
        """Record API key usage"""
        api_key = self._keys.get(key_id)
        if api_key:
            api_key.usage_count += 1
            api_key.last_used_at = datetime.now(timezone.utc)
        
        # Log usage
        usage = APIKeyUsage(
            key_id=key_id,
            timestamp=datetime.now(timezone.utc),
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            response_time_ms=response_time_ms,
            ip_address=ip_address,
            user_agent=user_agent
        )
        self._usage_logs.append(usage)
        
        # Trim logs if too many
        if len(self._usage_logs) > 10000:
            self._usage_logs = self._usage_logs[-5000:]
    
    def check_rate_limit(self, key_id: str) -> tuple[bool, int, int]:
        """
        Check if key is within rate limit.
        
        Returns:
            Tuple of (is_allowed, limit, remaining)
        """
        api_key = self._keys.get(key_id)
        if not api_key:
            return False, 0, 0
        
        if not api_key.rate_limit:
            return True, -1, -1
        
        now = datetime.now(timezone.utc)
        window_start = now - timedelta(minutes=1)
        
        # Get recent requests
        if key_id not in self._rate_limit_windows:
            self._rate_limit_windows[key_id] = []
        
        # Filter to current window
        self._rate_limit_windows[key_id] = [
            t for t in self._rate_limit_windows[key_id]
            if t > window_start
        ]
        
        current_count = len(self._rate_limit_windows[key_id])
        
        if current_count >= api_key.rate_limit:
            return False, api_key.rate_limit, 0
        
        # Record this request
        self._rate_limit_windows[key_id].append(now)
        
        return True, api_key.rate_limit, api_key.rate_limit - current_count - 1
    
    def check_ip_allowed(self, key_id: str, ip_address: str) -> bool:
        """Check if IP is allowed for key"""
        api_key = self._keys.get(key_id)
        if not api_key:
            return False
        
        # If no allowed IPs specified, allow all
        if not api_key.allowed_ips:
            return True
        
        return ip_address in api_key.allowed_ips
    
    def check_endpoint_allowed(self, key_id: str, endpoint: str) -> bool:
        """Check if endpoint is allowed for key"""
        api_key = self._keys.get(key_id)
        if not api_key:
            return False
        
        # If no allowed endpoints specified, allow all
        if not api_key.allowed_endpoints:
            return True
        
        # Check if endpoint matches any allowed pattern
        for allowed in api_key.allowed_endpoints:
            if endpoint.startswith(allowed) or endpoint == allowed:
                return True
        
        return False
    
    def get_usage_stats(
        self,
        key_id: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get usage statistics for a key"""
        api_key = self._keys.get(key_id)
        if not api_key:
            raise HTTPException(status_code=404, detail="API key not found")
        
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        
        # Filter logs
        key_logs = [
            log for log in self._usage_logs
            if log.key_id == key_id and log.timestamp > cutoff
        ]
        
        # Calculate stats
        total_requests = len(key_logs)
        
        endpoint_counts = {}
        status_counts = {}
        response_times = []
        
        for log in key_logs:
            endpoint_counts[log.endpoint] = endpoint_counts.get(log.endpoint, 0) + 1
            status_counts[log.status_code] = status_counts.get(log.status_code, 0) + 1
            response_times.append(log.response_time_ms)
        
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        return {
            "key_id": key_id,
            "period_days": days,
            "total_requests": total_requests,
            "endpoint_breakdown": endpoint_counts,
            "status_code_breakdown": status_counts,
            "average_response_time_ms": round(avg_response_time, 2),
            "lifetime_usage": api_key.usage_count
        }


# Global API key manager instance
_api_key_manager: Optional[APIKeyManager] = None


def get_api_key_manager() -> APIKeyManager:
    """Get or create global API key manager"""
    global _api_key_manager
    if _api_key_manager is None:
        _api_key_manager = APIKeyManager()
    return _api_key_manager


# FastAPI middleware for API key authentication

async def validate_api_key(
    request: Request,
    api_key: Optional[str] = Depends(api_key_header)
) -> Optional[APIKey]:
    """
    FastAPI dependency to validate API key from header.
    
    Usage:
        @app.get("/api/data")
        async def get_data(api_key: APIKey = Depends(validate_api_key)):
            pass
    """
    if not api_key:
        # Try query parameter
        api_key = request.query_params.get("api_key")
    
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required",
            headers={"WWW-Authenticate": "ApiKey"}
        )
    
    manager = get_api_key_manager()
    key_obj = manager.validate_key(api_key)
    
    if not key_obj:
        raise HTTPException(
            status_code=401,
            detail="Invalid or revoked API key"
        )
    
    # Check IP allowlist
    client_ip = request.client.host if request.client else None
    if client_ip and not manager.check_ip_allowed(key_obj.id, client_ip):
        raise HTTPException(
            status_code=403,
            detail="IP address not allowed for this API key"
        )
    
    # Check rate limit
    allowed, limit, remaining = manager.check_rate_limit(key_obj.id)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={"X-RateLimit-Limit": str(limit), "X-RateLimit-Remaining": "0"}
        )
    
    # Check endpoint access
    if not manager.check_endpoint_allowed(key_obj.id, request.url.path):
        raise HTTPException(
            status_code=403,
            detail="This API key cannot access this endpoint"
        )
    
    # Store in request state for later use
    request.state.api_key = key_obj
    request.state.rate_limit_remaining = remaining
    
    return key_obj


async def require_api_key_permission(permission: str):
    """
    Require a specific permission for API key.
    
    Usage:
        @app.post("/api/execute")
        async def execute(
            api_key: APIKey = Depends(require_api_key_permission("execute"))
        ):
            pass
    """
    async def checker(
        api_key: APIKey = Depends(validate_api_key)
    ) -> APIKey:
        if not api_key.has_permission(permission):
            raise HTTPException(
                status_code=403,
                detail=f"API key lacks required permission: {permission}"
            )
        return api_key
    return checker


class APIKeyMiddleware:
    """
    Middleware to automatically validate API keys and track usage.
    
    Usage:
        app.add_middleware(APIKeyMiddleware)
    """
    
    def __init__(self, app, protected_paths: Optional[List[str]] = None):
        self.app = app
        self.protected_paths = protected_paths or ["/api/v1/"]
        self.manager = get_api_key_manager()
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        request = Request(scope, receive)
        
        # Check if path is protected
        is_protected = any(
            request.url.path.startswith(path)
            for path in self.protected_paths
        )
        
        if is_protected:
            api_key = request.headers.get("X-API-Key")
            
            if api_key:
                key_obj = self.manager.validate_key(api_key)
                
                if key_obj:
                    # Store in request state
                    request.state.api_key = key_obj
        
        await self.app(scope, receive, send)
