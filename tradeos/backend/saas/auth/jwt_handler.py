"""
TradeOS JWT Authentication Handler
==================================
Secure JWT token generation, verification, and management.
"""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import jwt
from fastapi import HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from ..config.saas_config import get_jwt_config, JWTConfig


# Token types
TOKEN_TYPE_ACCESS = "access"
TOKEN_TYPE_REFRESH = "refresh"

# HTTP Bearer security scheme
security_bearer = HTTPBearer(auto_error=False)


@dataclass
class TokenPayload:
    """Decoded JWT token payload"""
    user_id: str
    email: str
    tier: str
    roles: List[str]
    token_type: str
    jti: str  # JWT ID for blacklisting
    iat: datetime
    exp: datetime
    iss: str
    aud: str
    permissions: List[str] = None
    
    def __post_init__(self):
        if self.permissions is None:
            self.permissions = []


@dataclass
class TokenPair:
    """Access and refresh token pair"""
    access_token: str
    refresh_token: str
    access_token_expires: datetime
    refresh_token_expires: datetime
    token_type: str = "bearer"


class JWTHandler:
    """
    JWT Token Handler for TradeOS
    
    Handles:
    - Token generation (access + refresh)
    - Token verification and validation
    - Token refresh logic
    - Token blacklisting for logout
    """
    
    def __init__(self, config: Optional[JWTConfig] = None):
        self.config = config or get_jwt_config()
        self._blacklist: set = set()  # In production, use Redis
        self._refresh_token_store: Dict[str, str] = {}  # user_id -> jti mapping
    
    def _generate_jti(self) -> str:
        """Generate unique JWT ID"""
        return str(uuid.uuid4())
    
    def _create_token(
        self,
        user_id: str,
        email: str,
        tier: str,
        roles: List[str],
        token_type: str,
        expires_delta: timedelta,
        permissions: Optional[List[str]] = None,
        extra_claims: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, datetime]:
        """
        Create a JWT token with the specified claims.
        
        Args:
            user_id: Unique user identifier
            email: User email address
            tier: Subscription tier (free, pro, enterprise)
            roles: List of user roles
            token_type: Type of token (access or refresh)
            expires_delta: Token expiration time
            permissions: Optional list of permissions
            extra_claims: Optional additional claims
            
        Returns:
            Tuple of (token string, expiration datetime)
        """
        now = datetime.now(timezone.utc)
        expires = now + expires_delta
        jti = self._generate_jti()
        
        payload = {
            "sub": user_id,
            "email": email,
            "tier": tier,
            "roles": roles,
            "type": token_type,
            "jti": jti,
            "iat": now,
            "exp": expires,
            "iss": self.config.token_issuer,
            "aud": self.config.token_audience,
        }
        
        if permissions:
            payload["permissions"] = permissions
            
        if extra_claims:
            payload.update(extra_claims)
        
        token = jwt.encode(
            payload,
            self.config.secret_key,
            algorithm=self.config.algorithm
        )
        
        return token, expires
    
    def create_token_pair(
        self,
        user_id: str,
        email: str,
        tier: str,
        roles: List[str],
        permissions: Optional[List[str]] = None,
        extra_claims: Optional[Dict[str, Any]] = None
    ) -> TokenPair:
        """
        Create a new access and refresh token pair.
        
        Args:
            user_id: Unique user identifier
            email: User email address
            tier: Subscription tier
            roles: List of user roles
            permissions: Optional list of permissions
            extra_claims: Optional additional claims
            
        Returns:
            TokenPair containing access and refresh tokens
        """
        # Create access token
        access_token, access_expires = self._create_token(
            user_id=user_id,
            email=email,
            tier=tier,
            roles=roles,
            token_type=TOKEN_TYPE_ACCESS,
            expires_delta=timedelta(minutes=self.config.access_token_expire_minutes),
            permissions=permissions,
            extra_claims=extra_claims
        )
        
        # Create refresh token
        refresh_token, refresh_expires = self._create_token(
            user_id=user_id,
            email=email,
            tier=tier,
            roles=roles,
            token_type=TOKEN_TYPE_REFRESH,
            expires_delta=timedelta(days=self.config.refresh_token_expire_days),
            extra_claims={"access_jti": self._get_token_jti(access_token)}
        )
        
        # Store refresh token mapping
        refresh_jti = self._get_token_jti(refresh_token)
        self._refresh_token_store[user_id] = refresh_jti
        
        return TokenPair(
            access_token=access_token,
            refresh_token=refresh_token,
            access_token_expires=access_expires,
            refresh_token_expires=refresh_expires
        )
    
    def _get_token_jti(self, token: str) -> str:
        """Extract JTI from token without full verification"""
        try:
            payload = jwt.decode(
                token,
                self.config.secret_key,
                algorithms=[self.config.algorithm],
                options={"verify_exp": False}
            )
            return payload.get("jti", "")
        except jwt.InvalidTokenError:
            return ""
    
    def decode_token(
        self,
        token: str,
        token_type: Optional[str] = None,
        verify_exp: bool = True
    ) -> TokenPayload:
        """
        Decode and verify a JWT token.
        
        Args:
            token: JWT token string
            token_type: Expected token type (access or refresh)
            verify_exp: Whether to verify expiration
            
        Returns:
            TokenPayload with decoded claims
            
        Raises:
            HTTPException: If token is invalid or expired
        """
        try:
            payload = jwt.decode(
                token,
                self.config.secret_key,
                algorithms=[self.config.algorithm],
                issuer=self.config.token_issuer,
                audience=self.config.token_audience,
                options={"verify_exp": verify_exp}
            )
            
            # Check token type if specified
            if token_type and payload.get("type") != token_type:
                raise HTTPException(
                    status_code=401,
                    detail=f"Invalid token type. Expected {token_type}."
                )
            
            # Check blacklist
            jti = payload.get("jti")
            if jti and jti in self._blacklist:
                raise HTTPException(
                    status_code=401,
                    detail="Token has been revoked."
                )
            
            return TokenPayload(
                user_id=payload.get("sub"),
                email=payload.get("email"),
                tier=payload.get("tier"),
                roles=payload.get("roles", []),
                token_type=payload.get("type"),
                jti=jti,
                iat=datetime.fromtimestamp(payload.get("iat"), tz=timezone.utc),
                exp=datetime.fromtimestamp(payload.get("exp"), tz=timezone.utc),
                iss=payload.get("iss"),
                aud=payload.get("aud"),
                permissions=payload.get("permissions", [])
            )
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=401,
                detail="Token has expired."
            )
        except jwt.InvalidIssuerError:
            raise HTTPException(
                status_code=401,
                detail="Invalid token issuer."
            )
        except jwt.InvalidAudienceError:
            raise HTTPException(
                status_code=401,
                detail="Invalid token audience."
            )
        except jwt.InvalidTokenError as e:
            raise HTTPException(
                status_code=401,
                detail=f"Invalid token: {str(e)}"
            )
    
    def verify_access_token(self, token: str) -> TokenPayload:
        """Verify an access token"""
        return self.decode_token(token, token_type=TOKEN_TYPE_ACCESS)
    
    def verify_refresh_token(self, token: str) -> TokenPayload:
        """Verify a refresh token"""
        return self.decode_token(token, token_type=TOKEN_TYPE_REFRESH)
    
    def refresh_access_token(
        self,
        refresh_token: str,
        user_data_callback: Optional[callable] = None
    ) -> TokenPair:
        """
        Create a new token pair using a refresh token.
        
        Args:
            refresh_token: Valid refresh token
            user_data_callback: Optional callback to fetch fresh user data
            
        Returns:
            New TokenPair
            
        Raises:
            HTTPException: If refresh token is invalid
        """
        # Verify refresh token
        payload = self.verify_refresh_token(refresh_token)
        
        # Check if refresh token is stored for this user
        stored_jti = self._refresh_token_store.get(payload.user_id)
        if stored_jti != payload.jti:
            raise HTTPException(
                status_code=401,
                detail="Refresh token has been revoked."
            )
        
        # Blacklist the old refresh token
        self.blacklist_token(refresh_token)
        
        # Get fresh user data if callback provided
        if user_data_callback:
            user_data = user_data_callback(payload.user_id)
            return self.create_token_pair(
                user_id=user_data["id"],
                email=user_data["email"],
                tier=user_data["tier"],
                roles=user_data["roles"],
                permissions=user_data.get("permissions")
            )
        
        # Otherwise use data from refresh token
        return self.create_token_pair(
            user_id=payload.user_id,
            email=payload.email,
            tier=payload.tier,
            roles=payload.roles,
            permissions=payload.permissions
        )
    
    def blacklist_token(self, token: str) -> bool:
        """
        Add a token to the blacklist (for logout).
        
        Args:
            token: Token to blacklist
            
        Returns:
            True if successfully blacklisted
        """
        try:
            jti = self._get_token_jti(token)
            if jti:
                self._blacklist.add(jti)
                return True
            return False
        except Exception:
            return False
    
    def is_token_blacklisted(self, jti: str) -> bool:
        """Check if a token JTI is blacklisted"""
        return jti in self._blacklist
    
    def revoke_all_user_tokens(self, user_id: str) -> bool:
        """
        Revoke all tokens for a user (force re-login).
        
        Args:
            user_id: User ID to revoke tokens for
            
        Returns:
            True if successful
        """
        if user_id in self._refresh_token_store:
            del self._refresh_token_store[user_id]
        return True
    
    def get_token_expiry(self, token: str) -> Optional[datetime]:
        """Get token expiration time"""
        try:
            payload = jwt.decode(
                token,
                self.config.secret_key,
                algorithms=[self.config.algorithm],
                options={"verify_exp": False}
            )
            exp_timestamp = payload.get("exp")
            if exp_timestamp:
                return datetime.fromtimestamp(exp_timestamp, tz=timezone.utc)
            return None
        except jwt.InvalidTokenError:
            return None


# Global JWT handler instance
_jwt_handler: Optional[JWTHandler] = None


def get_jwt_handler() -> JWTHandler:
    """Get or create global JWT handler"""
    global _jwt_handler
    if _jwt_handler is None:
        _jwt_handler = JWTHandler()
    return _jwt_handler


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security_bearer)
) -> TokenPayload:
    """
    FastAPI dependency to get current authenticated user.
    
    Usage:
        @app.get("/protected")
        async def protected_route(user: TokenPayload = Depends(get_current_user)):
            return {"user_id": user.user_id}
    """
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    handler = get_jwt_handler()
    return handler.verify_access_token(credentials.credentials)


async def get_current_active_user(
    user: TokenPayload = Depends(get_current_user)
) -> TokenPayload:
    """Get current user and verify they are active"""
    # Add additional checks here (e.g., account status)
    return user


def require_tiers(*allowed_tiers: str):
    """
    Create a dependency that requires specific subscription tiers.
    
    Usage:
        @app.get("/pro-feature")
        async def pro_feature(user: TokenPayload = Depends(require_tiers("pro", "enterprise"))):
            return {"message": "Pro feature"}
    """
    async def tier_checker(user: TokenPayload = Depends(get_current_user)) -> TokenPayload:
        if user.tier not in allowed_tiers:
            raise HTTPException(
                status_code=403,
                detail=f"This feature requires one of these tiers: {', '.join(allowed_tiers)}"
            )
        return user
    return tier_checker


def require_roles(*required_roles: str):
    """
    Create a dependency that requires specific roles.
    
    Usage:
        @app.get("/admin-only")
        async def admin_route(user: TokenPayload = Depends(require_roles("admin"))):
            return {"message": "Admin only"}
    """
    async def role_checker(user: TokenPayload = Depends(get_current_user)) -> TokenPayload:
        if not any(role in user.roles for role in required_roles):
            raise HTTPException(
                status_code=403,
                detail=f"This action requires one of these roles: {', '.join(required_roles)}"
            )
        return user
    return role_checker
