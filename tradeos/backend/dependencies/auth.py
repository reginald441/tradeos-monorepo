"""
TradeOS Authentication Dependencies
JWT token validation and user authentication dependencies for FastAPI.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Union
from uuid import UUID

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import settings
from database.connection import get_db_session
from database.models import ApiKey, ApiKeyPermission, User, UserRole

# Configure logging
logger = logging.getLogger(__name__)

# Password hashing context
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=settings.security.bcrypt_rounds
)

# OAuth2 scheme for token endpoint
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)

# HTTP Bearer scheme for API documentation
http_bearer = HTTPBearer(auto_error=False)


# ============================================================================
# Password Utilities
# ============================================================================

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain password against a hashed password.
    
    Args:
        plain_password: The plain text password.
        hashed_password: The hashed password to compare against.
    
    Returns:
        bool: True if passwords match, False otherwise.
    """
    return pwd_context.verify(plain_password, hashed_password)


def hash_password(password: str) -> str:
    """
    Hash a plain text password.
    
    Args:
        password: The plain text password to hash.
    
    Returns:
        str: The hashed password.
    """
    return pwd_context.hash(password)


def is_password_strong(password: str) -> tuple[bool, Optional[str]]:
    """
    Check if a password meets strength requirements.
    
    Args:
        password: The password to check.
    
    Returns:
        tuple: (is_strong, error_message)
    """
    min_length = settings.security.password_min_length
    
    if len(password) < min_length:
        return False, f"Password must be at least {min_length} characters long"
    
    if not any(c.isupper() for c in password):
        return False, "Password must contain at least one uppercase letter"
    
    if not any(c.islower() for c in password):
        return False, "Password must contain at least one lowercase letter"
    
    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one digit"
    
    if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
        return False, "Password must contain at least one special character"
    
    return True, None


# ============================================================================
# JWT Token Utilities
# ============================================================================

def create_access_token(
    user_id: Union[str, UUID],
    email: str,
    role: UserRole,
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a new JWT access token.
    
    Args:
        user_id: The user's ID.
        email: The user's email.
        role: The user's role.
        expires_delta: Optional custom expiration time.
    
    Returns:
        str: The encoded JWT token.
    """
    if expires_delta is None:
        expires_delta = timedelta(minutes=settings.jwt.access_token_expire_minutes)
    
    expire = datetime.utcnow() + expires_delta
    
    payload = {
        "sub": str(user_id),
        "email": email,
        "role": role.value,
        "type": "access",
        "exp": expire,
        "iat": datetime.utcnow(),
        "jti": str(UUID(int=datetime.utcnow().timestamp()))
    }
    
    encoded_jwt = jwt.encode(
        payload,
        settings.jwt.secret_key,
        algorithm=settings.jwt.algorithm
    )
    
    return encoded_jwt


def create_refresh_token(user_id: Union[str, UUID]) -> str:
    """
    Create a new JWT refresh token.
    
    Args:
        user_id: The user's ID.
    
    Returns:
        str: The encoded JWT refresh token.
    """
    expires_delta = timedelta(days=settings.jwt.refresh_token_expire_days)
    expire = datetime.utcnow() + expires_delta
    
    payload = {
        "sub": str(user_id),
        "type": "refresh",
        "exp": expire,
        "iat": datetime.utcnow(),
        "jti": str(UUID(int=datetime.utcnow().timestamp()))
    }
    
    encoded_jwt = jwt.encode(
        payload,
        settings.jwt.secret_key,
        algorithm=settings.jwt.algorithm
    )
    
    return encoded_jwt


def decode_token(token: str) -> Optional[dict]:
    """
    Decode and validate a JWT token.
    
    Args:
        token: The JWT token to decode.
    
    Returns:
        Optional[dict]: The decoded token payload or None if invalid.
    """
    try:
        payload = jwt.decode(
            token,
            settings.jwt.secret_key,
            algorithms=[settings.jwt.algorithm]
        )
        return payload
    except JWTError as e:
        logger.warning(f"Token decode error: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected token error: {e}")
        return None


def extract_token_from_request(request: Request) -> Optional[str]:
    """
    Extract JWT token from request headers or cookies.
    
    Args:
        request: The FastAPI request object.
    
    Returns:
        Optional[str]: The extracted token or None.
    """
    # Check Authorization header
    auth_header = request.headers.get("Authorization")
    if auth_header:
        parts = auth_header.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1]
    
    # Check cookies
    token = request.cookies.get("access_token")
    if token:
        return token
    
    return None


# ============================================================================
# User Authentication Dependencies
# ============================================================================

async def get_current_user(
    request: Request,
    db: AsyncSession = Depends(get_db_session),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(http_bearer)
) -> User:
    """
    Get the current authenticated user from JWT token.
    
    This dependency validates the JWT token and returns the user.
    
    Args:
        request: The FastAPI request object.
        db: Database session.
        credentials: HTTP Bearer credentials.
    
    Returns:
        User: The authenticated user.
    
    Raises:
        HTTPException: If authentication fails.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    # Get token from various sources
    token = None
    
    if credentials:
        token = credentials.credentials
    else:
        token = extract_token_from_request(request)
    
    if not token:
        raise credentials_exception
    
    # Decode and validate token
    payload = decode_token(token)
    if payload is None:
        raise credentials_exception
    
    # Check token type
    token_type = payload.get("type")
    if token_type != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get user ID from token
    user_id = payload.get("sub")
    if user_id is None:
        raise credentials_exception
    
    # Fetch user from database
    try:
        user_uuid = UUID(user_id)
    except ValueError:
        raise credentials_exception
    
    result = await db.execute(select(User).where(User.id == user_uuid))
    user = result.scalar_one_or_none()
    
    if user is None:
        raise credentials_exception
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is deactivated"
        )
    
    if user.locked_until and datetime.utcnow() < user.locked_until:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Account locked until {user.locked_until}"
        )
    
    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Get the current active user.
    
    Args:
        current_user: The authenticated user.
    
    Returns:
        User: The active user.
    
    Raises:
        HTTPException: If user is not active.
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    return current_user


async def get_current_verified_user(
    current_user: User = Depends(get_current_active_user)
) -> User:
    """
    Get the current verified user.
    
    Args:
        current_user: The active user.
    
    Returns:
        User: The verified user.
    
    Raises:
        HTTPException: If user is not verified.
    """
    if not current_user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email verification required"
        )
    return current_user


def require_role(required_role: UserRole):
    """
    Create a dependency that requires a specific user role.
    
    Args:
        required_role: The required role.
    
    Returns:
        Callable: Dependency function.
    """
    async def role_checker(current_user: User = Depends(get_current_user)) -> User:
        role_hierarchy = {
            UserRole.USER: 1,
            UserRole.ANALYST: 2,
            UserRole.ADMIN: 3
        }
        
        user_level = role_hierarchy.get(current_user.role, 0)
        required_level = role_hierarchy.get(required_role, 0)
        
        if user_level < required_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required role: {required_role.value}"
            )
        
        return current_user
    
    return role_checker


# Admin role dependency
require_admin = require_role(UserRole.ADMIN)


# ============================================================================
# API Key Authentication
# ============================================================================

async def get_api_key_user(
    request: Request,
    db: AsyncSession = Depends(get_db_session)
) -> tuple[User, ApiKey]:
    """
    Authenticate user via API key.
    
    Args:
        request: The FastAPI request object.
        db: Database session.
    
    Returns:
        tuple: (User, ApiKey) if authentication succeeds.
    
    Raises:
        HTTPException: If API key authentication fails.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API key",
        headers={"WWW-Authenticate": "ApiKey"},
    )
    
    # Get API key from header
    api_key = request.headers.get("X-API-Key")
    if not api_key:
        raise credentials_exception
    
    # Validate API key format
    if not api_key.startswith(settings.security.api_key_prefix):
        raise credentials_exception
    
    # Hash the provided key for comparison
    from hashlib import sha256
    key_hash = sha256(api_key.encode()).hexdigest()
    
    # Fetch API key from database
    result = await db.execute(
        select(ApiKey)
        .where(ApiKey.key_hash == key_hash)
        .where(ApiKey.is_active == True)
    )
    api_key_obj = result.scalar_one_or_none()
    
    if not api_key_obj:
        raise credentials_exception
    
    # Check expiration
    if api_key_obj.is_expired:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key has expired"
        )
    
    # Fetch associated user
    result = await db.execute(
        select(User).where(User.id == api_key_obj.user_id)
    )
    user = result.scalar_one_or_none()
    
    if not user or not user.is_active:
        raise credentials_exception
    
    # Update last used timestamp
    api_key_obj.last_used_at = datetime.utcnow()
    api_key_obj.request_count += 1
    await db.commit()
    
    return user, api_key_obj


async def authenticate_user_or_api_key(
    request: Request,
    db: AsyncSession = Depends(get_db_session),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(http_bearer)
) -> User:
    """
    Authenticate user via JWT token or API key.
    
    This tries JWT authentication first, then falls back to API key.
    
    Args:
        request: The FastAPI request object.
        db: Database session.
        credentials: HTTP Bearer credentials.
    
    Returns:
        User: The authenticated user.
    """
    # Try JWT authentication first
    try:
        return await get_current_user(request, db, credentials)
    except HTTPException:
        # Fall back to API key
        user, _ = await get_api_key_user(request, db)
        return user


def require_api_permission(permission: ApiKeyPermission):
    """
    Create a dependency that requires a specific API key permission.
    
    Args:
        permission: The required permission.
    
    Returns:
        Callable: Dependency function.
    """
    async def permission_checker(
        request: Request,
        db: AsyncSession = Depends(get_db_session)
    ) -> User:
        user, api_key = await get_api_key_user(request, db)
        
        if not api_key.has_permission(permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"API key missing required permission: {permission.value}"
            )
        
        return user
    
    return permission_checker


# ============================================================================
# Optional Authentication
# ============================================================================

async def get_optional_user(
    request: Request,
    db: AsyncSession = Depends(get_db_session),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(http_bearer)
) -> Optional[User]:
    """
    Get the current user if authenticated, otherwise None.
    
    This is useful for endpoints that work with or without authentication.
    
    Args:
        request: The FastAPI request object.
        db: Database session.
        credentials: HTTP Bearer credentials.
    
    Returns:
        Optional[User]: The user if authenticated, None otherwise.
    """
    try:
        return await get_current_user(request, db, credentials)
    except HTTPException:
        return None


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Password utilities
    "verify_password",
    "hash_password",
    "is_password_strong",
    # Token utilities
    "create_access_token",
    "create_refresh_token",
    "decode_token",
    "extract_token_from_request",
    # Dependencies
    "get_current_user",
    "get_current_active_user",
    "get_current_verified_user",
    "require_role",
    "require_admin",
    "get_api_key_user",
    "authenticate_user_or_api_key",
    "require_api_permission",
    "get_optional_user",
    # Schemes
    "oauth2_scheme",
    "http_bearer",
    "pwd_context",
]
