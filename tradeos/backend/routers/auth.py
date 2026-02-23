"""
Authentication Router

Handles user registration, login, logout, token refresh, and OAuth.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime

from ..saas.auth.jwt_handler import JWTHandler
from ..saas.users.user_manager import UserManager
from ..saas.auth.oauth import OAuthHandler
from ..database.connection import get_db_session
from ..dependencies.auth import get_current_user
from ..database.models import User

router = APIRouter(prefix="/auth", tags=["Authentication"])


# Request/Response Models
class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None


class RegisterResponse(BaseModel):
    success: bool
    message: str
    user_id: Optional[int] = None


class LoginResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: dict


class TokenRefreshRequest(BaseModel):
    refresh_token: str


class TokenRefreshResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class PasswordResetRequest(BaseModel):
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str


class OAuthUrlResponse(BaseModel):
    authorization_url: str
    state: str


# Initialize handlers
jwt_handler = JWTHandler()
oauth_handler = OAuthHandler()


@router.post("/register", response_model=RegisterResponse)
async def register(request: RegisterRequest):
    """Register a new user account."""
    try:
        user_manager = UserManager()
        user = await user_manager.register_user(
            email=request.email,
            password=request.password,
            first_name=request.first_name,
            last_name=request.last_name
        )
        return RegisterResponse(
            success=True,
            message="Registration successful. Please verify your email.",
            user_id=user.id
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/login", response_model=LoginResponse)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Authenticate user and return JWT tokens."""
    user_manager = UserManager()
    user = await user_manager.authenticate_user(
        email=form_data.username,
        password=form_data.password
    )
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Generate tokens
    access_token = jwt_handler.create_access_token(
        user_id=user.id,
        email=user.email,
        role=user.role
    )
    refresh_token = jwt_handler.create_refresh_token(user_id=user.id)
    
    return LoginResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=3600,  # 1 hour
        user={
            "id": user.id,
            "email": user.email,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "role": user.role,
            "subscription_tier": user.subscription_tier,
            "is_verified": user.is_verified
        }
    )


@router.post("/refresh", response_model=TokenRefreshResponse)
async def refresh_token(request: TokenRefreshRequest):
    """Refresh access token using refresh token."""
    try:
        # Verify refresh token
        payload = jwt_handler.verify_token(request.refresh_token, token_type="refresh")
        user_id = int(payload.get("sub"))
        
        # Generate new tokens
        user_manager = UserManager()
        user = await user_manager.get_user_by_id(user_id)
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        access_token = jwt_handler.create_access_token(
            user_id=user.id,
            email=user.email,
            role=user.role
        )
        refresh_token = jwt_handler.create_refresh_token(user_id=user.id)
        
        # Blacklist old refresh token
        await jwt_handler.blacklist_token(request.refresh_token)
        
        return TokenRefreshResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=3600
        )
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid refresh token: {str(e)}")


@router.post("/logout")
async def logout(request: Request, current_user: User = Depends(get_current_user)):
    """Logout user and invalidate tokens."""
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        await jwt_handler.blacklist_token(token)
    
    return {"success": True, "message": "Logged out successfully"}


@router.post("/password-reset-request")
async def request_password_reset(request: PasswordResetRequest):
    """Request password reset email."""
    user_manager = UserManager()
    await user_manager.request_password_reset(email=request.email)
    return {
        "success": True,
        "message": "If an account exists with this email, you will receive a password reset link."
    }


@router.post("/password-reset-confirm")
async def confirm_password_reset(request: PasswordResetConfirm):
    """Confirm password reset with token."""
    user_manager = UserManager()
    success = await user_manager.reset_password(
        token=request.token,
        new_password=request.new_password
    )
    
    if success:
        return {"success": True, "message": "Password reset successful"}
    else:
        raise HTTPException(status_code=400, detail="Invalid or expired token")


@router.get("/oauth/google", response_model=OAuthUrlResponse)
async def google_oauth_url():
    """Get Google OAuth authorization URL."""
    url, state = await oauth_handler.get_google_auth_url()
    return OAuthUrlResponse(authorization_url=url, state=state)


@router.get("/oauth/google/callback")
async def google_oauth_callback(code: str, state: str):
    """Handle Google OAuth callback."""
    try:
        user_info = await oauth_handler.handle_google_callback(code, state)
        
        # Create or get user
        user_manager = UserManager()
        user = await user_manager.get_or_create_oauth_user(
            email=user_info["email"],
            provider="google",
            provider_id=user_info["id"],
            first_name=user_info.get("given_name"),
            last_name=user_info.get("family_name")
        )
        
        # Generate tokens
        access_token = jwt_handler.create_access_token(
            user_id=user.id,
            email=user.email,
            role=user.role
        )
        refresh_token = jwt_handler.create_refresh_token(user_id=user.id)
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": 3600,
            "user": {
                "id": user.id,
                "email": user.email,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "role": user.role
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"OAuth failed: {str(e)}")


@router.get("/me")
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user profile."""
    return {
        "id": current_user.id,
        "email": current_user.email,
        "first_name": current_user.first_name,
        "last_name": current_user.last_name,
        "role": current_user.role,
        "subscription_tier": current_user.subscription_tier,
        "is_verified": current_user.is_verified,
        "created_at": current_user.created_at.isoformat() if current_user.created_at else None
    }
