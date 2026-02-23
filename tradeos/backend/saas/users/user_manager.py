"""
TradeOS User Management
=======================
User registration, authentication, and profile management.
"""

import re
import secrets
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

import bcrypt
from fastapi import HTTPException, BackgroundTasks

from ..config.saas_config import get_config, SecurityConfig
from ..auth.jwt_handler import JWTHandler, TokenPair, get_jwt_handler
from ..notifications.email import EmailService, get_email_service


class UserStatus(Enum):
    PENDING = "pending"          # Awaiting email verification
    ACTIVE = "active"            # Fully active user
    SUSPENDED = "suspended"      # Temporarily suspended
    DEACTIVATED = "deactivated"  # User deactivated account


class AccountType(Enum):
    INDIVIDUAL = "individual"
    TEAM = "team"
    ENTERPRISE = "enterprise"


@dataclass
class UserProfile:
    """User profile data"""
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    company: Optional[str] = None
    job_title: Optional[str] = None
    phone: Optional[str] = None
    timezone: str = "UTC"
    preferences: Dict[str, Any] = field(default_factory=dict)


@dataclass
class User:
    """User entity"""
    id: str
    email: str
    hashed_password: Optional[str]  # None for OAuth users
    status: UserStatus
    tier: str  # free, pro, enterprise
    roles: List[str]
    email_verified: bool
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None
    profile: UserProfile = field(default_factory=UserProfile)
    account_type: AccountType = AccountType.INDIVIDUAL
    oauth_providers: Dict[str, str] = field(default_factory=dict)  # provider -> provider_user_id
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None
    login_attempts: int = 0
    locked_until: Optional[datetime] = None
    password_changed_at: Optional[datetime] = None
    
    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Convert user to dictionary"""
        data = {
            "id": self.id,
            "email": self.email,
            "status": self.status.value,
            "tier": self.tier,
            "roles": self.roles,
            "email_verified": self.email_verified,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "profile": {
                "first_name": self.profile.first_name,
                "last_name": self.profile.last_name,
                "company": self.profile.company,
                "job_title": self.profile.job_title,
                "timezone": self.profile.timezone,
            },
            "account_type": self.account_type.value,
            "oauth_providers": list(self.oauth_providers.keys()),
            "mfa_enabled": self.mfa_enabled,
        }
        
        if include_sensitive:
            data["hashed_password"] = self.hashed_password is not None
            data["login_attempts"] = self.login_attempts
            data["locked_until"] = self.locked_until.isoformat() if self.locked_until else None
        
        return data


@dataclass
class PasswordResetToken:
    """Password reset token"""
    token: str
    user_id: str
    expires_at: datetime
    used: bool = False


@dataclass
class EmailVerificationToken:
    """Email verification token"""
    token: str
    user_id: str
    email: str
    expires_at: datetime
    used: bool = False


class PasswordValidator:
    """Password validation with security requirements"""
    
    MIN_LENGTH = 8
    MAX_LENGTH = 128
    
    @classmethod
    def validate(cls, password: str) -> tuple[bool, List[str]]:
        """
        Validate password strength.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        if len(password) < cls.MIN_LENGTH:
            errors.append(f"Password must be at least {cls.MIN_LENGTH} characters")
        
        if len(password) > cls.MAX_LENGTH:
            errors.append(f"Password must not exceed {cls.MAX_LENGTH} characters")
        
        if not re.search(r'[A-Z]', password):
            errors.append("Password must contain at least one uppercase letter")
        
        if not re.search(r'[a-z]', password):
            errors.append("Password must contain at least one lowercase letter")
        
        if not re.search(r'\d', password):
            errors.append("Password must contain at least one digit")
        
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            errors.append("Password must contain at least one special character")
        
        # Check for common passwords (simplified - use a real list in production)
        common_passwords = ['password', '123456', 'qwerty', 'admin']
        if password.lower() in common_passwords:
            errors.append("Password is too common")
        
        return len(errors) == 0, errors


class UserManager:
    """
    User management for TradeOS.
    
    Handles:
    - User registration with email verification
    - User login/logout with rate limiting
    - Password reset flow
    - Profile management
    - OAuth user linking
    """
    
    def __init__(
        self,
        jwt_handler: Optional[JWTHandler] = None,
        email_service: Optional[EmailService] = None,
        security_config: Optional[SecurityConfig] = None
    ):
        self.jwt_handler = jwt_handler or get_jwt_handler()
        self.email_service = email_service or get_email_service()
        self.security_config = security_config or get_config().security
        
        # In-memory storage (replace with database in production)
        self._users: Dict[str, User] = {}
        self._users_by_email: Dict[str, str] = {}  # email -> user_id
        self._password_reset_tokens: Dict[str, PasswordResetToken] = {}
        self._email_verification_tokens: Dict[str, EmailVerificationToken] = {}
    
    def _hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt(rounds=self.security_config.bcrypt_rounds)
        return bcrypt.hashpw(password.encode(), salt).decode()
    
    def _verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode(), hashed.encode())
    
    def _generate_id(self) -> str:
        """Generate unique user ID"""
        return f"usr_{secrets.token_urlsafe(16)}"
    
    def _generate_token(self) -> str:
        """Generate secure random token"""
        return secrets.token_urlsafe(32)
    
    def _is_account_locked(self, user: User) -> bool:
        """Check if user account is locked"""
        if user.locked_until and user.locked_until > datetime.now(timezone.utc):
            return True
        return False
    
    def _record_failed_login(self, user: User) -> None:
        """Record failed login attempt"""
        user.login_attempts += 1
        
        if user.login_attempts >= self.security_config.max_login_attempts:
            user.locked_until = datetime.now(timezone.utc) + timedelta(
                minutes=self.security_config.login_lockout_minutes
            )
    
    def _reset_login_attempts(self, user: User) -> None:
        """Reset login attempts after successful login"""
        user.login_attempts = 0
        user.locked_until = None
    
    async def register(
        self,
        email: str,
        password: str,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
        background_tasks: Optional[BackgroundTasks] = None
    ) -> tuple[User, EmailVerificationToken]:
        """
        Register a new user.
        
        Args:
            email: User email address
            password: User password
            first_name: Optional first name
            last_name: Optional last name
            background_tasks: FastAPI background tasks for email
            
        Returns:
            Tuple of (User, EmailVerificationToken)
            
        Raises:
            HTTPException: If registration fails
        """
        # Normalize email
        email = email.lower().strip()
        
        # Check if email already exists
        if email in self._users_by_email:
            raise HTTPException(
                status_code=409,
                detail="Email already registered"
            )
        
        # Validate password
        is_valid, errors = PasswordValidator.validate(password)
        if not is_valid:
            raise HTTPException(
                status_code=400,
                detail={"password_errors": errors}
            )
        
        # Create user
        user_id = self._generate_id()
        now = datetime.now(timezone.utc)
        
        user = User(
            id=user_id,
            email=email,
            hashed_password=self._hash_password(password),
            status=UserStatus.PENDING,
            tier="free",
            roles=["trader"],
            email_verified=False,
            created_at=now,
            updated_at=now,
            profile=UserProfile(
                first_name=first_name,
                last_name=last_name
            ),
            password_changed_at=now
        )
        
        # Store user
        self._users[user_id] = user
        self._users_by_email[email] = user_id
        
        # Create email verification token
        verification_token = EmailVerificationToken(
            token=self._generate_token(),
            user_id=user_id,
            email=email,
            expires_at=now + timedelta(hours=24)
        )
        self._email_verification_tokens[verification_token.token] = verification_token
        
        # Send verification email
        if background_tasks and self.email_service:
            background_tasks.add_task(
                self.email_service.send_verification_email,
                email=email,
                name=first_name or email,
                verification_token=verification_token.token
            )
        
        return user, verification_token
    
    async def verify_email(self, token: str) -> User:
        """
        Verify user email address.
        
        Args:
            token: Email verification token
            
        Returns:
            Updated user
        """
        verification = self._email_verification_tokens.get(token)
        
        if not verification:
            raise HTTPException(status_code=400, detail="Invalid verification token")
        
        if verification.used:
            raise HTTPException(status_code=400, detail="Token already used")
        
        if verification.expires_at < datetime.now(timezone.utc):
            raise HTTPException(status_code=400, detail="Token expired")
        
        user = self._users.get(verification.user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Update user
        user.email_verified = True
        user.status = UserStatus.ACTIVE
        user.updated_at = datetime.now(timezone.utc)
        
        # Mark token as used
        verification.used = True
        
        return user
    
    async def login(
        self,
        email: str,
        password: str,
        ip_address: Optional[str] = None
    ) -> tuple[User, TokenPair]:
        """
        Authenticate user and create tokens.
        
        Args:
            email: User email
            password: User password
            ip_address: Optional IP for logging
            
        Returns:
            Tuple of (User, TokenPair)
        """
        email = email.lower().strip()
        user_id = self._users_by_email.get(email)
        
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        user = self._users[user_id]
        
        # Check account status
        if user.status == UserStatus.SUSPENDED:
            raise HTTPException(status_code=403, detail="Account suspended")
        
        if user.status == UserStatus.DEACTIVATED:
            raise HTTPException(status_code=403, detail="Account deactivated")
        
        # Check if locked
        if self._is_account_locked(user):
            raise HTTPException(
                status_code=429,
                detail=f"Account locked. Try again after {user.locked_until}"
            )
        
        # Verify password
        if not user.hashed_password or not self._verify_password(password, user.hashed_password):
            self._record_failed_login(user)
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Check email verification
        if not user.email_verified:
            raise HTTPException(
                status_code=403,
                detail="Email not verified. Please check your email."
            )
        
        # Reset login attempts
        self._reset_login_attempts(user)
        
        # Update last login
        user.last_login = datetime.now(timezone.utc)
        
        # Create tokens
        tokens = self.jwt_handler.create_token_pair(
            user_id=user.id,
            email=user.email,
            tier=user.tier,
            roles=user.roles
        )
        
        return user, tokens
    
    async def logout(self, user_id: str, token: str) -> bool:
        """
        Logout user and invalidate token.
        
        Args:
            user_id: User ID
            token: Access token to invalidate
            
        Returns:
            True if successful
        """
        # Blacklist token
        self.jwt_handler.blacklist_token(token)
        
        # Revoke all refresh tokens for user
        self.jwt_handler.revoke_all_user_tokens(user_id)
        
        return True
    
    async def logout_all_devices(self, user_id: str) -> bool:
        """Logout user from all devices"""
        self.jwt_handler.revoke_all_user_tokens(user_id)
        return True
    
    async def request_password_reset(
        self,
        email: str,
        background_tasks: Optional[BackgroundTasks] = None
    ) -> Optional[PasswordResetToken]:
        """
        Request password reset.
        
        Args:
            email: User email
            background_tasks: FastAPI background tasks
            
        Returns:
            PasswordResetToken or None if user not found
        """
        email = email.lower().strip()
        user_id = self._users_by_email.get(email)
        
        if not user_id:
            # Don't reveal if email exists
            return None
        
        user = self._users[user_id]
        
        # Create reset token
        token = PasswordResetToken(
            token=self._generate_token(),
            user_id=user_id,
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1)
        )
        self._password_reset_tokens[token.token] = token
        
        # Send reset email
        if background_tasks and self.email_service:
            background_tasks.add_task(
                self.email_service.send_password_reset_email,
                email=email,
                name=user.profile.first_name or email,
                reset_token=token.token
            )
        
        return token
    
    async def reset_password(
        self,
        token: str,
        new_password: str
    ) -> User:
        """
        Reset password using token.
        
        Args:
            token: Password reset token
            new_password: New password
            
        Returns:
            Updated user
        """
        reset_token = self._password_reset_tokens.get(token)
        
        if not reset_token:
            raise HTTPException(status_code=400, detail="Invalid reset token")
        
        if reset_token.used:
            raise HTTPException(status_code=400, detail="Token already used")
        
        if reset_token.expires_at < datetime.now(timezone.utc):
            raise HTTPException(status_code=400, detail="Token expired")
        
        # Validate new password
        is_valid, errors = PasswordValidator.validate(new_password)
        if not is_valid:
            raise HTTPException(
                status_code=400,
                detail={"password_errors": errors}
            )
        
        user = self._users.get(reset_token.user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Update password
        user.hashed_password = self._hash_password(new_password)
        user.password_changed_at = datetime.now(timezone.utc)
        user.updated_at = datetime.now(timezone.utc)
        
        # Mark token as used
        reset_token.used = True
        
        # Revoke all tokens (force re-login)
        self.jwt_handler.revoke_all_user_tokens(user.id)
        
        return user
    
    async def change_password(
        self,
        user_id: str,
        current_password: str,
        new_password: str
    ) -> User:
        """
        Change password (authenticated user).
        
        Args:
            user_id: User ID
            current_password: Current password
            new_password: New password
            
        Returns:
            Updated user
        """
        user = self._users.get(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        if not user.hashed_password:
            raise HTTPException(
                status_code=400,
                detail="OAuth users cannot change password"
            )
        
        # Verify current password
        if not self._verify_password(current_password, user.hashed_password):
            raise HTTPException(status_code=401, detail="Current password is incorrect")
        
        # Validate new password
        is_valid, errors = PasswordValidator.validate(new_password)
        if not is_valid:
            raise HTTPException(
                status_code=400,
                detail={"password_errors": errors}
            )
        
        # Update password
        user.hashed_password = self._hash_password(new_password)
        user.password_changed_at = datetime.now(timezone.utc)
        user.updated_at = datetime.now(timezone.utc)
        
        return user
    
    async def update_profile(
        self,
        user_id: str,
        **updates
    ) -> User:
        """
        Update user profile.
        
        Args:
            user_id: User ID
            **updates: Profile fields to update
            
        Returns:
            Updated user
        """
        user = self._users.get(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Update profile fields
        profile_fields = ['first_name', 'last_name', 'company', 'job_title', 'phone', 'timezone']
        for field in profile_fields:
            if field in updates:
                setattr(user.profile, field, updates[field])
        
        # Update preferences
        if 'preferences' in updates:
            user.profile.preferences.update(updates['preferences'])
        
        user.updated_at = datetime.now(timezone.utc)
        
        return user
    
    async def link_oauth_account(
        self,
        user_id: str,
        provider: str,
        provider_user_id: str
    ) -> User:
        """
        Link OAuth account to existing user.
        
        Args:
            user_id: User ID
            provider: OAuth provider (google, github)
            provider_user_id: Provider's user ID
            
        Returns:
            Updated user
        """
        user = self._users.get(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        user.oauth_providers[provider] = provider_user_id
        user.updated_at = datetime.now(timezone.utc)
        
        return user
    
    async def create_oauth_user(
        self,
        email: str,
        provider: str,
        provider_user_id: str,
        name: Optional[str] = None,
        avatar_url: Optional[str] = None
    ) -> User:
        """
        Create a new user from OAuth login.
        
        Args:
            email: User email
            provider: OAuth provider
            provider_user_id: Provider's user ID
            name: User's display name
            avatar_url: Profile picture URL
            
        Returns:
            Created user
        """
        email = email.lower().strip()
        
        # Check if user exists
        if email in self._users_by_email:
            user_id = self._users_by_email[email]
            user = self._users[user_id]
            
            # Link OAuth if not already linked
            if provider not in user.oauth_providers:
                await self.link_oauth_account(user_id, provider, provider_user_id)
            
            return user
        
        # Create new user
        user_id = self._generate_id()
        now = datetime.now(timezone.utc)
        
        # Parse name
        first_name = None
        last_name = None
        if name:
            parts = name.split(maxsplit=1)
            first_name = parts[0]
            last_name = parts[1] if len(parts) > 1 else None
        
        user = User(
            id=user_id,
            email=email,
            hashed_password=None,  # OAuth user
            status=UserStatus.ACTIVE,
            tier="free",
            roles=["trader"],
            email_verified=True,  # OAuth emails are pre-verified
            created_at=now,
            updated_at=now,
            last_login=now,
            profile=UserProfile(
                first_name=first_name,
                last_name=last_name
            ),
            oauth_providers={provider: provider_user_id}
        )
        
        self._users[user_id] = user
        self._users_by_email[email] = user_id
        
        return user
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return self._users.get(user_id)
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        email = email.lower().strip()
        user_id = self._users_by_email.get(email)
        if user_id:
            return self._users.get(user_id)
        return None
    
    def list_users(
        self,
        tier: Optional[str] = None,
        status: Optional[UserStatus] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[User]:
        """List users with optional filtering"""
        users = list(self._users.values())
        
        if tier:
            users = [u for u in users if u.tier == tier]
        
        if status:
            users = [u for u in users if u.status == status]
        
        users.sort(key=lambda u: u.created_at, reverse=True)
        
        return users[offset:offset + limit]


# Global user manager instance
_user_manager: Optional[UserManager] = None


def get_user_manager() -> UserManager:
    """Get or create global user manager"""
    global _user_manager
    if _user_manager is None:
        _user_manager = UserManager()
    return _user_manager
