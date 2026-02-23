"""
TradeOS SaaS Layer
==================
Complete monetization and user management layer for TradeOS.

Modules:
- auth: JWT and OAuth2 authentication
- users: User registration, login, profile management
- rbac: Role-based access control
- subscriptions: Subscription tier management
- billing: Stripe integration
- api_keys: API key management
- usage: Usage tracking and quotas
- notifications: Email notifications
- webhooks: Webhook handlers
- admin: Admin panel APIs
- config: SaaS configuration
"""

__version__ = "1.0.0"

# Auth exports
from .auth.jwt_handler import (
    JWTHandler,
    TokenPayload,
    TokenPair,
    get_jwt_handler,
    get_current_user,
    get_current_active_user,
    require_tiers,
    require_roles,
)

from .auth.oauth import (
    OAuthManager,
    OAuthUserInfo,
    OAuthTokens,
    GoogleOAuthHandler,
    GitHubOAuthHandler,
    get_oauth_manager,
    oauth_login,
    oauth_callback,
)

# Users exports
from .users.user_manager import (
    UserManager,
    User,
    UserProfile,
    UserStatus,
    AccountType,
    PasswordValidator,
    get_user_manager,
)

# RBAC exports
from .rbac.permissions import (
    Permission,
    Role,
    PermissionChecker,
    require_permission,
    require_permissions,
    require_ownership_or_permission,
    get_permission_checker,
    can_access_feature,
    list_user_permissions,
)

# Subscriptions exports
from .subscriptions.tier_manager import (
    TierManager,
    Subscription,
    SubscriptionStatus,
    BillingInterval,
    Feature,
    Plan,
    get_tier_manager,
    require_feature,
    feature_required,
)

# Billing exports
from .billing.stripe_integration import (
    StripeBillingManager,
    StripeClient,
    SubscriptionDetails,
    Invoice,
    PaymentIntent,
    get_billing_manager,
)

# API Keys exports
from .api_keys.manager import (
    APIKeyManager,
    APIKey,
    APIKeyPermission,
    APIKeyStatus,
    validate_api_key,
    require_api_key_permission,
    APIKeyMiddleware,
    get_api_key_manager,
)

# Usage exports
from .usage.tracker import (
    UsageTracker,
    UsageType,
    UsageQuota,
    UsageRecord,
    track_usage,
    UsageTrackingMiddleware,
    get_usage_tracker,
)

# Notifications exports
from .notifications.email import (
    EmailService,
    EmailTemplate,
    EmailMessage,
    get_email_service,
)

# Webhooks exports
from .webhooks.handlers import (
    WebhookHandler,
    WebhookEvent,
    WebhookSource,
    WebhookEventType,
    UserWebhookManager,
    get_webhook_handler,
    dispatch_event,
)

# Admin exports
from .admin.admin_routes import (
    router as admin_router,
    get_admin_router,
    audit_logger,
)

# Config exports
from .config.saas_config import (
    SaaSConfig,
    JWTConfig,
    OAuthConfig,
    StripeConfig,
    EmailConfig,
    DatabaseConfig,
    RedisConfig,
    SecurityConfig,
    RateLimitConfig,
    SubscriptionLimits,
    SubscriptionTier,
    Environment,
    get_config,
    get_jwt_config,
    get_oauth_config,
    get_stripe_config,
    get_email_config,
    get_subscription_limits,
)


__all__ = [
    # Version
    "__version__",
    
    # Auth
    "JWTHandler",
    "TokenPayload",
    "TokenPair",
    "get_jwt_handler",
    "get_current_user",
    "get_current_active_user",
    "require_tiers",
    "require_roles",
    
    # OAuth
    "OAuthManager",
    "OAuthUserInfo",
    "OAuthTokens",
    "GoogleOAuthHandler",
    "GitHubOAuthHandler",
    "get_oauth_manager",
    "oauth_login",
    "oauth_callback",
    
    # Users
    "UserManager",
    "User",
    "UserProfile",
    "UserStatus",
    "AccountType",
    "PasswordValidator",
    "get_user_manager",
    
    # RBAC
    "Permission",
    "Role",
    "PermissionChecker",
    "require_permission",
    "require_permissions",
    "require_ownership_or_permission",
    "get_permission_checker",
    "can_access_feature",
    "list_user_permissions",
    
    # Subscriptions
    "TierManager",
    "Subscription",
    "SubscriptionStatus",
    "BillingInterval",
    "Feature",
    "Plan",
    "get_tier_manager",
    "require_feature",
    "feature_required",
    
    # Billing
    "StripeBillingManager",
    "StripeClient",
    "SubscriptionDetails",
    "Invoice",
    "PaymentIntent",
    "get_billing_manager",
    
    # API Keys
    "APIKeyManager",
    "APIKey",
    "APIKeyPermission",
    "APIKeyStatus",
    "validate_api_key",
    "require_api_key_permission",
    "APIKeyMiddleware",
    "get_api_key_manager",
    
    # Usage
    "UsageTracker",
    "UsageType",
    "UsageQuota",
    "UsageRecord",
    "track_usage",
    "UsageTrackingMiddleware",
    "get_usage_tracker",
    
    # Notifications
    "EmailService",
    "EmailTemplate",
    "EmailMessage",
    "get_email_service",
    
    # Webhooks
    "WebhookHandler",
    "WebhookEvent",
    "WebhookSource",
    "WebhookEventType",
    "UserWebhookManager",
    "get_webhook_handler",
    "dispatch_event",
    
    # Admin
    "admin_router",
    "get_admin_router",
    "audit_logger",
    
    # Config
    "SaaSConfig",
    "JWTConfig",
    "OAuthConfig",
    "StripeConfig",
    "EmailConfig",
    "DatabaseConfig",
    "RedisConfig",
    "SecurityConfig",
    "RateLimitConfig",
    "SubscriptionLimits",
    "SubscriptionTier",
    "Environment",
    "get_config",
    "get_jwt_config",
    "get_oauth_config",
    "get_stripe_config",
    "get_email_config",
    "get_subscription_limits",
]
