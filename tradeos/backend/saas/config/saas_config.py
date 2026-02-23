"""
TradeOS SaaS Configuration
==========================
Centralized configuration for all SaaS components.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class Environment(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class SubscriptionTier(Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


@dataclass
class JWTConfig:
    """JWT Authentication Configuration"""
    secret_key: str = field(default_factory=lambda: os.getenv("JWT_SECRET_KEY", ""))
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    token_issuer: str = "tradeos"
    token_audience: str = "tradeos-api"


@dataclass
class OAuthConfig:
    """OAuth2 Provider Configuration"""
    # Google OAuth
    google_client_id: str = field(default_factory=lambda: os.getenv("GOOGLE_CLIENT_ID", ""))
    google_client_secret: str = field(default_factory=lambda: os.getenv("GOOGLE_CLIENT_SECRET", ""))
    google_redirect_uri: str = field(default_factory=lambda: os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/google/callback"))
    
    # GitHub OAuth
    github_client_id: str = field(default_factory=lambda: os.getenv("GITHUB_CLIENT_ID", ""))
    github_client_secret: str = field(default_factory=lambda: os.getenv("GITHUB_CLIENT_SECRET", ""))
    github_redirect_uri: str = field(default_factory=lambda: os.getenv("GITHUB_REDIRECT_URI", "http://localhost:8000/auth/github/callback"))


@dataclass
class StripeConfig:
    """Stripe Payment Configuration"""
    secret_key: str = field(default_factory=lambda: os.getenv("STRIPE_SECRET_KEY", ""))
    publishable_key: str = field(default_factory=lambda: os.getenv("STRIPE_PUBLISHABLE_KEY", ""))
    webhook_secret: str = field(default_factory=lambda: os.getenv("STRIPE_WEBHOOK_SECRET", ""))
    
    # Price IDs for subscription tiers
    price_ids: Dict[str, str] = field(default_factory=lambda: {
        "pro_monthly": os.getenv("STRIPE_PRO_MONTHLY_PRICE_ID", ""),
        "pro_yearly": os.getenv("STRIPE_PRO_YEARLY_PRICE_ID", ""),
        "enterprise_monthly": os.getenv("STRIPE_ENTERPRISE_MONTHLY_PRICE_ID", ""),
        "enterprise_yearly": os.getenv("STRIPE_ENTERPRISE_YEARLY_PRICE_ID", ""),
    })


@dataclass
class EmailConfig:
    """Email Service Configuration"""
    smtp_host: str = field(default_factory=lambda: os.getenv("SMTP_HOST", "smtp.sendgrid.net"))
    smtp_port: int = field(default_factory=lambda: int(os.getenv("SMTP_PORT", "587")))
    smtp_username: str = field(default_factory=lambda: os.getenv("SMTP_USERNAME", "apikey"))
    smtp_password: str = field(default_factory=lambda: os.getenv("SMTP_PASSWORD", ""))
    from_email: str = field(default_factory=lambda: os.getenv("FROM_EMAIL", "noreply@tradeos.io"))
    from_name: str = "TradeOS"
    use_tls: bool = True


@dataclass
class DatabaseConfig:
    """Database Configuration"""
    url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", "postgresql://localhost/tradeos"))
    pool_size: int = 20
    max_overflow: int = 10
    pool_timeout: int = 30
    echo: bool = False


@dataclass
class RedisConfig:
    """Redis Configuration for caching and sessions"""
    host: str = field(default_factory=lambda: os.getenv("REDIS_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("REDIS_PORT", "6379")))
    db: int = field(default_factory=lambda: int(os.getenv("REDIS_DB", "0")))
    password: Optional[str] = field(default_factory=lambda: os.getenv("REDIS_PASSWORD"))
    ssl: bool = field(default_factory=lambda: os.getenv("REDIS_SSL", "false").lower() == "true")


@dataclass
class SecurityConfig:
    """Security Configuration"""
    bcrypt_rounds: int = 12
    password_min_length: int = 8
    max_login_attempts: int = 5
    login_lockout_minutes: int = 30
    api_key_length: int = 64
    api_key_prefix: str = "tr_"


@dataclass
class RateLimitConfig:
    """Rate Limiting Configuration"""
    # Tier-based rate limits (requests per minute)
    tier_limits: Dict[str, int] = field(default_factory=lambda: {
        "free": 60,
        "pro": 600,
        "enterprise": 6000,
    })
    
    # Specific endpoint limits
    endpoint_limits: Dict[str, Dict[str, int]] = field(default_factory=lambda: {
        "/api/v1/backtest": {"free": 10, "pro": 100, "enterprise": 1000},
        "/api/v1/strategies/execute": {"free": 5, "pro": 50, "enterprise": 500},
    })


@dataclass
class SubscriptionLimits:
    """Feature limits per subscription tier"""
    limits: Dict[str, Dict[str, any]] = field(default_factory=lambda: {
        "free": {
            "max_strategies": 1,
            "max_backtests_per_day": 10,
            "max_api_calls_per_day": 100,
            "strategy_executions_per_day": 50,
            "historical_data_days": 30,
            "support_level": "community",
            "features": ["basic_backtest", "paper_trading"],
        },
        "pro": {
            "max_strategies": 10,
            "max_backtests_per_day": -1,  # Unlimited
            "max_api_calls_per_day": 10000,
            "strategy_executions_per_day": 1000,
            "historical_data_days": 365,
            "support_level": "email",
            "features": [
                "basic_backtest", "advanced_backtest", "paper_trading",
                "live_trading", "webhooks", "api_access", "custom_indicators"
            ],
        },
        "enterprise": {
            "max_strategies": -1,  # Unlimited
            "max_backtests_per_day": -1,  # Unlimited
            "max_api_calls_per_day": -1,  # Unlimited
            "strategy_executions_per_day": -1,  # Unlimited
            "historical_data_days": -1,  # All available
            "support_level": "priority",
            "features": [
                "basic_backtest", "advanced_backtest", "paper_trading",
                "live_trading", "webhooks", "api_access", "custom_indicators",
                "white_label", "dedicated_support", "sla", "custom_development",
                "on_premise_option"
            ],
        },
    })


@dataclass
class SaaSConfig:
    """Main SaaS Configuration"""
    environment: Environment = field(
        default_factory=lambda: Environment(os.getenv("ENVIRONMENT", "development"))
    )
    app_name: str = "TradeOS"
    app_url: str = field(default_factory=lambda: os.getenv("APP_URL", "http://localhost:3000"))
    api_url: str = field(default_factory=lambda: os.getenv("API_URL", "http://localhost:8000"))
    
    jwt: JWTConfig = field(default_factory=JWTConfig)
    oauth: OAuthConfig = field(default_factory=OAuthConfig)
    stripe: StripeConfig = field(default_factory=StripeConfig)
    email: EmailConfig = field(default_factory=EmailConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    subscription_limits: SubscriptionLimits = field(default_factory=SubscriptionLimits)
    
    # Feature flags
    features: Dict[str, bool] = field(default_factory=lambda: {
        "oauth_login": True,
        "email_verification": True,
        "password_reset": True,
        "api_keys": True,
        "webhooks": True,
        "team_collaboration": False,  # Coming soon
        "affiliate_program": False,  # Coming soon
    })


# Global configuration instance
_config: Optional[SaaSConfig] = None


def get_config() -> SaaSConfig:
    """Get or create global SaaS configuration"""
    global _config
    if _config is None:
        _config = SaaSConfig()
    return _config


def reset_config():
    """Reset global configuration (useful for testing)"""
    global _config
    _config = None


# Convenience exports
def get_jwt_config() -> JWTConfig:
    return get_config().jwt


def get_oauth_config() -> OAuthConfig:
    return get_config().oauth


def get_stripe_config() -> StripeConfig:
    return get_config().stripe


def get_email_config() -> EmailConfig:
    return get_config().email


def get_subscription_limits(tier: str) -> Dict[str, any]:
    return get_config().subscription_limits.limits.get(tier, {})
