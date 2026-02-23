"""
TradeOS Configuration Settings
Uses pydantic-settings for environment-based configuration management.
"""

from functools import lru_cache
from typing import List, Optional
from pydantic import Field, PostgresDsn, RedisDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    model_config = SettingsConfigDict(env_prefix="DB_")
    
    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, description="Database port")
    name: str = Field(default="tradeos", description="Database name")
    user: str = Field(default="tradeos_user", description="Database user")
    password: str = Field(default="", description="Database password")
    
    # Connection pool settings
    pool_size: int = Field(default=20, description="Connection pool size")
    max_overflow: int = Field(default=10, description="Max overflow connections")
    pool_timeout: int = Field(default=30, description="Pool timeout in seconds")
    pool_recycle: int = Field(default=1800, description="Connection recycle time")
    echo: bool = Field(default=False, description="Echo SQL statements")
    
    @property
    def async_url(self) -> str:
        """Build async PostgreSQL URL."""
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"
    
    @property
    def sync_url(self) -> str:
        """Build sync PostgreSQL URL for migrations."""
        return f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class RedisSettings(BaseSettings):
    """Redis configuration settings."""
    model_config = SettingsConfigDict(env_prefix="REDIS_")
    
    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, description="Redis port")
    db: int = Field(default=0, description="Redis database number")
    password: Optional[str] = Field(default=None, description="Redis password")
    
    # Connection settings
    socket_timeout: int = Field(default=5, description="Socket timeout")
    socket_connect_timeout: int = Field(default=5, description="Socket connect timeout")
    max_connections: int = Field(default=50, description="Max connections in pool")
    
    @property
    def url(self) -> str:
        """Build Redis URL."""
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"


class JWTSettings(BaseSettings):
    """JWT authentication settings."""
    model_config = SettingsConfigDict(env_prefix="JWT_")
    
    secret_key: str = Field(default="your-super-secret-key-change-in-production", description="JWT secret key")
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(default=30, description="Access token expiry")
    refresh_token_expire_days: int = Field(default=7, description="Refresh token expiry")
    
    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v: str) -> str:
        if len(v) < 32:
            raise ValueError("JWT secret key must be at least 32 characters")
        return v


class ExchangeAPISettings(BaseSettings):
    """Exchange API configuration settings."""
    model_config = SettingsConfigDict(env_prefix="EXCHANGE_")
    
    # Binance
    binance_api_key: Optional[str] = Field(default=None, description="Binance API key")
    binance_secret_key: Optional[str] = Field(default=None, description="Binance secret key")
    binance_testnet: bool = Field(default=True, description="Use Binance testnet")
    
    # Coinbase
    coinbase_api_key: Optional[str] = Field(default=None, description="Coinbase API key")
    coinbase_secret_key: Optional[str] = Field(default=None, description="Coinbase secret key")
    coinbase_passphrase: Optional[str] = Field(default=None, description="Coinbase passphrase")
    
    # Kraken
    kraken_api_key: Optional[str] = Field(default=None, description="Kraken API key")
    kraken_secret_key: Optional[str] = Field(default=None, description="Kraken secret key")
    
    # Rate limiting
    rate_limit_requests_per_second: float = Field(default=10.0, description="Rate limit per second")
    rate_limit_burst: int = Field(default=20, description="Rate limit burst")


class SecuritySettings(BaseSettings):
    """Security-related settings."""
    model_config = SettingsConfigDict(env_prefix="SECURITY_")
    
    password_min_length: int = Field(default=8, description="Minimum password length")
    password_hash_algorithm: str = Field(default="bcrypt", description="Password hash algorithm")
    bcrypt_rounds: int = Field(default=12, description="Bcrypt rounds")
    api_key_prefix: str = Field(default="trd_", description="API key prefix")
    max_login_attempts: int = Field(default=5, description="Max login attempts before lockout")
    lockout_duration_minutes: int = Field(default=30, description="Account lockout duration")


class LoggingSettings(BaseSettings):
    """Logging configuration settings."""
    model_config = SettingsConfigDict(env_prefix="LOG_")
    
    level: str = Field(default="INFO", description="Log level")
    format: str = Field(
        default="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        description="Log format"
    )
    json_format: bool = Field(default=True, description="Use JSON format for logs")
    file_path: Optional[str] = Field(default=None, description="Log file path")
    max_bytes: int = Field(default=10_485_760, description="Max log file size (10MB)")
    backup_count: int = Field(default=5, description="Number of backup files")


class RateLimitSettings(BaseSettings):
    """Rate limiting settings."""
    model_config = SettingsConfigDict(env_prefix="RATE_LIMIT_")
    
    enabled: bool = Field(default=True, description="Enable rate limiting")
    default_limit: int = Field(default=100, description="Default requests per window")
    default_window: int = Field(default=60, description="Default window in seconds")
    authenticated_limit: int = Field(default=1000, description="Limit for authenticated users")
    
    # Endpoint-specific limits
    auth_limit: int = Field(default=10, description="Auth endpoint limit")
    trade_limit: int = Field(default=50, description="Trade endpoint limit")
    market_data_limit: int = Field(default=200, description="Market data endpoint limit")


class StripeSettings(BaseSettings):
    """Stripe payment settings."""
    model_config = SettingsConfigDict(env_prefix="STRIPE_")
    
    secret_key: Optional[str] = Field(default=None, description="Stripe secret key")
    webhook_secret: Optional[str] = Field(default=None, description="Stripe webhook secret")
    publishable_key: Optional[str] = Field(default=None, description="Stripe publishable key")
    
    # Subscription tiers
    free_tier_price_id: Optional[str] = Field(default=None, description="Free tier price ID")
    pro_tier_price_id: Optional[str] = Field(default=None, description="Pro tier price ID")
    enterprise_tier_price_id: Optional[str] = Field(default=None, description="Enterprise tier price ID")


class ApplicationSettings(BaseSettings):
    """Main application settings aggregating all sub-settings."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # Application info
    app_name: str = Field(default="TradeOS", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    environment: str = Field(default="development", description="Environment (development/staging/production)")
    
    # Server settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    workers: int = Field(default=1, description="Number of worker processes")
    
    # CORS settings
    cors_origins: List[str] = Field(default=["*"], description="Allowed CORS origins")
    cors_allow_credentials: bool = Field(default=True, description="Allow credentials in CORS")
    cors_allow_methods: List[str] = Field(default=["*"], description="Allowed CORS methods")
    cors_allow_headers: List[str] = Field(default=["*"], description="Allowed CORS headers")
    
    # Sub-settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    jwt: JWTSettings = Field(default_factory=JWTSettings)
    exchange: ExchangeAPISettings = Field(default_factory=ExchangeAPISettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    rate_limit: RateLimitSettings = Field(default_factory=RateLimitSettings)
    stripe: StripeSettings = Field(default_factory=StripeSettings)
    
    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() == "development"


@lru_cache()
def get_settings() -> ApplicationSettings:
    """
    Get cached application settings.
    
    Returns:
        ApplicationSettings: The application settings instance.
    """
    return ApplicationSettings()


# Export settings instance
settings = get_settings()
