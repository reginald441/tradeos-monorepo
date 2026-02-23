"""
TradeOS Middleware Package
"""

from middleware.logging import (
    ErrorLoggingMiddleware,
    RequestLoggingMiddleware,
    StructuredLoggingMiddleware,
    create_logging_middleware,
)
from middleware.rate_limit import (
    AUTHENTICATED_RATE_LIMIT,
    AUTH_RATE_LIMIT,
    MARKET_DATA_RATE_LIMIT,
    TRADE_RATE_LIMIT,
    RateLimitConfig,
    RateLimitMiddleware,
    RateLimitStrategy,
    RateLimiter,
    create_rate_limit_middleware,
    get_rate_limiter,
)

__all__ = [
    # Logging
    "RequestLoggingMiddleware",
    "StructuredLoggingMiddleware",
    "ErrorLoggingMiddleware",
    "create_logging_middleware",
    # Rate Limiting
    "RateLimitMiddleware",
    "RateLimiter",
    "RateLimitConfig",
    "RateLimitStrategy",
    "create_rate_limit_middleware",
    "get_rate_limiter",
    "AUTH_RATE_LIMIT",
    "TRADE_RATE_LIMIT",
    "MARKET_DATA_RATE_LIMIT",
    "AUTHENTICATED_RATE_LIMIT",
]
