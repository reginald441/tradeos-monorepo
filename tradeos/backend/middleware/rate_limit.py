"""
TradeOS Rate Limiting Middleware
Rate limiting middleware using Redis for distributed rate limiting.
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Union

import redis.asyncio as redis
from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from config.settings import settings

logger = logging.getLogger(__name__)


class RateLimitStrategy(Enum):
    """Rate limiting strategies."""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    requests: int
    window: int  # seconds
    strategy: RateLimitStrategy = RateLimitStrategy.FIXED_WINDOW
    key_prefix: str = "ratelimit"


class RateLimiter:
    """
    Redis-based rate limiter with multiple strategies.
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        """
        Initialize the rate limiter.
        
        Args:
            redis_client: Optional Redis client. If not provided, will create one.
        """
        self._redis = redis_client
        self._local_cache = {}  # Fallback for when Redis is unavailable
    
    async def _get_redis(self) -> Optional[redis.Redis]:
        """Get or create Redis connection."""
        if self._redis is None:
            try:
                self._redis = redis.from_url(
                    settings.redis.url,
                    decode_responses=True,
                    socket_connect_timeout=2,
                    socket_timeout=2
                )
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")
                return None
        return self._redis
    
    def _get_key(self, identifier: str, config: RateLimitConfig) -> str:
        """Generate rate limit key."""
        return f"{config.key_prefix}:{identifier}"
    
    async def is_allowed(
        self,
        identifier: str,
        config: RateLimitConfig
    ) -> tuple[bool, dict]:
        """
        Check if a request is allowed under rate limit.
        
        Args:
            identifier: Unique identifier (IP, user ID, API key).
            config: Rate limit configuration.
        
        Returns:
            tuple: (is_allowed, rate_limit_info)
        """
        key = self._get_key(identifier, config)
        
        redis_client = await self._get_redis()
        
        if redis_client:
            return await self._check_redis(redis_client, key, identifier, config)
        else:
            return self._check_local(key, identifier, config)
    
    async def _check_redis(
        self,
        redis_client: redis.Redis,
        key: str,
        identifier: str,
        config: RateLimitConfig
    ) -> tuple[bool, dict]:
        """Check rate limit using Redis."""
        now = time.time()
        
        if config.strategy == RateLimitStrategy.FIXED_WINDOW:
            return await self._fixed_window_redis(redis_client, key, now, config)
        elif config.strategy == RateLimitStrategy.SLIDING_WINDOW:
            return await self._sliding_window_redis(redis_client, key, now, config)
        else:
            return await self._token_bucket_redis(redis_client, key, now, config)
    
    async def _fixed_window_redis(
        self,
        redis_client: redis.Redis,
        key: str,
        now: float,
        config: RateLimitConfig
    ) -> tuple[bool, dict]:
        """Fixed window rate limiting with Redis."""
        window_start = int(now // config.window) * config.window
        window_key = f"{key}:{window_start}"
        
        pipe = redis_client.pipeline()
        pipe.incr(window_key)
        pipe.expire(window_key, config.window)
        results = await pipe.execute()
        
        current_count = results[0]
        is_allowed = current_count <= config.requests
        
        reset_time = window_start + config.window
        remaining = max(0, config.requests - current_count)
        
        return is_allowed, {
            "limit": config.requests,
            "remaining": remaining,
            "reset": int(reset_time),
            "window": config.window
        }
    
    async def _sliding_window_redis(
        self,
        redis_client: redis.Redis,
        key: str,
        now: float,
        config: RateLimitConfig
    ) -> tuple[bool, dict]:
        """Sliding window rate limiting with Redis."""
        window_start = now - config.window
        
        pipe = redis_client.pipeline()
        pipe.zremrangebyscore(key, 0, window_start)
        pipe.zcard(key)
        pipe.zadd(key, {str(now): now})
        pipe.expire(key, config.window)
        results = await pipe.execute()
        
        current_count = results[1] + 1  # +1 for the current request
        is_allowed = current_count <= config.requests
        
        remaining = max(0, config.requests - current_count)
        reset_time = now + config.window
        
        return is_allowed, {
            "limit": config.requests,
            "remaining": remaining,
            "reset": int(reset_time),
            "window": config.window
        }
    
    async def _token_bucket_redis(
        self,
        redis_client: redis.Redis,
        key: str,
        now: float,
        config: RateLimitConfig
    ) -> tuple[bool, dict]:
        """Token bucket rate limiting with Redis."""
        bucket_key = f"{key}:tokens"
        last_update_key = f"{key}:last_update"
        
        # Lua script for atomic token bucket operation
        lua_script = """
            local bucket_key = KEYS[1]
            local last_update_key = KEYS[2]
            local rate = tonumber(ARGV[1])
            local capacity = tonumber(ARGV[2])
            local now = tonumber(ARGV[3])
            
            local tokens = redis.call('GET', bucket_key)
            local last_update = redis.call('GET', last_update_key)
            
            if tokens == false then
                tokens = capacity
                last_update = now
            else
                tokens = tonumber(tokens)
                last_update = tonumber(last_update)
            end
            
            -- Add tokens based on time passed
            local time_passed = now - last_update
            tokens = math.min(capacity, tokens + (time_passed * rate))
            
            if tokens >= 1 then
                tokens = tokens - 1
                redis.call('SET', bucket_key, tokens)
                redis.call('SET', last_update_key, now)
                redis.call('EXPIRE', bucket_key, 3600)
                redis.call('EXPIRE', last_update_key, 3600)
                return {1, math.floor(tokens)}
            else
                redis.call('SET', bucket_key, tokens)
                redis.call('SET', last_update_key, now)
                redis.call('EXPIRE', bucket_key, 3600)
                redis.call('EXPIRE', last_update_key, 3600)
                return {0, math.floor(tokens)}
            """
        
        rate = config.requests / config.window
        
        result = await redis_client.eval(
            lua_script,
            2,
            bucket_key,
            last_update_key,
            rate,
            config.requests,
            now
        )
        
        is_allowed = result[0] == 1
        remaining = result[1]
        reset_time = now + (1 / rate if rate > 0 else config.window)
        
        return is_allowed, {
            "limit": config.requests,
            "remaining": remaining,
            "reset": int(reset_time),
            "window": config.window
        }
    
    def _check_local(
        self,
        key: str,
        identifier: str,
        config: RateLimitConfig
    ) -> tuple[bool, dict]:
        """Fallback local rate limiting (not distributed)."""
        now = time.time()
        window_start = int(now // config.window) * config.window
        window_key = f"{key}:{window_start}"
        
        # Clean old entries
        self._cleanup_local_cache(now, config.window)
        
        # Increment counter
        if window_key not in self._local_cache:
            self._local_cache[window_key] = {"count": 0, "expires": window_start + config.window}
        
        self._local_cache[window_key]["count"] += 1
        current_count = self._local_cache[window_key]["count"]
        
        is_allowed = current_count <= config.requests
        remaining = max(0, config.requests - current_count)
        reset_time = window_start + config.window
        
        return is_allowed, {
            "limit": config.requests,
            "remaining": remaining,
            "reset": int(reset_time),
            "window": config.window
        }
    
    def _cleanup_local_cache(self, now: float, window: int):
        """Clean expired entries from local cache."""
        expired_keys = [
            k for k, v in self._local_cache.items()
            if v.get("expires", 0) < now
        ]
        for k in expired_keys:
            del self._local_cache[k]


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get or create global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware for FastAPI.
    
    Applies rate limits based on IP address, user ID, or API key.
    """
    
    def __init__(
        self,
        app: ASGIApp,
        default_config: Optional[RateLimitConfig] = None,
        path_configs: Optional[dict] = None,
        skip_paths: Optional[list] = None
    ):
        """
        Initialize rate limit middleware.
        
        Args:
            app: The ASGI application.
            default_config: Default rate limit configuration.
            path_configs: Path-specific rate limit configurations.
            skip_paths: Paths to skip rate limiting.
        """
        super().__init__(app)
        self.default_config = default_config or RateLimitConfig(
            requests=settings.rate_limit.default_limit,
            window=settings.rate_limit.default_window
        )
        self.path_configs = path_configs or {}
        self.skip_paths = skip_paths or [
            "/health",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json"
        ]
        self.rate_limiter = get_rate_limiter()
    
    def _get_client_identifier(self, request: Request) -> str:
        """Get client identifier for rate limiting."""
        # Try to get user ID from request state
        if hasattr(request.state, "user_id"):
            return f"user:{request.state.user_id}"
        
        # Try to get API key
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"apikey:{api_key[:16]}"
        
        # Fall back to IP address
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return f"ip:{forwarded.split(',')[0].strip()}"
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return f"ip:{real_ip}"
        
        if request.client:
            return f"ip:{request.client.host}"
        
        return "ip:unknown"
    
    def _get_config_for_path(self, path: str) -> Optional[RateLimitConfig]:
        """Get rate limit config for a specific path."""
        # Check for exact match
        if path in self.path_configs:
            return self.path_configs[path]
        
        # Check for prefix match
        for prefix, config in self.path_configs.items():
            if path.startswith(prefix):
                return config
        
        return self.default_config
    
    def _should_skip(self, path: str) -> bool:
        """Check if path should skip rate limiting."""
        for skip_path in self.skip_paths:
            if path.startswith(skip_path):
                return True
        return False
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Apply rate limiting to the request.
        
        Args:
            request: The incoming request.
            call_next: The next middleware/endpoint.
        
        Returns:
            Response: The response from the next handler.
        """
        path = request.url.path
        
        # Skip rate limiting for certain paths
        if self._should_skip(path):
            return await call_next(request)
        
        # Skip if rate limiting is disabled
        if not settings.rate_limit.enabled:
            return await call_next(request)
        
        # Get rate limit config
        config = self._get_config_for_path(path)
        if config is None:
            return await call_next(request)
        
        # Get client identifier
        identifier = self._get_client_identifier(request)
        
        # Check rate limit
        is_allowed, rate_info = await self.rate_limiter.is_allowed(identifier, config)
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(rate_info["limit"])
        response.headers["X-RateLimit-Remaining"] = str(rate_info["remaining"])
        response.headers["X-RateLimit-Reset"] = str(rate_info["reset"])
        response.headers["X-RateLimit-Window"] = str(rate_info["window"])
        
        # Reject if rate limit exceeded
        if not is_allowed:
            retry_after = rate_info["reset"] - int(time.time())
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
                headers={
                    "Retry-After": str(max(1, retry_after)),
                    "X-RateLimit-Limit": str(rate_info["limit"]),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(rate_info["reset"])
                }
            )
        
        return response


def create_rate_limit_middleware(
    requests: int = None,
    window: int = None,
    path_configs: Optional[dict] = None
) -> type:
    """
    Create a configured rate limit middleware class.
    
    Args:
        requests: Default number of requests allowed.
        window: Default time window in seconds.
        path_configs: Path-specific configurations.
    
    Returns:
        Configured middleware class.
    """
    default_config = RateLimitConfig(
        requests=requests or settings.rate_limit.default_limit,
        window=window or settings.rate_limit.default_window
    )
    
    class ConfiguredRateLimitMiddleware(RateLimitMiddleware):
        def __init__(self, app: ASGIApp):
            super().__init__(app, default_config, path_configs)
    
    return ConfiguredRateLimitMiddleware


# Pre-configured rate limit configs for common use cases
AUTH_RATE_LIMIT = RateLimitConfig(
    requests=settings.rate_limit.auth_limit,
    window=60,
    key_prefix="ratelimit:auth"
)

TRADE_RATE_LIMIT = RateLimitConfig(
    requests=settings.rate_limit.trade_limit,
    window=60,
    key_prefix="ratelimit:trade"
)

MARKET_DATA_RATE_LIMIT = RateLimitConfig(
    requests=settings.rate_limit.market_data_limit,
    window=60,
    key_prefix="ratelimit:market"
)

AUTHENTICATED_RATE_LIMIT = RateLimitConfig(
    requests=settings.rate_limit.authenticated_limit,
    window=60,
    key_prefix="ratelimit:auth_user"
)


__all__ = [
    "RateLimitStrategy",
    "RateLimitConfig",
    "RateLimiter",
    "RateLimitMiddleware",
    "create_rate_limit_middleware",
    "get_rate_limiter",
    "AUTH_RATE_LIMIT",
    "TRADE_RATE_LIMIT",
    "MARKET_DATA_RATE_LIMIT",
    "AUTHENTICATED_RATE_LIMIT",
]
