"""
TradeOS Redis Cache
High-performance caching layer for hot market data.

Features:
- Async Redis operations
- Connection pooling
- TTL-based expiration
- Pub/Sub for real-time updates
- Pipeline support for batch operations
- Serialization/deserialization
- Circuit breaker pattern
- Metrics and monitoring
"""

import asyncio
import json
import logging
import pickle
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Union, Callable, Set
from functools import wraps
from contextlib import asynccontextmanager

import aioredis
from aioredis import Redis, ConnectionPool

# Configure logging
logger = logging.getLogger(__name__)


class CacheError(Exception):
    """Base cache error."""
    pass


class CacheConnectionError(CacheError):
    """Cache connection error."""
    pass


class CacheTimeoutError(CacheError):
    """Cache operation timeout."""
    pass


@dataclass
class RedisConfig:
    """Redis connection configuration."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    username: Optional[str] = None
    ssl: bool = False
    ssl_ca_certs: Optional[str] = None
    
    # Connection pool settings
    max_connections: int = 50
    min_connections: int = 10
    connection_timeout: float = 5.0
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    socket_keepalive: bool = True
    
    # Retry settings
    retry_on_timeout: bool = True
    retry_on_error: List[type] = None
    max_retries: int = 3
    retry_delay: float = 0.1
    
    # Health check
    health_check_interval: float = 30.0
    
    def __post_init__(self):
        if self.retry_on_error is None:
            self.retry_on_error = [ConnectionError, TimeoutError]


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    ttl: Optional[int] = None
    created_at: float = 0.0
    accessed_at: float = 0.0
    access_count: int = 0
    
    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()
        if self.accessed_at == 0.0:
            self.accessed_at = time.time()


@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    errors: int = 0
    total_keys: int = 0
    memory_used: int = 0
    connections_active: int = 0
    operations_per_second: float = 0.0
    avg_latency_ms: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "sets": self.sets,
            "deletes": self.deletes,
            "errors": self.errors,
            "total_keys": self.total_keys,
            "memory_used_bytes": self.memory_used,
            "connections_active": self.connections_active,
            "operations_per_second": self.operations_per_second,
            "avg_latency_ms": self.avg_latency_ms,
        }


class RedisCache:
    """High-performance Redis cache manager."""
    
    # Key prefixes for different data types
    PREFIX_TICKER = "ticker"
    PREFIX_ORDERBOOK = "orderbook"
    PREFIX_TRADE = "trade"
    PREFIX_OHLC = "ohlc"
    PREFIX_MARK_PRICE = "mark_price"
    PREFIX_FUNDING_RATE = "funding_rate"
    PREFIX_LIQUIDATION = "liquidation"
    PREFIX_VOLUME = "volume"
    PREFIX_STATS = "stats"
    PREFIX_CONFIG = "config"
    PREFIX_SESSION = "session"
    PREFIX_RATE_LIMIT = "ratelimit"
    
    def __init__(self, config: Optional[RedisConfig] = None):
        self.config = config or RedisConfig()
        self._redis: Optional[Redis] = None
        self._pool: Optional[ConnectionPool] = None
        self._stats = CacheStats()
        self._operation_times: List[float] = []
        self._pubsub_handlers: Dict[str, List[Callable]] = {}
        self._pubsub_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self._connected = False
        self._circuit_breaker_failures = 0
        self._circuit_breaker_threshold = 5
        self._circuit_breaker_reset_time = 30.0
        self._last_failure_time: Optional[float] = None
        
    @property
    def is_connected(self) -> bool:
        """Check if connected to Redis."""
        return self._connected and self._redis is not None
    
    @property
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats
    
    def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker allows operations."""
        if self._circuit_breaker_failures >= self._circuit_breaker_threshold:
            if self._last_failure_time:
                elapsed = time.time() - self._last_failure_time
                if elapsed < self._circuit_breaker_reset_time:
                    return False
                # Reset circuit breaker
                self._circuit_breaker_failures = 0
        return True
    
    def _record_failure(self):
        """Record a failure for circuit breaker."""
        self._circuit_breaker_failures += 1
        self._last_failure_time = time.time()
        self._stats.errors += 1
    
    def _record_operation_time(self, duration: float):
        """Record operation time for metrics."""
        self._operation_times.append(duration)
        # Keep last 1000 measurements
        if len(self._operation_times) > 1000:
            self._operation_times = self._operation_times[-1000:]
        self._stats.avg_latency_ms = sum(self._operation_times) / len(self._operation_times) * 1000
    
    async def connect(self) -> bool:
        """Establish connection to Redis."""
        async with self._lock:
            if self._connected:
                return True
            
            try:
                logger.info(f"Connecting to Redis at {self.config.host}:{self.config.port}")
                
                # Create connection pool
                self._pool = ConnectionPool(
                    host=self.config.host,
                    port=self.config.port,
                    db=self.config.db,
                    password=self.config.password,
                    username=self.config.username,
                    max_connections=self.config.max_connections,
                    socket_connect_timeout=self.config.socket_connect_timeout,
                    socket_keepalive=self.config.socket_keepalive,
                    retry_on_timeout=self.config.retry_on_timeout,
                )
                
                # Create Redis client
                self._redis = Redis(connection_pool=self._pool)
                
                # Test connection
                await self._redis.ping()
                
                self._connected = True
                self._circuit_breaker_failures = 0
                
                # Update stats
                info = await self._redis.info()
                self._stats.connections_active = info.get('connected_clients', 0)
                
                logger.info("Successfully connected to Redis")
                return True
                
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                self._record_failure()
                raise CacheConnectionError(f"Redis connection failed: {e}")
    
    async def disconnect(self) -> None:
        """Close Redis connection."""
        async with self._lock:
            if self._pubsub_task:
                self._pubsub_task.cancel()
                try:
                    await self._pubsub_task
                except asyncio.CancelledError:
                    pass
            
            if self._pool:
                await self._pool.disconnect()
            
            self._redis = None
            self._pool = None
            self._connected = False
            logger.info("Disconnected from Redis")
    
    async def health_check(self) -> bool:
        """Perform health check on Redis connection."""
        if not self._redis:
            return False
        
        try:
            await self._redis.ping()
            return True
        except Exception as e:
            logger.warning(f"Redis health check failed: {e}")
            return False
    
    # =========================================================================
    # Basic Operations
    # =========================================================================
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self._check_circuit_breaker():
            return None
        
        start_time = time.time()
        
        try:
            if not self._redis:
                await self.connect()
            
            value = await self._redis.get(key)
            
            if value is not None:
                self._stats.hits += 1
                return self._deserialize(value)
            else:
                self._stats.misses += 1
                return None
                
        except Exception as e:
            logger.error(f"Cache get error for key '{key}': {e}")
            self._record_failure()
            return None
        finally:
            self._record_operation_time(time.time() - start_time)
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        nx: bool = False,
        xx: bool = False
    ) -> bool:
        """Set value in cache."""
        if not self._check_circuit_breaker():
            return False
        
        start_time = time.time()
        
        try:
            if not self._redis:
                await self.connect()
            
            serialized = self._serialize(value)
            
            result = await self._redis.set(
                key,
                serialized,
                ex=ttl,
                nx=nx,
                xx=xx
            )
            
            if result:
                self._stats.sets += 1
                return True
            return False
            
        except Exception as e:
            logger.error(f"Cache set error for key '{key}': {e}")
            self._record_failure()
            return False
        finally:
            self._record_operation_time(time.time() - start_time)
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if not self._check_circuit_breaker():
            return False
        
        try:
            if not self._redis:
                await self.connect()
            
            result = await self._redis.delete(key)
            if result:
                self._stats.deletes += 1
            return result > 0
            
        except Exception as e:
            logger.error(f"Cache delete error for key '{key}': {e}")
            self._record_failure()
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        try:
            if not self._redis:
                await self.connect()
            
            return await self._redis.exists(key) > 0
            
        except Exception as e:
            logger.error(f"Cache exists error for key '{key}': {e}")
            return False
    
    async def ttl(self, key: str) -> int:
        """Get remaining TTL for key."""
        try:
            if not self._redis:
                await self.connect()
            
            return await self._redis.ttl(key)
            
        except Exception as e:
            logger.error(f"Cache ttl error for key '{key}': {e}")
            return -2
    
    async def expire(self, key: str, seconds: int) -> bool:
        """Set expiration for key."""
        try:
            if not self._redis:
                await self.connect()
            
            return await self._redis.expire(key, seconds)
            
        except Exception as e:
            logger.error(f"Cache expire error for key '{key}': {e}")
            return False
    
    # =========================================================================
    # Batch Operations
    # =========================================================================
    
    async def mget(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values from cache."""
        if not keys:
            return {}
        
        try:
            if not self._redis:
                await self.connect()
            
            values = await self._redis.mget(keys)
            result = {}
            
            for key, value in zip(keys, values):
                if value is not None:
                    result[key] = self._deserialize(value)
                    self._stats.hits += 1
                else:
                    self._stats.misses += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Cache mget error: {e}")
            return {}
    
    async def mset(self, mapping: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set multiple values in cache."""
        if not mapping:
            return True
        
        try:
            if not self._redis:
                await self.connect()
            
            # Serialize all values
            serialized = {k: self._serialize(v) for k, v in mapping.items()}
            
            # Use pipeline for efficiency
            async with self._redis.pipeline() as pipe:
                for key, value in serialized.items():
                    pipe.set(key, value, ex=ttl)
                await pipe.execute()
            
            self._stats.sets += len(mapping)
            return True
            
        except Exception as e:
            logger.error(f"Cache mset error: {e}")
            return False
    
    async def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern."""
        try:
            if not self._redis:
                await self.connect()
            
            keys = await self._redis.keys(pattern)
            if keys:
                return await self._redis.delete(*keys)
            return 0
            
        except Exception as e:
            logger.error(f"Cache delete_pattern error for pattern '{pattern}': {e}")
            return 0
    
    # =========================================================================
    # Hash Operations
    # =========================================================================
    
    async def hget(self, key: str, field: str) -> Optional[Any]:
        """Get hash field value."""
        try:
            if not self._redis:
                await self.connect()
            
            value = await self._redis.hget(key, field)
            if value:
                self._stats.hits += 1
                return self._deserialize(value)
            self._stats.misses += 1
            return None
            
        except Exception as e:
            logger.error(f"Cache hget error: {e}")
            return None
    
    async def hset(self, key: str, field: str, value: Any) -> bool:
        """Set hash field value."""
        try:
            if not self._redis:
                await self.connect()
            
            serialized = self._serialize(value)
            result = await self._redis.hset(key, field, serialized)
            self._stats.sets += 1
            return result >= 0
            
        except Exception as e:
            logger.error(f"Cache hset error: {e}")
            return False
    
    async def hgetall(self, key: str) -> Dict[str, Any]:
        """Get all hash fields."""
        try:
            if not self._redis:
                await self.connect()
            
            data = await self._redis.hgetall(key)
            return {k: self._deserialize(v) for k, v in data.items()}
            
        except Exception as e:
            logger.error(f"Cache hgetall error: {e}")
            return {}
    
    async def hsetall(self, key: str, mapping: Dict[str, Any]) -> bool:
        """Set multiple hash fields."""
        try:
            if not self._redis:
                await self.connect()
            
            serialized = {k: self._serialize(v) for k, v in mapping.items()}
            await self._redis.hset(key, mapping=serialized)
            self._stats.sets += len(mapping)
            return True
            
        except Exception as e:
            logger.error(f"Cache hsetall error: {e}")
            return False
    
    # =========================================================================
    # Sorted Set Operations
    # =========================================================================
    
    async def zadd(self, key: str, mapping: Dict[Any, float]) -> int:
        """Add to sorted set."""
        try:
            if not self._redis:
                await self.connect()
            
            # Redis expects {score: member}, we have {member: score}
            redis_mapping = {self._serialize(k): v for k, v in mapping.items()}
            return await self._redis.zadd(key, redis_mapping)
            
        except Exception as e:
            logger.error(f"Cache zadd error: {e}")
            return 0
    
    async def zrange(
        self,
        key: str,
        start: int = 0,
        stop: int = -1,
        withscores: bool = False
    ) -> List[Any]:
        """Get range from sorted set."""
        try:
            if not self._redis:
                await self.connect()
            
            results = await self._redis.zrange(key, start, stop, withscores=withscores)
            
            if withscores:
                return [(self._deserialize(item), score) for item, score in results]
            return [self._deserialize(item) for item in results]
            
        except Exception as e:
            logger.error(f"Cache zrange error: {e}")
            return []
    
    async def zrangebyscore(
        self,
        key: str,
        min_score: float,
        max_score: float,
        withscores: bool = False
    ) -> List[Any]:
        """Get range by score from sorted set."""
        try:
            if not self._redis:
                await self.connect()
            
            results = await self._redis.zrangebyscore(
                key, min_score, max_score, withscores=withscores
            )
            
            if withscores:
                return [(self._deserialize(item), score) for item, score in results]
            return [self._deserialize(item) for item in results]
            
        except Exception as e:
            logger.error(f"Cache zrangebyscore error: {e}")
            return []
    
    # =========================================================================
    # List Operations
    # =========================================================================
    
    async def lpush(self, key: str, *values: Any) -> int:
        """Push to left of list."""
        try:
            if not self._redis:
                await self.connect()
            
            serialized = [self._serialize(v) for v in values]
            return await self._redis.lpush(key, *serialized)
            
        except Exception as e:
            logger.error(f"Cache lpush error: {e}")
            return 0
    
    async def rpush(self, key: str, *values: Any) -> int:
        """Push to right of list."""
        try:
            if not self._redis:
                await self.connect()
            
            serialized = [self._serialize(v) for v in values]
            return await self._redis.rpush(key, *serialized)
            
        except Exception as e:
            logger.error(f"Cache rpush error: {e}")
            return 0
    
    async def lrange(self, key: str, start: int = 0, stop: int = -1) -> List[Any]:
        """Get range from list."""
        try:
            if not self._redis:
                await self.connect()
            
            results = await self._redis.lrange(key, start, stop)
            return [self._deserialize(item) for item in results]
            
        except Exception as e:
            logger.error(f"Cache lrange error: {e}")
            return []
    
    async def ltrim(self, key: str, start: int, stop: int) -> bool:
        """Trim list to range."""
        try:
            if not self._redis:
                await self.connect()
            
            await self._redis.ltrim(key, start, stop)
            return True
            
        except Exception as e:
            logger.error(f"Cache ltrim error: {e}")
            return False
    
    # =========================================================================
    # Pub/Sub Operations
    # =========================================================================
    
    async def publish(self, channel: str, message: Any) -> int:
        """Publish message to channel."""
        try:
            if not self._redis:
                await self.connect()
            
            serialized = self._serialize(message)
            return await self._redis.publish(channel, serialized)
            
        except Exception as e:
            logger.error(f"Cache publish error: {e}")
            return 0
    
    def subscribe(self, channel: str, handler: Callable[[Any], None]) -> None:
        """Subscribe to channel."""
        if channel not in self._pubsub_handlers:
            self._pubsub_handlers[channel] = []
        self._pubsub_handlers[channel].append(handler)
        
        # Start pubsub task if not running
        if self._pubsub_task is None or self._pubsub_task.done():
            self._pubsub_task = asyncio.create_task(self._pubsub_loop())
    
    def unsubscribe(self, channel: str, handler: Callable[[Any], None]) -> bool:
        """Unsubscribe from channel."""
        if channel in self._pubsub_handlers:
            if handler in self._pubsub_handlers[channel]:
                self._pubsub_handlers[channel].remove(handler)
                return True
        return False
    
    async def _pubsub_loop(self) -> None:
        """Pub/Sub message loop."""
        try:
            # Create separate connection for pub/sub
            pubsub = self._redis.pubsub()
            
            # Subscribe to all channels
            channels = list(self._pubsub_handlers.keys())
            if channels:
                await pubsub.subscribe(*channels)
                logger.info(f"Subscribed to channels: {channels}")
            
            async for message in pubsub.listen():
                if message['type'] == 'message':
                    channel = message['channel']
                    data = self._deserialize(message['data'])
                    
                    # Call handlers
                    for handler in self._pubsub_handlers.get(channel, []):
                        try:
                            if asyncio.iscoroutinefunction(handler):
                                asyncio.create_task(handler(data))
                            else:
                                handler(data)
                        except Exception as e:
                            logger.error(f"Pub/Sub handler error: {e}")
                            
        except asyncio.CancelledError:
            logger.info("Pub/Sub loop cancelled")
        except Exception as e:
            logger.error(f"Pub/Sub loop error: {e}")
    
    # =========================================================================
    # Market Data Specific Methods
    # =========================================================================
    
    async def set_ticker(self, symbol: str, data: Dict[str, Any], ttl: int = 5) -> bool:
        """Cache ticker data."""
        key = f"{self.PREFIX_TICKER}:{symbol.upper()}"
        return await self.set(key, data, ttl=ttl)
    
    async def get_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get cached ticker data."""
        key = f"{self.PREFIX_TICKER}:{symbol.upper()}"
        return await self.get(key)
    
    async def set_orderbook(self, symbol: str, data: Dict[str, Any], ttl: int = 1) -> bool:
        """Cache orderbook data."""
        key = f"{self.PREFIX_ORDERBOOK}:{symbol.upper()}"
        return await self.set(key, data, ttl=ttl)
    
    async def get_orderbook(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get cached orderbook data."""
        key = f"{self.PREFIX_ORDERBOOK}:{symbol.upper()}"
        return await self.get(key)
    
    async def set_ohlc(
        self,
        symbol: str,
        timeframe: str,
        data: List[Dict[str, Any]],
        ttl: int = 3600
    ) -> bool:
        """Cache OHLC candle data."""
        key = f"{self.PREFIX_OHLC}:{symbol.upper()}:{timeframe}"
        return await self.set(key, data, ttl=ttl)
    
    async def get_ohlc(self, symbol: str, timeframe: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached OHLC data."""
        key = f"{self.PREFIX_OHLC}:{symbol.upper()}:{timeframe}"
        return await self.get(key)
    
    async def add_trade(
        self,
        symbol: str,
        trade: Dict[str, Any],
        max_trades: int = 1000
    ) -> bool:
        """Add trade to recent trades list."""
        key = f"{self.PREFIX_TRADE}:{symbol.upper()}"
        
        try:
            await self.lpush(key, trade)
            await self.ltrim(key, 0, max_trades - 1)
            return True
        except Exception as e:
            logger.error(f"Add trade error: {e}")
            return False
    
    async def get_recent_trades(self, symbol: str, count: int = 100) -> List[Dict[str, Any]]:
        """Get recent trades."""
        key = f"{self.PREFIX_TRADE}:{symbol.upper()}"
        return await self.lrange(key, 0, count - 1)
    
    async def set_mark_price(self, symbol: str, price: float, ttl: int = 5) -> bool:
        """Cache mark price for futures."""
        key = f"{self.PREFIX_MARK_PRICE}:{symbol.upper()}"
        return await self.set(key, {"price": price, "timestamp": time.time()}, ttl=ttl)
    
    async def get_mark_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get cached mark price."""
        key = f"{self.PREFIX_MARK_PRICE}:{symbol.upper()}"
        return await self.get(key)
    
    async def set_funding_rate(
        self,
        symbol: str,
        rate: float,
        next_funding_time: float,
        ttl: int = 300
    ) -> bool:
        """Cache funding rate."""
        key = f"{self.PREFIX_FUNDING_RATE}:{symbol.upper()}"
        data = {
            "rate": rate,
            "next_funding_time": next_funding_time,
            "timestamp": time.time()
        }
        return await self.set(key, data, ttl=ttl)
    
    async def get_funding_rate(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get cached funding rate."""
        key = f"{self.PREFIX_FUNDING_RATE}:{symbol.upper()}"
        return await self.get(key)
    
    async def set_volume_stats(
        self,
        symbol: str,
        volume_24h: float,
        quote_volume_24h: float,
        ttl: int = 60
    ) -> bool:
        """Cache volume statistics."""
        key = f"{self.PREFIX_VOLUME}:{symbol.upper()}"
        data = {
            "volume_24h": volume_24h,
            "quote_volume_24h": quote_volume_24h,
            "timestamp": time.time()
        }
        return await self.set(key, data, ttl=ttl)
    
    async def get_volume_stats(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get cached volume statistics."""
        key = f"{self.PREFIX_VOLUME}:{symbol.upper()}"
        return await self.get(key)
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def _serialize(self, value: Any) -> Union[str, bytes]:
        """Serialize value for storage."""
        if isinstance(value, (str, bytes)):
            return value
        try:
            return json.dumps(value, default=str)
        except (TypeError, ValueError):
            return pickle.dumps(value)
    
    def _deserialize(self, value: Union[str, bytes]) -> Any:
        """Deserialize value from storage."""
        if value is None:
            return None
        
        if isinstance(value, bytes):
            try:
                return pickle.loads(value)
            except Exception:
                value = value.decode('utf-8')
        
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    
    async def clear_all(self) -> bool:
        """Clear all cache data."""
        try:
            if not self._redis:
                await self.connect()
            
            await self._redis.flushdb()
            logger.warning("Cache cleared (flushdb)")
            return True
            
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        try:
            if not self._redis:
                await self.connect()
            
            info = await self._redis.info()
            
            # Update stats
            self._stats.total_keys = await self._redis.dbsize()
            self._stats.memory_used = info.get('used_memory', 0)
            self._stats.connections_active = info.get('connected_clients', 0)
            
            return {
                **self._stats.to_dict(),
                "redis_version": info.get('redis_version'),
                "uptime_seconds": info.get('uptime_in_seconds'),
                "total_commands_processed": info.get('total_commands_processed'),
                "keyspace_hits": info.get('keyspace_hits'),
                "keyspace_misses": info.get('keyspace_misses'),
            }
            
        except Exception as e:
            logger.error(f"Get stats error: {e}")
            return self._stats.to_dict()
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern."""
        try:
            if not self._redis:
                await self.connect()
            
            return [k.decode() if isinstance(k, bytes) else k 
                    for k in await self._redis.keys(pattern)]
            
        except Exception as e:
            logger.error(f"Keys error: {e}")
            return []


# Singleton instance
_cache_instance: Optional[RedisCache] = None


def get_cache(config: Optional[RedisConfig] = None) -> RedisCache:
    """Get singleton cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = RedisCache(config)
    return _cache_instance


async def init_cache(config: Optional[RedisConfig] = None) -> RedisCache:
    """Initialize and connect cache."""
    cache = get_cache(config)
    await cache.connect()
    return cache


# Decorator for caching function results
def cached(
    key_prefix: str = "",
    ttl: int = 60,
    key_builder: Optional[Callable] = None
):
    """Decorator to cache function results."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            cache = get_cache()
            
            # Build cache key
            if key_builder:
                cache_key = key_builder(*args, **kwargs)
            else:
                cache_key = f"{key_prefix}:{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cached_value = await cache.get(cache_key)
            if cached_value is not None:
                return cached_value
            
            # Call function and cache result
            result = await func(*args, **kwargs)
            await cache.set(cache_key, result, ttl=ttl)
            return result
        
        return async_wrapper
    return decorator


if __name__ == "__main__":
    async def test_cache():
        """Test Redis cache functionality."""
        cache = RedisCache()
        
        try:
            await cache.connect()
            print("Connected to Redis")
            
            # Test basic operations
            await cache.set("test_key", {"value": 123}, ttl=60)
            value = await cache.get("test_key")
            print(f"Get test_key: {value}")
            
            # Test market data
            await cache.set_ticker("BTCUSDT", {"price": 50000, "volume": 1000})
            ticker = await cache.get_ticker("BTCUSDT")
            print(f"BTCUSDT Ticker: {ticker}")
            
            # Test stats
            stats = await cache.get_stats()
            print(f"Cache stats: {stats}")
            
            await cache.disconnect()
            print("Disconnected from Redis")
            
        except Exception as e:
            print(f"Test failed: {e}")
    
    asyncio.run(test_cache())
