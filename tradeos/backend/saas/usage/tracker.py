"""
TradeOS Usage Tracking
======================
API call counting, quota enforcement, and usage analytics.
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from collections import defaultdict
import threading

from fastapi import HTTPException, Request, Response

from ..config.saas_config import get_config, get_subscription_limits
from ..subscriptions.tier_manager import get_tier_manager


class UsageType(Enum):
    """Types of usage to track"""
    API_CALL = "api_call"
    BACKTEST = "backtest"
    STRATEGY_EXECUTION = "strategy_execution"
    STRATEGY_CREATE = "strategy_create"
    WEBHOOK_CALL = "webhook_call"
    DATA_EXPORT = "data_export"


@dataclass
class UsageRecord:
    """Single usage record"""
    user_id: str
    usage_type: UsageType
    timestamp: datetime
    quantity: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UsageQuota:
    """Usage quota for a user"""
    user_id: str
    usage_type: UsageType
    limit: int
    used: int
    period_start: datetime
    period_end: datetime
    
    @property
    def remaining(self) -> int:
        if self.limit < 0:  # Unlimited
            return -1
        return max(0, self.limit - self.used)
    
    @property
    def is_exceeded(self) -> bool:
        if self.limit < 0:  # Unlimited
            return False
        return self.used >= self.limit
    
    @property
    def percent_used(self) -> float:
        if self.limit <= 0:
            return 0.0
        return (self.used / self.limit) * 100


@dataclass
class DailyUsage:
    """Daily usage summary"""
    date: str
    api_calls: int = 0
    backtests: int = 0
    strategy_executions: int = 0
    strategies_created: int = 0
    webhook_calls: int = 0


@dataclass
class UsageSummary:
    """User usage summary"""
    user_id: str
    current_tier: str
    period_start: datetime
    period_end: datetime
    quotas: Dict[str, UsageQuota]
    daily_usage: List[DailyUsage]
    total_requests: int


class UsageTracker:
    """
    Usage tracking system for TradeOS.
    
    Tracks:
    - API calls per user
    - Strategy executions
    - Backtest runs
    - Quota enforcement
    - Usage analytics
    
    Features:
    - Daily/period-based tracking
    - Quota enforcement with configurable limits
    - Warning thresholds
    - Usage analytics and reporting
    """
    
    def __init__(self, tier_manager=None):
        self.tier_manager = tier_manager or get_tier_manager()
        self.config = get_config()
        
        # In-memory storage (use Redis/database in production)
        self._usage_records: List[UsageRecord] = []
        self._daily_usage: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )  # user_id -> date -> type -> count
        
        # User quotas cache
        self._user_quotas: Dict[str, Dict[str, UsageQuota]] = {}
        
        # Warning thresholds (percentage)
        self.warning_thresholds = [80, 90, 95, 99]
        
        # Lock for thread safety
        self._lock = threading.Lock()
    
    def _get_date_key(self, dt: Optional[datetime] = None) -> str:
        """Get date key for daily tracking"""
        if dt is None:
            dt = datetime.now(timezone.utc)
        return dt.strftime("%Y-%m-%d")
    
    def _get_period_dates(self) -> tuple[datetime, datetime]:
        """Get current billing period dates"""
        now = datetime.now(timezone.utc)
        # Assume monthly periods starting from 1st
        period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        # Next month
        if now.month == 12:
            period_end = now.replace(year=now.year + 1, month=1, day=1)
        else:
            period_end = now.replace(month=now.month + 1, day=1)
        
        return period_start, period_end
    
    # ============== Recording Usage ==============
    
    def record(
        self,
        user_id: str,
        usage_type: UsageType,
        quantity: int = 1,
        metadata: Optional[Dict[str, Any]] = None
    ) -> UsageRecord:
        """
        Record usage for a user.
        
        Args:
            user_id: User ID
            usage_type: Type of usage
            quantity: Amount to record
            metadata: Optional metadata
            
        Returns:
            UsageRecord
        """
        with self._lock:
            record = UsageRecord(
                user_id=user_id,
                usage_type=usage_type,
                timestamp=datetime.now(timezone.utc),
                quantity=quantity,
                metadata=metadata or {}
            )
            
            self._usage_records.append(record)
            
            # Update daily usage
            date_key = self._get_date_key()
            self._daily_usage[user_id][date_key][usage_type.value] += quantity
            
            # Trim old records (keep last 100k)
            if len(self._usage_records) > 100000:
                self._usage_records = self._usage_records[-50000:]
            
            return record
    
    def record_api_call(
        self,
        user_id: str,
        endpoint: str,
        method: str = "GET",
        status_code: int = 200
    ) -> UsageRecord:
        """Record an API call"""
        return self.record(
            user_id=user_id,
            usage_type=UsageType.API_CALL,
            metadata={
                "endpoint": endpoint,
                "method": method,
                "status_code": status_code
            }
        )
    
    def record_backtest(
        self,
        user_id: str,
        strategy_id: str,
        duration_ms: Optional[int] = None
    ) -> UsageRecord:
        """Record a backtest execution"""
        return self.record(
            user_id=user_id,
            usage_type=UsageType.BACKTEST,
            metadata={
                "strategy_id": strategy_id,
                "duration_ms": duration_ms
            }
        )
    
    def record_strategy_execution(
        self,
        user_id: str,
        strategy_id: str,
        execution_type: str = "paper"
    ) -> UsageRecord:
        """Record a strategy execution"""
        return self.record(
            user_id=user_id,
            usage_type=UsageType.STRATEGY_EXECUTION,
            metadata={
                "strategy_id": strategy_id,
                "execution_type": execution_type
            }
        )
    
    def record_strategy_created(self, user_id: str, strategy_id: str) -> UsageRecord:
        """Record a strategy creation"""
        return self.record(
            user_id=user_id,
            usage_type=UsageType.STRATEGY_CREATE,
            metadata={"strategy_id": strategy_id}
        )
    
    # ============== Quota Management ==============
    
    def get_quota(
        self,
        user_id: str,
        tier: str,
        usage_type: UsageType
    ) -> UsageQuota:
        """
        Get current quota status for a user.
        
        Args:
            user_id: User ID
            tier: User's subscription tier
            usage_type: Type of usage
            
        Returns:
            UsageQuota with current usage
        """
        # Get limit for tier
        limit_key = self._get_limit_key(usage_type)
        limit = self.tier_manager.get_limit(tier, limit_key)
        
        # Get current usage
        used = self.get_usage_count(user_id, usage_type)
        
        period_start, period_end = self._get_period_dates()
        
        return UsageQuota(
            user_id=user_id,
            usage_type=usage_type,
            limit=limit if limit is not None else 0,
            used=used,
            period_start=period_start,
            period_end=period_end
        )
    
    def get_all_quotas(
        self,
        user_id: str,
        tier: str
    ) -> Dict[str, UsageQuota]:
        """Get all quotas for a user"""
        quotas = {}
        for usage_type in UsageType:
            quotas[usage_type.value] = self.get_quota(user_id, tier, usage_type)
        return quotas
    
    def _get_limit_key(self, usage_type: UsageType) -> str:
        """Map usage type to limit key"""
        mapping = {
            UsageType.API_CALL: "max_api_calls_per_day",
            UsageType.BACKTEST: "max_backtests_per_day",
            UsageType.STRATEGY_EXECUTION: "strategy_executions_per_day",
            UsageType.STRATEGY_CREATE: "max_strategies",
            UsageType.WEBHOOK_CALL: "max_webhook_calls",
            UsageType.DATA_EXPORT: "max_data_exports",
        }
        return mapping.get(usage_type, "")
    
    def get_usage_count(
        self,
        user_id: str,
        usage_type: UsageType,
        period: str = "daily"
    ) -> int:
        """
        Get usage count for a user.
        
        Args:
            user_id: User ID
            usage_type: Type of usage
            period: "daily", "monthly", or "total"
            
        Returns:
            Usage count
        """
        with self._lock:
            if period == "daily":
                date_key = self._get_date_key()
                return self._daily_usage[user_id][date_key][usage_type.value]
            
            elif period == "monthly":
                # Sum current month
                period_start, _ = self._get_period_dates()
                total = 0
                current = period_start
                while current.month == period_start.month:
                    date_key = self._get_date_key(current)
                    total += self._daily_usage[user_id][date_key][usage_type.value]
                    current += timedelta(days=1)
                return total
            
            else:  # total
                return sum(
                    day_data[usage_type.value]
                    for day_data in self._daily_usage[user_id].values()
                )
    
    # ============== Quota Enforcement ==============
    
    def check_quota(
        self,
        user_id: str,
        tier: str,
        usage_type: UsageType,
        raise_exception: bool = True
    ) -> tuple[bool, UsageQuota]:
        """
        Check if user is within quota.
        
        Args:
            user_id: User ID
            tier: User's subscription tier
            usage_type: Type of usage
            raise_exception: Whether to raise exception if exceeded
            
        Returns:
            Tuple of (is_within_quota, quota)
            
        Raises:
            HTTPException: If quota exceeded and raise_exception=True
        """
        quota = self.get_quota(user_id, tier, usage_type)
        
        if quota.is_exceeded:
            if raise_exception:
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "quota_exceeded",
                        "message": f"{usage_type.value} quota exceeded",
                        "quota": {
                            "limit": quota.limit,
                            "used": quota.used,
                            "remaining": quota.remaining
                        },
                        "upgrade_url": "/billing/upgrade"
                    }
                )
            return False, quota
        
        return True, quota
    
    def enforce_quota(
        self,
        user_id: str,
        tier: str,
        usage_type: UsageType
    ) -> UsageQuota:
        """
        Enforce quota and record usage if within limit.
        
        Args:
            user_id: User ID
            tier: User's subscription tier
            usage_type: Type of usage
            
        Returns:
            Updated quota
            
        Raises:
            HTTPException: If quota exceeded
        """
        is_within, quota = self.check_quota(user_id, tier, usage_type)
        
        if is_within:
            self.record(user_id, usage_type)
            # Return updated quota
            return self.get_quota(user_id, tier, usage_type)
        
        return quota
    
    def get_warning_level(
        self,
        user_id: str,
        tier: str,
        usage_type: UsageType
    ) -> Optional[int]:
        """
        Get warning level if approaching quota.
        
        Returns:
            Warning threshold percentage or None
        """
        quota = self.get_quota(user_id, tier, usage_type)
        
        if quota.limit < 0:  # Unlimited
            return None
        
        percent = quota.percent_used
        
        for threshold in sorted(self.warning_thresholds, reverse=True):
            if percent >= threshold:
                return threshold
        
        return None
    
    # ============== Analytics ==============
    
    def get_usage_summary(
        self,
        user_id: str,
        tier: str,
        days: int = 30
    ) -> UsageSummary:
        """Get comprehensive usage summary for a user"""
        # Get quotas
        quotas = self.get_all_quotas(user_id, tier)
        
        # Get daily usage
        daily_usage = []
        for i in range(days):
            date = datetime.now(timezone.utc) - timedelta(days=i)
            date_key = self._get_date_key(date)
            
            day_data = self._daily_usage[user_id].get(date_key, {})
            
            daily_usage.append(DailyUsage(
                date=date_key,
                api_calls=day_data.get(UsageType.API_CALL.value, 0),
                backtests=day_data.get(UsageType.BACKTEST.value, 0),
                strategy_executions=day_data.get(UsageType.STRATEGY_EXECUTION.value, 0),
                strategies_created=day_data.get(UsageType.STRATEGY_CREATE.value, 0),
                webhook_calls=day_data.get(UsageType.WEBHOOK_CALL.value, 0)
            ))
        
        # Calculate total requests
        total_requests = sum(
            day.api_calls for day in daily_usage
        )
        
        period_start, period_end = self._get_period_dates()
        
        return UsageSummary(
            user_id=user_id,
            current_tier=tier,
            period_start=period_start,
            period_end=period_end,
            quotas=quotas,
            daily_usage=daily_usage,
            total_requests=total_requests
        )
    
    def get_top_users(
        self,
        usage_type: UsageType,
        limit: int = 10,
        days: int = 7
    ) -> List[Dict[str, Any]]:
        """Get top users by usage type"""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        
        user_counts: Dict[str, int] = defaultdict(int)
        
        for record in self._usage_records:
            if record.usage_type == usage_type and record.timestamp > cutoff:
                user_counts[record.user_id] += record.quantity
        
        # Sort by count
        sorted_users = sorted(
            user_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]
        
        return [
            {"user_id": user_id, "usage": count}
            for user_id, count in sorted_users
        ]
    
    def get_endpoint_usage(
        self,
        days: int = 7
    ) -> Dict[str, int]:
        """Get usage by endpoint"""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        
        endpoint_counts: Dict[str, int] = defaultdict(int)
        
        for record in self._usage_records:
            if (record.usage_type == UsageType.API_CALL and 
                record.timestamp > cutoff and
                record.metadata):
                endpoint = record.metadata.get("endpoint", "unknown")
                endpoint_counts[endpoint] += record.quantity
        
        return dict(endpoint_counts)
    
    def reset_daily_usage(self, user_id: Optional[str] = None) -> None:
        """Reset daily usage counters (for testing)"""
        with self._lock:
            if user_id:
                date_key = self._get_date_key()
                if user_id in self._daily_usage:
                    self._daily_usage[user_id][date_key] = defaultdict(int)
            else:
                self._daily_usage.clear()


# Global usage tracker instance
_usage_tracker: Optional[UsageTracker] = None


def get_usage_tracker() -> UsageTracker:
    """Get or create global usage tracker"""
    global _usage_tracker
    if _usage_tracker is None:
        _usage_tracker = UsageTracker()
    return _usage_tracker


# FastAPI middleware for automatic usage tracking

class UsageTrackingMiddleware:
    """
    Middleware to automatically track API usage.
    
    Usage:
        app.add_middleware(UsageTrackingMiddleware)
    """
    
    def __init__(
        self,
        app,
        skip_paths: Optional[List[str]] = None
    ):
        self.app = app
        self.skip_paths = skip_paths or ["/health", "/docs", "/openapi.json"]
        self.tracker = get_usage_tracker()
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        request = Request(scope, receive)
        
        # Check if path should be skipped
        should_skip = any(
            request.url.path.startswith(path)
            for path in self.skip_paths
        )
        
        if not should_skip:
            # Get user ID from request state (set by auth middleware)
            user_id = getattr(request.state, "user_id", None)
            
            if user_id:
                # Track the request
                self.tracker.record_api_call(
                    user_id=user_id,
                    endpoint=request.url.path,
                    method=request.method
                )
        
        await self.app(scope, receive, send)


# Decorator for tracking specific function usage

def track_usage(usage_type: UsageType, quantity: int = 1):
    """
    Decorator to track usage for a function.
    
    Usage:
        @track_usage(UsageType.BACKTEST)
        def run_backtest(user_id, strategy):
            pass
    """
    from functools import wraps
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Try to find user_id in args/kwargs
            user_id = kwargs.get('user_id')
            if not user_id:
                for arg in args:
                    if isinstance(arg, str) and arg.startswith("usr_"):
                        user_id = arg
                        break
            
            result = func(*args, **kwargs)
            
            if user_id:
                tracker = get_usage_tracker()
                tracker.record(user_id, usage_type, quantity)
            
            return result
        return wrapper
    return decorator
