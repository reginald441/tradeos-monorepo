"""
TradeOS Subscription Tier Management
====================================
Subscription tiers, feature gating, and plan management.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

from fastapi import HTTPException

from ..config.saas_config import get_config, get_subscription_limits, SubscriptionTier


class BillingInterval(Enum):
    MONTHLY = "monthly"
    YEARLY = "yearly"


class SubscriptionStatus(Enum):
    ACTIVE = "active"
    TRIALING = "trialing"
    PAST_DUE = "past_due"
    CANCELED = "canceled"
    UNPAID = "unpaid"
    PAUSED = "paused"


@dataclass
class Feature:
    """Feature definition with gating logic"""
    id: str
    name: str
    description: str
    tiers: List[str]  # Tiers that have access
    requires_permission: Optional[str] = None
    quota_key: Optional[str] = None  # Key for usage quota
    is_beta: bool = False


@dataclass
class Plan:
    """Subscription plan definition"""
    id: str
    name: str
    tier: str
    description: str
    price_monthly: int  # In cents
    price_yearly: int   # In cents (with discount)
    features: List[str]
    limits: Dict[str, Any]
    stripe_price_id_monthly: Optional[str] = None
    stripe_price_id_yearly: Optional[str] = None
    is_popular: bool = False
    trial_days: int = 0


@dataclass
class Subscription:
    """User subscription record"""
    id: str
    user_id: str
    tier: str
    status: SubscriptionStatus
    current_period_start: datetime
    current_period_end: datetime
    cancel_at_period_end: bool
    stripe_subscription_id: Optional[str] = None
    stripe_customer_id: Optional[str] = None
    trial_end: Optional[datetime] = None
    canceled_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def is_active(self) -> bool:
        """Check if subscription is active"""
        return self.status in [SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIALING]
    
    def is_trialing(self) -> bool:
        """Check if subscription is in trial"""
        return self.status == SubscriptionStatus.TRIALING
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "tier": self.tier,
            "status": self.status.value,
            "current_period_start": self.current_period_start.isoformat() if self.current_period_start else None,
            "current_period_end": self.current_period_end.isoformat() if self.current_period_end else None,
            "cancel_at_period_end": self.cancel_at_period_end,
            "trial_end": self.trial_end.isoformat() if self.trial_end else None,
            "is_active": self.is_active(),
            "is_trialing": self.is_trialing(),
        }


# Feature definitions
FEATURES: Dict[str, Feature] = {
    "basic_backtest": Feature(
        id="basic_backtest",
        name="Basic Backtesting",
        description="Run basic backtests with historical data",
        tiers=["free", "pro", "enterprise"],
        quota_key="max_backtests_per_day"
    ),
    "advanced_backtest": Feature(
        id="advanced_backtest",
        name="Advanced Backtesting",
        description="Advanced backtesting with custom metrics and optimization",
        tiers=["pro", "enterprise"]
    ),
    "paper_trading": Feature(
        id="paper_trading",
        name="Paper Trading",
        description="Test strategies with simulated trading",
        tiers=["free", "pro", "enterprise"]
    ),
    "live_trading": Feature(
        id="live_trading",
        name="Live Trading",
        description="Execute trades on live exchanges",
        tiers=["pro", "enterprise"]
    ),
    "strategy_creation": Feature(
        id="strategy_creation",
        name="Strategy Creation",
        description="Create and save trading strategies",
        tiers=["free", "pro", "enterprise"],
        quota_key="max_strategies"
    ),
    "custom_indicators": Feature(
        id="custom_indicators",
        name="Custom Indicators",
        description="Create and use custom technical indicators",
        tiers=["pro", "enterprise"]
    ),
    "api_access": Feature(
        id="api_access",
        name="API Access",
        description="Access TradeOS via REST API",
        tiers=["pro", "enterprise"],
        quota_key="max_api_calls_per_day"
    ),
    "webhooks": Feature(
        id="webhooks",
        name="Webhooks",
        description="Receive real-time webhook notifications",
        tiers=["pro", "enterprise"]
    ),
    "team_collaboration": Feature(
        id="team_collaboration",
        name="Team Collaboration",
        description="Share strategies with team members",
        tiers=["enterprise"],
        is_beta=True
    ),
    "priority_support": Feature(
        id="priority_support",
        name="Priority Support",
        description="Get priority customer support",
        tiers=["pro", "enterprise"]
    ),
    "dedicated_support": Feature(
        id="dedicated_support",
        name="Dedicated Support",
        description="Dedicated account manager and support",
        tiers=["enterprise"]
    ),
    "white_label": Feature(
        id="white_label",
        name="White Label",
        description="White label options for enterprise",
        tiers=["enterprise"]
    ),
    "sla": Feature(
        id="sla",
        name="SLA Guarantee",
        description="Service Level Agreement with uptime guarantee",
        tiers=["enterprise"]
    ),
    "custom_development": Feature(
        id="custom_development",
        name="Custom Development",
        description="Custom feature development",
        tiers=["enterprise"]
    ),
}


# Plan definitions
PLANS: Dict[str, Plan] = {
    "free": Plan(
        id="free",
        name="Free",
        tier="free",
        description="Perfect for getting started with algorithmic trading",
        price_monthly=0,
        price_yearly=0,
        features=[
            "basic_backtest",
            "paper_trading",
            "strategy_creation"
        ],
        limits={
            "max_strategies": 1,
            "max_backtests_per_day": 10,
            "max_api_calls_per_day": 100,
            "strategy_executions_per_day": 50,
            "historical_data_days": 30,
            "support_level": "community"
        }
    ),
    "pro_monthly": Plan(
        id="pro_monthly",
        name="Pro",
        tier="pro",
        description="For serious traders who need more power",
        price_monthly=4900,  # $49/month
        price_yearly=47040,  # $39/month billed yearly (20% discount)
        features=[
            "basic_backtest",
            "advanced_backtest",
            "paper_trading",
            "live_trading",
            "strategy_creation",
            "custom_indicators",
            "api_access",
            "webhooks",
            "priority_support"
        ],
        limits={
            "max_strategies": 10,
            "max_backtests_per_day": -1,  # Unlimited
            "max_api_calls_per_day": 10000,
            "strategy_executions_per_day": 1000,
            "historical_data_days": 365,
            "support_level": "email"
        },
        stripe_price_id_monthly="price_pro_monthly",
        stripe_price_id_yearly="price_pro_yearly",
        is_popular=True,
        trial_days=14
    ),
    "enterprise_monthly": Plan(
        id="enterprise_monthly",
        name="Enterprise",
        tier="enterprise",
        description="For teams and businesses with advanced needs",
        price_monthly=29900,  # $299/month
        price_yearly=287040,  # $239/month billed yearly (20% discount)
        features=[
            "basic_backtest",
            "advanced_backtest",
            "paper_trading",
            "live_trading",
            "strategy_creation",
            "custom_indicators",
            "api_access",
            "webhooks",
            "team_collaboration",
            "priority_support",
            "dedicated_support",
            "white_label",
            "sla",
            "custom_development"
        ],
        limits={
            "max_strategies": -1,  # Unlimited
            "max_backtests_per_day": -1,  # Unlimited
            "max_api_calls_per_day": -1,  # Unlimited
            "strategy_executions_per_day": -1,  # Unlimited
            "historical_data_days": -1,  # All available
            "support_level": "priority"
        },
        stripe_price_id_monthly="price_enterprise_monthly",
        stripe_price_id_yearly="price_enterprise_yearly",
        trial_days=30
    ),
}


class TierManager:
    """
    Subscription tier and feature management.
    
    Handles:
    - Feature availability by tier
    - Plan comparisons
    - Tier upgrades/downgrades
    - Usage limit enforcement
    """
    
    def __init__(self):
        self.features = FEATURES
        self.plans = PLANS
        self.config = get_config()
    
    def get_plan(self, plan_id: str) -> Optional[Plan]:
        """Get plan by ID"""
        return self.plans.get(plan_id)
    
    def get_plans_for_tier(self, tier: str) -> List[Plan]:
        """Get all plans for a tier"""
        return [p for p in self.plans.values() if p.tier == tier]
    
    def get_all_plans(self) -> List[Plan]:
        """Get all available plans"""
        return list(self.plans.values())
    
    def get_feature(self, feature_id: str) -> Optional[Feature]:
        """Get feature by ID"""
        return self.features.get(feature_id)
    
    def is_feature_available(self, tier: str, feature_id: str) -> bool:
        """
        Check if a feature is available for a tier.
        
        Args:
            tier: User's subscription tier
            feature_id: Feature to check
            
        Returns:
            True if feature is available
        """
        feature = self.features.get(feature_id)
        if not feature:
            return False
        
        return tier in feature.tiers
    
    def get_available_features(self, tier: str) -> List[Feature]:
        """Get all features available for a tier"""
        return [f for f in self.features.values() if tier in f.tiers]
    
    def get_feature_list(self, tier: str) -> List[Dict[str, Any]]:
        """Get feature list with availability for comparison"""
        result = []
        for feature in self.features.values():
            result.append({
                "id": feature.id,
                "name": feature.name,
                "description": feature.description,
                "available": tier in feature.tiers,
                "is_beta": feature.is_beta
            })
        return result
    
    def get_limits(self, tier: str) -> Dict[str, Any]:
        """Get usage limits for a tier"""
        plan = next((p for p in self.plans.values() if p.tier == tier), None)
        if plan:
            return plan.limits
        return self.plans["free"].limits
    
    def get_limit(self, tier: str, limit_key: str) -> Any:
        """Get specific limit value for a tier"""
        limits = self.get_limits(tier)
        return limits.get(limit_key)
    
    def is_unlimited(self, value: Any) -> bool:
        """Check if a limit value means unlimited"""
        return value == -1 or value is None or value == "unlimited"
    
    def check_limit(
        self,
        tier: str,
        limit_key: str,
        current_value: int
    ) -> tuple[bool, int, int]:
        """
        Check if current usage is within limit.
        
        Args:
            tier: User's subscription tier
            limit_key: Limit to check
            current_value: Current usage value
            
        Returns:
            Tuple of (is_within_limit, limit, remaining)
        """
        limit = self.get_limit(tier, limit_key)
        
        # Unlimited
        if self.is_unlimited(limit):
            return True, -1, -1
        
        limit = int(limit)
        remaining = max(0, limit - current_value)
        
        return current_value < limit, limit, remaining
    
    def enforce_limit(
        self,
        tier: str,
        limit_key: str,
        current_value: int,
        feature_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Enforce a usage limit and raise exception if exceeded.
        
        Args:
            tier: User's subscription tier
            limit_key: Limit to enforce
            current_value: Current usage value
            feature_name: Name of feature for error message
            
        Returns:
            Dict with limit info
            
        Raises:
            HTTPException: If limit exceeded
        """
        is_within, limit, remaining = self.check_limit(tier, limit_key, current_value)
        
        if not is_within:
            feature = feature_name or limit_key.replace("_", " ").title()
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "limit_exceeded",
                    "message": f"{feature} limit exceeded for your {tier} plan",
                    "limit": limit,
                    "current": current_value,
                    "upgrade_url": "/billing/upgrade"
                }
            )
        
        return {
            "limit": limit,
            "current": current_value,
            "remaining": remaining
        }
    
    def compare_plans(self) -> List[Dict[str, Any]]:
        """Generate plan comparison data"""
        comparison = []
        
        for plan in self.plans.values():
            plan_data = {
                "id": plan.id,
                "name": plan.name,
                "tier": plan.tier,
                "description": plan.description,
                "price": {
                    "monthly": plan.price_monthly,
                    "yearly": plan.price_yearly,
                    "monthly_yearly_equivalent": plan.price_yearly // 12 if plan.price_yearly > 0 else 0
                },
                "features": [
                    {
                        "id": f,
                        "name": self.features[f].name if f in self.features else f,
                        "available": True
                    }
                    for f in plan.features
                ],
                "limits": plan.limits,
                "is_popular": plan.is_popular,
                "trial_days": plan.trial_days
            }
            comparison.append(plan_data)
        
        return comparison
    
    def can_upgrade(self, from_tier: str, to_tier: str) -> bool:
        """Check if upgrade path is valid"""
        tier_order = ["free", "pro", "enterprise"]
        
        if from_tier not in tier_order or to_tier not in tier_order:
            return False
        
        return tier_order.index(to_tier) > tier_order.index(from_tier)
    
    def can_downgrade(self, from_tier: str, to_tier: str) -> bool:
        """Check if downgrade path is valid"""
        tier_order = ["free", "pro", "enterprise"]
        
        if from_tier not in tier_order or to_tier not in tier_order:
            return False
        
        return tier_order.index(to_tier) < tier_order.index(from_tier)
    
    def get_recommended_plan(self, usage: Dict[str, int]) -> Optional[str]:
        """
        Recommend a plan based on usage patterns.
        
        Args:
            usage: Dict of usage metrics
            
        Returns:
            Recommended plan ID or None
        """
        # Check if exceeding free limits
        free_limits = self.plans["free"].limits
        
        needs_upgrade = False
        for key, value in usage.items():
            if key in free_limits:
                limit = free_limits[key]
                if limit > 0 and value >= limit * 0.8:  # 80% of limit
                    needs_upgrade = True
                    break
        
        if needs_upgrade:
            # Recommend pro if not using enterprise features
            return "pro_monthly"
        
        return None
    
    def calculate_proration(
        self,
        current_plan_id: str,
        new_plan_id: str,
        days_remaining: int,
        days_in_period: int = 30
    ) -> Dict[str, int]:
        """
        Calculate prorated amounts for plan change.
        
        Args:
            current_plan_id: Current plan ID
            new_plan_id: New plan ID
            days_remaining: Days left in current period
            days_in_period: Total days in billing period
            
        Returns:
            Dict with credit, charge, and net amounts
        """
        current_plan = self.plans.get(current_plan_id)
        new_plan = self.plans.get(new_plan_id)
        
        if not current_plan or not new_plan:
            raise ValueError("Invalid plan ID")
        
        # Calculate unused portion of current plan
        daily_rate_current = current_plan.price_monthly / days_in_period
        unused_credit = int(daily_rate_current * days_remaining)
        
        # Calculate cost of new plan for remaining period
        daily_rate_new = new_plan.price_monthly / days_in_period
        new_plan_cost = int(daily_rate_new * days_remaining)
        
        net_amount = new_plan_cost - unused_credit
        
        return {
            "unused_credit": unused_credit,
            "new_plan_cost": new_plan_cost,
            "net_amount": max(0, net_amount),
            "refund_due": max(0, -net_amount)
        }


# Global tier manager instance
_tier_manager: Optional[TierManager] = None


def get_tier_manager() -> TierManager:
    """Get or create global tier manager"""
    global _tier_manager
    if _tier_manager is None:
        _tier_manager = TierManager()
    return _tier_manager


# FastAPI dependency for feature checking

def require_feature(feature_id: str):
    """
    FastAPI dependency to require a feature.
    
    Usage:
        @app.post("/strategies/{id}/live-trade")
        async def live_trade(
            user: TokenPayload = Depends(require_feature("live_trading"))
        ):
            pass
    """
    from ..auth.jwt_handler import get_current_user
    
    async def checker(user=Depends(get_current_user)):
        manager = get_tier_manager()
        
        if not manager.is_feature_available(user.tier, feature_id):
            feature = manager.get_feature(feature_id)
            feature_name = feature.name if feature else feature_id
            
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "feature_not_available",
                    "message": f"{feature_name} is not available on your {user.tier} plan",
                    "feature": feature_id,
                    "upgrade_url": "/billing/upgrade"
                }
            )
        
        return user
    return checker


# Decorator for feature gating
def feature_required(feature_id: str):
    """
    Decorator to require a feature.
    
    Usage:
        @feature_required("live_trading")
        def execute_live_trade(user, strategy_id):
            pass
    """
    from functools import wraps
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Find user in args/kwargs
            user = kwargs.get('user')
            if not user:
                for arg in args:
                    if hasattr(arg, 'tier'):
                        user = arg
                        break
            
            if not user:
                raise HTTPException(status_code=401, detail="User not found")
            
            manager = get_tier_manager()
            
            if not manager.is_feature_available(user.tier, feature_id):
                feature = manager.get_feature(feature_id)
                feature_name = feature.name if feature else feature_id
                
                raise HTTPException(
                    status_code=403,
                    detail=f"{feature_name} requires a higher tier plan"
                )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator
