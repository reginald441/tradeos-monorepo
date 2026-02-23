"""
Billing Router

Handles subscription management, payments, and billing history.
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, date
from decimal import Decimal

from ..dependencies.auth import get_current_user
from ..database.models import User, Subscription
from ..database.connection import get_db_session
from ..saas.billing.stripe_integration import StripeIntegration
from ..saas.subscriptions.tier_manager import TierManager, SubscriptionTier

router = APIRouter(prefix="/billing", tags=["Billing"])


# Request/Response Models
class SubscriptionPlan(BaseModel):
    id: str
    name: str
    description: str
    price_monthly: float
    price_yearly: float
    currency: str
    features: List[str]
    limits: Dict[str, Any]


class CurrentSubscription(BaseModel):
    id: int
    tier: str
    status: str
    current_period_start: datetime
    current_period_end: datetime
    cancel_at_period_end: bool
    amount: float
    currency: str
    payment_method_last4: Optional[str]


class PaymentMethodRequest(BaseModel):
    payment_method_id: str


class PaymentMethodResponse(BaseModel):
    id: str
    type: str
    last4: str
    exp_month: int
    exp_year: int
    brand: str
    is_default: bool


class InvoiceResponse(BaseModel):
    id: str
    amount_due: float
    amount_paid: float
    currency: str
    status: str
    description: str
    period_start: datetime
    period_end: datetime
    paid_at: Optional[datetime]
    pdf_url: Optional[str]


class CreateSubscriptionRequest(BaseModel):
    tier: str
    payment_method_id: str
    billing_cycle: Literal["monthly", "yearly"] = "monthly"


class UsageResponse(BaseModel):
    strategies_used: int
    strategies_limit: int
    backtests_used: int
    backtests_limit: int
    api_calls_used: int
    api_calls_limit: int
    reset_date: date


# Initialize components
stripe_integration = StripeIntegration()
tier_manager = TierManager()


@router.get("/plans", response_model=List[SubscriptionPlan])
async def get_subscription_plans():
    """Get all available subscription plans."""
    return [
        SubscriptionPlan(
            id="free",
            name="Free",
            description="Perfect for getting started with algorithmic trading",
            price_monthly=0,
            price_yearly=0,
            currency="USD",
            features=[
                "1 active strategy",
                "5 backtests per month",
                "Basic indicators",
                "Paper trading",
                "Community support"
            ],
            limits={
                "strategies": 1,
                "backtests_per_month": 5,
                "api_calls_per_day": 100,
                "exchanges": 1,
                "data_history_days": 30
            }
        ),
        SubscriptionPlan(
            id="pro",
            name="Pro",
            description="For serious traders who need more power",
            price_monthly=99,
            price_yearly=79,
            currency="USD",
            features=[
                "10 active strategies",
                "Unlimited backtests",
                "Advanced indicators & ML models",
                "Live trading",
                "Risk management tools",
                "Priority support",
                "API access"
            ],
            limits={
                "strategies": 10,
                "backtests_per_month": -1,  # Unlimited
                "api_calls_per_day": 10000,
                "exchanges": 3,
                "data_history_days": 365
            }
        ),
        SubscriptionPlan(
            id="enterprise",
            name="Enterprise",
            description="For professional trading firms and institutions",
            price_monthly=499,
            price_yearly=399,
            currency="USD",
            features=[
                "Unlimited strategies",
                "Unlimited backtests",
                "All Pro features",
                "Custom strategy development",
                "Dedicated support",
                "SLA guarantee",
                "White-label options",
                "On-premise deployment"
            ],
            limits={
                "strategies": -1,  # Unlimited
                "backtests_per_month": -1,
                "api_calls_per_day": 100000,
                "exchanges": -1,
                "data_history_days": -1
            }
        )
    ]


@router.get("/subscription", response_model=CurrentSubscription)
async def get_current_subscription(current_user: User = Depends(get_current_user)):
    """Get current user's subscription details."""
    async with get_db_session() as session:
        from sqlalchemy import select
        
        result = await session.execute(
            select(Subscription).where(
                Subscription.user_id == current_user.id,
                Subscription.status.in_(["active", "trialing", "past_due"])
            )
        )
        subscription = result.scalar_one_or_none()
        
        if not subscription:
            # Return free tier info
            return CurrentSubscription(
                id=0,
                tier="free",
                status="active",
                current_period_start=datetime.utcnow(),
                current_period_end=datetime.utcnow(),
                cancel_at_period_end=False,
                amount=0,
                currency="USD",
                payment_method_last4=None
            )
        
        return CurrentSubscription(
            id=subscription.id,
            tier=subscription.tier,
            status=subscription.status,
            current_period_start=subscription.current_period_start,
            current_period_end=subscription.current_period_end,
            cancel_at_period_end=subscription.cancel_at_period_end,
            amount=float(subscription.amount) if subscription.amount else 0,
            currency=subscription.currency or "USD",
            payment_method_last4=subscription.payment_method_last4
        )


@router.post("/subscribe")
async def create_subscription(
    request: CreateSubscriptionRequest,
    current_user: User = Depends(get_current_user)
):
    """Create a new subscription."""
    try:
        subscription = await stripe_integration.create_subscription(
            user_id=current_user.id,
            tier=request.tier,
            payment_method_id=request.payment_method_id,
            billing_cycle=request.billing_cycle
        )
        
        return {
            "success": True,
            "message": "Subscription created successfully",
            "subscription": {
                "id": subscription.id,
                "tier": subscription.tier,
                "status": subscription.status,
                "current_period_end": subscription.current_period_end.isoformat() if subscription.current_period_end else None
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to create subscription: {str(e)}")


@router.post("/cancel")
async def cancel_subscription(
    at_period_end: bool = True,
    current_user: User = Depends(get_current_user)
):
    """Cancel current subscription."""
    async with get_db_session() as session:
        from sqlalchemy import select
        
        result = await session.execute(
            select(Subscription).where(
                Subscription.user_id == current_user.id,
                Subscription.status == "active"
            )
        )
        subscription = result.scalar_one_or_none()
        
        if not subscription:
            raise HTTPException(status_code=404, detail="No active subscription found")
        
        try:
            await stripe_integration.cancel_subscription(
                subscription_id=subscription.stripe_subscription_id,
                at_period_end=at_period_end
            )
            
            return {
                "success": True,
                "message": f"Subscription will be cancelled {'at period end' if at_period_end else 'immediately'}"
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to cancel subscription: {str(e)}")


@router.post("/reactivate")
async def reactivate_subscription(current_user: User = Depends(get_current_user)):
    """Reactivate a cancelled subscription."""
    async with get_db_session() as session:
        from sqlalchemy import select
        
        result = await session.execute(
            select(Subscription).where(
                Subscription.user_id == current_user.id,
                Subscription.cancel_at_period_end == True
            )
        )
        subscription = result.scalar_one_or_none()
        
        if not subscription:
            raise HTTPException(status_code=404, detail="No cancelled subscription found")
        
        try:
            await stripe_integration.reactivate_subscription(
                subscription_id=subscription.stripe_subscription_id
            )
            
            return {
                "success": True,
                "message": "Subscription reactivated successfully"
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to reactivate subscription: {str(e)}")


@router.get("/payment-methods", response_model=List[PaymentMethodResponse])
async def get_payment_methods(current_user: User = Depends(get_current_user)):
    """Get user's saved payment methods."""
    try:
        methods = await stripe_integration.get_payment_methods(current_user.id)
        
        return [
            PaymentMethodResponse(
                id=pm["id"],
                type=pm["type"],
                last4=pm["last4"],
                exp_month=pm["exp_month"],
                exp_year=pm["exp_year"],
                brand=pm["brand"],
                is_default=pm.get("is_default", False)
            )
            for pm in methods
        ]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to get payment methods: {str(e)}")


@router.post("/payment-methods")
async def add_payment_method(
    request: PaymentMethodRequest,
    current_user: User = Depends(get_current_user)
):
    """Add a new payment method."""
    try:
        pm = await stripe_integration.attach_payment_method(
            user_id=current_user.id,
            payment_method_id=request.payment_method_id
        )
        
        return {
            "success": True,
            "message": "Payment method added successfully",
            "payment_method": PaymentMethodResponse(
                id=pm["id"],
                type=pm["type"],
                last4=pm["last4"],
                exp_month=pm["exp_month"],
                exp_year=pm["exp_year"],
                brand=pm["brand"],
                is_default=False
            )
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to add payment method: {str(e)}")


@router.delete("/payment-methods/{payment_method_id}")
async def remove_payment_method(
    payment_method_id: str,
    current_user: User = Depends(get_current_user)
):
    """Remove a payment method."""
    try:
        await stripe_integration.detach_payment_method(payment_method_id)
        
        return {
            "success": True,
            "message": "Payment method removed successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to remove payment method: {str(e)}")


@router.get("/invoices", response_model=List[InvoiceResponse])
async def get_invoices(
    limit: int = 10,
    current_user: User = Depends(get_current_user)
):
    """Get billing history/invoices."""
    try:
        invoices = await stripe_integration.get_invoices(
            user_id=current_user.id,
            limit=limit
        )
        
        return [
            InvoiceResponse(
                id=inv["id"],
                amount_due=inv["amount_due"],
                amount_paid=inv["amount_paid"],
                currency=inv["currency"],
                status=inv["status"],
                description=inv["description"],
                period_start=inv["period_start"],
                period_end=inv["period_end"],
                paid_at=inv.get("paid_at"),
                pdf_url=inv.get("pdf_url")
            )
            for inv in invoices
        ]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to get invoices: {str(e)}")


@router.get("/usage", response_model=UsageResponse)
async def get_usage(current_user: User = Depends(get_current_user)):
    """Get current usage statistics."""
    async with get_db_session() as session:
        from sqlalchemy import select, func
        from ..database.models import Strategy, BacktestResult
        from datetime import timedelta
        
        # Get tier limits
        tier = current_user.subscription_tier
        limits = tier_manager.get_tier_limits(tier)
        
        # Count strategies
        result = await session.execute(
            select(func.count(Strategy.id)).where(Strategy.user_id == current_user.id)
        )
        strategies_used = result.scalar() or 0
        
        # Count backtests this month
        today = date.today()
        month_start = today.replace(day=1)
        result = await session.execute(
            select(func.count(BacktestResult.id)).where(
                BacktestResult.user_id == current_user.id,
                func.date(BacktestResult.created_at) >= month_start
            )
        )
        backtests_used = result.scalar() or 0
        
        # Mock API calls (would come from usage tracking)
        api_calls_used = 500
        
        # Calculate reset date
        if today.month == 12:
            reset_date = date(today.year + 1, 1, 1)
        else:
            reset_date = date(today.year, today.month + 1, 1)
        
        return UsageResponse(
            strategies_used=strategies_used,
            strategies_limit=limits.get("strategies", 1),
            backtests_used=backtests_used,
            backtests_limit=limits.get("backtests_per_month", 5),
            api_calls_used=api_calls_used,
            api_calls_limit=limits.get("api_calls_per_day", 100),
            reset_date=reset_date
        )


@router.post("/upgrade")
async def upgrade_subscription(
    target_tier: str,
    current_user: User = Depends(get_current_user)
):
    """Upgrade subscription to a higher tier."""
    async with get_db_session() as session:
        from sqlalchemy import select
        
        result = await session.execute(
            select(Subscription).where(
                Subscription.user_id == current_user.id,
                Subscription.status == "active"
            )
        )
        subscription = result.scalar_one_or_none()
        
        if not subscription:
            raise HTTPException(status_code=404, detail="No active subscription found")
        
        try:
            await stripe_integration.update_subscription_tier(
                subscription_id=subscription.stripe_subscription_id,
                new_tier=target_tier
            )
            
            return {
                "success": True,
                "message": f"Subscription upgraded to {target_tier}"
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to upgrade: {str(e)}")


@router.post("/webhook")
async def stripe_webhook(request: Request):
    """Handle Stripe webhook events."""
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    
    try:
        event = await stripe_integration.handle_webhook(payload, sig_header)
        
        # Process the event
        if event["type"] == "invoice.payment_succeeded":
            # Handle successful payment
            pass
        elif event["type"] == "invoice.payment_failed":
            # Handle failed payment
            pass
        elif event["type"] == "customer.subscription.deleted":
            # Handle subscription cancellation
            pass
        
        return {"received": True}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Webhook error: {str(e)}")


@router.get("/customer-portal")
async def get_customer_portal(current_user: User = Depends(get_current_user)):
    """Get Stripe customer portal URL."""
    try:
        portal_url = await stripe_integration.create_portal_session(current_user.id)
        
        return {
            "url": portal_url
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to create portal session: {str(e)}")
