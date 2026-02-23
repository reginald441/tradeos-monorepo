"""
TradeOS Stripe Billing Integration
==================================
Stripe integration for subscriptions, payments, and invoicing.
"""

import secrets
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

import stripe
from fastapi import HTTPException, Request, BackgroundTasks

from ..config.saas_config import get_stripe_config, StripeConfig
from ..subscriptions.tier_manager import get_tier_manager, SubscriptionStatus


class PaymentMethodType(Enum):
    CARD = "card"
    BANK_TRANSFER = "bank_transfer"
    PAYPAL = "paypal"


@dataclass
class PaymentIntent:
    """Payment intent result"""
    id: str
    client_secret: str
    status: str
    amount: int
    currency: str


@dataclass
class SubscriptionDetails:
    """Subscription details"""
    id: str
    status: str
    tier: str
    current_period_start: datetime
    current_period_end: datetime
    cancel_at_period_end: bool
    trial_end: Optional[datetime]
    payment_method: Optional[Dict[str, Any]]
    latest_invoice: Optional[Dict[str, Any]]


@dataclass
class Invoice:
    """Invoice details"""
    id: str
    subscription_id: Optional[str]
    status: str
    amount_due: int
    amount_paid: int
    currency: str
    created: datetime
    period_start: Optional[datetime]
    period_end: Optional[datetime]
    pdf_url: Optional[str]
    line_items: List[Dict[str, Any]]


class StripeClient:
    """
    Stripe API client wrapper.
    
    Handles all Stripe API interactions with proper error handling.
    """
    
    def __init__(self, config: Optional[StripeConfig] = None):
        self.config = config or get_stripe_config()
        stripe.api_key = self.config.secret_key
        self.webhook_secret = self.config.webhook_secret
    
    def _handle_stripe_error(self, error: stripe.error.StripeError) -> None:
        """Convert Stripe errors to HTTP exceptions"""
        if isinstance(error, stripe.error.CardError):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "card_error",
                    "message": error.user_message or str(error),
                    "code": error.code
                }
            )
        elif isinstance(error, stripe.error.InvalidRequestError):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "invalid_request",
                    "message": str(error)
                }
            )
        else:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "payment_error",
                    "message": "An error occurred processing your request"
                }
            )


class StripeBillingManager:
    """
    Stripe billing manager for TradeOS.
    
    Handles:
    - Customer creation and management
    - Subscription lifecycle
    - Payment method handling
    - Invoice generation
    - Webhook processing
    """
    
    def __init__(
        self,
        stripe_client: Optional[StripeClient] = None,
        tier_manager=None
    ):
        self.stripe = stripe_client or StripeClient()
        self.tier_manager = tier_manager or get_tier_manager()
        
        # In-memory storage (use database in production)
        self._customers: Dict[str, Dict[str, Any]] = {}  # user_id -> customer data
        self._subscriptions: Dict[str, Dict[str, Any]] = {}  # user_id -> subscription data
    
    # ============== Customer Management ==============
    
    async def create_customer(
        self,
        user_id: str,
        email: str,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Create a Stripe customer for a user.
        
        Args:
            user_id: Internal user ID
            email: Customer email
            name: Customer name
            metadata: Additional metadata
            
        Returns:
            Stripe customer object
        """
        try:
            customer_data = {
                "email": email,
                "metadata": {
                    "user_id": user_id,
                    **(metadata or {})
                }
            }
            
            if name:
                customer_data["name"] = name
            
            customer = stripe.Customer.create(**customer_data)
            
            # Store locally
            self._customers[user_id] = {
                "stripe_customer_id": customer.id,
                "email": email,
                "name": name,
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            
            return {
                "id": customer.id,
                "email": customer.email,
                "created": customer.created
            }
            
        except stripe.error.StripeError as e:
            self.stripe._handle_stripe_error(e)
    
    async def get_customer(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get customer by user ID"""
        customer_data = self._customers.get(user_id)
        if not customer_data:
            return None
        
        try:
            customer = stripe.Customer.retrieve(
                customer_data["stripe_customer_id"]
            )
            return {
                "id": customer.id,
                "email": customer.email,
                "name": customer.name,
                "balance": customer.balance,
                "currency": customer.currency
            }
        except stripe.error.StripeError:
            return None
    
    async def update_customer(
        self,
        user_id: str,
        **updates
    ) -> Dict[str, Any]:
        """Update customer information"""
        customer_data = self._customers.get(user_id)
        if not customer_data:
            raise HTTPException(status_code=404, detail="Customer not found")
        
        try:
            stripe_updates = {}
            if "email" in updates:
                stripe_updates["email"] = updates["email"]
            if "name" in updates:
                stripe_updates["name"] = updates["name"]
            if "phone" in updates:
                stripe_updates["phone"] = updates["phone"]
            
            if stripe_updates:
                customer = stripe.Customer.modify(
                    customer_data["stripe_customer_id"],
                    **stripe_updates
                )
                
                # Update local storage
                self._customers[user_id].update(stripe_updates)
                
                return {
                    "id": customer.id,
                    "email": customer.email,
                    "name": customer.name
                }
            
            return customer_data
            
        except stripe.error.StripeError as e:
            self.stripe._handle_stripe_error(e)
    
    # ============== Payment Methods ==============
    
    async def create_setup_intent(
        self,
        user_id: str
    ) -> Dict[str, Any]:
        """Create a setup intent for adding payment method"""
        customer_data = self._customers.get(user_id)
        if not customer_data:
            raise HTTPException(status_code=404, detail="Customer not found")
        
        try:
            setup_intent = stripe.SetupIntent.create(
                customer=customer_data["stripe_customer_id"],
                payment_method_types=["card"],
                usage="off_session"
            )
            
            return {
                "client_secret": setup_intent.client_secret,
                "status": setup_intent.status
            }
            
        except stripe.error.StripeError as e:
            self.stripe._handle_stripe_error(e)
    
    async def list_payment_methods(
        self,
        user_id: str,
        type: str = "card"
    ) -> List[Dict[str, Any]]:
        """List customer's payment methods"""
        customer_data = self._customers.get(user_id)
        if not customer_data:
            raise HTTPException(status_code=404, detail="Customer not found")
        
        try:
            methods = stripe.PaymentMethod.list(
                customer=customer_data["stripe_customer_id"],
                type=type
            )
            
            return [
                {
                    "id": pm.id,
                    "type": pm.type,
                    "card": {
                        "brand": pm.card.brand if pm.card else None,
                        "last4": pm.card.last4 if pm.card else None,
                        "exp_month": pm.card.exp_month if pm.card else None,
                        "exp_year": pm.card.exp_year if pm.card else None,
                    } if pm.card else None,
                    "billing_details": pm.billing_details
                }
                for pm in methods.data
            ]
            
        except stripe.error.StripeError as e:
            self.stripe._handle_stripe_error(e)
    
    async def set_default_payment_method(
        self,
        user_id: str,
        payment_method_id: str
    ) -> Dict[str, Any]:
        """Set default payment method for customer"""
        customer_data = self._customers.get(user_id)
        if not customer_data:
            raise HTTPException(status_code=404, detail="Customer not found")
        
        try:
            customer = stripe.Customer.modify(
                customer_data["stripe_customer_id"],
                invoice_settings={
                    "default_payment_method": payment_method_id
                }
            )
            
            return {
                "default_payment_method": customer.invoice_settings.default_payment_method
            }
            
        except stripe.error.StripeError as e:
            self.stripe._handle_stripe_error(e)
    
    async def detach_payment_method(
        self,
        payment_method_id: str
    ) -> bool:
        """Remove a payment method"""
        try:
            stripe.PaymentMethod.detach(payment_method_id)
            return True
        except stripe.error.StripeError as e:
            self.stripe._handle_stripe_error(e)
    
    # ============== Subscriptions ==============
    
    async def create_subscription(
        self,
        user_id: str,
        plan_id: str,
        payment_method_id: Optional[str] = None,
        trial_days: Optional[int] = None
    ) -> SubscriptionDetails:
        """
        Create a new subscription.
        
        Args:
            user_id: User ID
            plan_id: Plan ID (e.g., "pro_monthly")
            payment_method_id: Optional payment method to use
            trial_days: Optional trial period
            
        Returns:
            Subscription details
        """
        customer_data = self._customers.get(user_id)
        if not customer_data:
            raise HTTPException(status_code=404, detail="Customer not found")
        
        plan = self.tier_manager.get_plan(plan_id)
        if not plan:
            raise HTTPException(status_code=404, detail="Plan not found")
        
        # Get Stripe price ID
        price_id = plan.stripe_price_id_monthly
        if not price_id:
            raise HTTPException(status_code=400, detail="Plan not configured for billing")
        
        try:
            subscription_data = {
                "customer": customer_data["stripe_customer_id"],
                "items": [{"price": price_id}],
                "payment_behavior": "default_incomplete",
                "expand": ["latest_invoice.payment_intent"],
                "metadata": {
                    "user_id": user_id,
                    "plan_id": plan_id,
                    "tier": plan.tier
                }
            }
            
            if payment_method_id:
                subscription_data["default_payment_method"] = payment_method_id
            
            if trial_days and trial_days > 0:
                subscription_data["trial_period_days"] = trial_days
            
            subscription = stripe.Subscription.create(**subscription_data)
            
            # Store locally
            self._subscriptions[user_id] = {
                "stripe_subscription_id": subscription.id,
                "plan_id": plan_id,
                "tier": plan.tier,
                "status": subscription.status
            }
            
            return self._format_subscription(subscription, plan.tier)
            
        except stripe.error.StripeError as e:
            self.stripe._handle_stripe_error(e)
    
    async def get_subscription(
        self,
        user_id: str
    ) -> Optional[SubscriptionDetails]:
        """Get user's subscription"""
        subscription_data = self._subscriptions.get(user_id)
        if not subscription_data:
            return None
        
        try:
            subscription = stripe.Subscription.retrieve(
                subscription_data["stripe_subscription_id"],
                expand=["default_payment_method", "latest_invoice"]
            )
            
            return self._format_subscription(
                subscription,
                subscription_data.get("tier", "free")
            )
            
        except stripe.error.StripeError:
            return None
    
    async def update_subscription(
        self,
        user_id: str,
        new_plan_id: str,
        proration_behavior: str = "create_prorations"
    ) -> SubscriptionDetails:
        """
        Update/upgrade subscription to new plan.
        
        Args:
            user_id: User ID
            new_plan_id: New plan ID
            proration_behavior: How to handle proration
            
        Returns:
            Updated subscription details
        """
        subscription_data = self._subscriptions.get(user_id)
        if not subscription_data:
            raise HTTPException(status_code=404, detail="No active subscription")
        
        new_plan = self.tier_manager.get_plan(new_plan_id)
        if not new_plan:
            raise HTTPException(status_code=404, detail="Plan not found")
        
        price_id = new_plan.stripe_price_id_monthly
        if not price_id:
            raise HTTPException(status_code=400, detail="Plan not configured for billing")
        
        try:
            subscription = stripe.Subscription.modify(
                subscription_data["stripe_subscription_id"],
                items=[{
                    "id": stripe.Subscription.retrieve(
                        subscription_data["stripe_subscription_id"]
                    ).items.data[0].id,
                    "price": price_id
                }],
                proration_behavior=proration_behavior,
                metadata={
                    "plan_id": new_plan_id,
                    "tier": new_plan.tier
                }
            )
            
            # Update local storage
            self._subscriptions[user_id]["plan_id"] = new_plan_id
            self._subscriptions[user_id]["tier"] = new_plan.tier
            
            return self._format_subscription(subscription, new_plan.tier)
            
        except stripe.error.StripeError as e:
            self.stripe._handle_stripe_error(e)
    
    async def cancel_subscription(
        self,
        user_id: str,
        at_period_end: bool = True
    ) -> SubscriptionDetails:
        """
        Cancel a subscription.
        
        Args:
            user_id: User ID
            at_period_end: If True, cancel at period end. If False, cancel immediately.
            
        Returns:
            Updated subscription details
        """
        subscription_data = self._subscriptions.get(user_id)
        if not subscription_data:
            raise HTTPException(status_code=404, detail="No active subscription")
        
        try:
            if at_period_end:
                subscription = stripe.Subscription.modify(
                    subscription_data["stripe_subscription_id"],
                    cancel_at_period_end=True
                )
            else:
                subscription = stripe.Subscription.delete(
                    subscription_data["stripe_subscription_id"]
                )
            
            return self._format_subscription(
                subscription,
                subscription_data.get("tier", "free")
            )
            
        except stripe.error.StripeError as e:
            self.stripe._handle_stripe_error(e)
    
    async def reactivate_subscription(self, user_id: str) -> SubscriptionDetails:
        """Reactivate a subscription scheduled for cancellation"""
        subscription_data = self._subscriptions.get(user_id)
        if not subscription_data:
            raise HTTPException(status_code=404, detail="No active subscription")
        
        try:
            subscription = stripe.Subscription.modify(
                subscription_data["stripe_subscription_id"],
                cancel_at_period_end=False
            )
            
            return self._format_subscription(
                subscription,
                subscription_data.get("tier", "free")
            )
            
        except stripe.error.StripeError as e:
            self.stripe._handle_stripe_error(e)
    
    def _format_subscription(
        self,
        subscription: stripe.Subscription,
        tier: str
    ) -> SubscriptionDetails:
        """Format Stripe subscription to our model"""
        return SubscriptionDetails(
            id=subscription.id,
            status=subscription.status,
            tier=tier,
            current_period_start=datetime.fromtimestamp(
                subscription.current_period_start,
                tz=timezone.utc
            ),
            current_period_end=datetime.fromtimestamp(
                subscription.current_period_end,
                tz=timezone.utc
            ),
            cancel_at_period_end=subscription.cancel_at_period_end,
            trial_end=datetime.fromtimestamp(
                subscription.trial_end,
                tz=timezone.utc
            ) if subscription.trial_end else None,
            payment_method={
                "id": subscription.default_payment_method.id if subscription.default_payment_method else None,
                "type": "card" if subscription.default_payment_method else None
            },
            latest_invoice={
                "id": subscription.latest_invoice.id if subscription.latest_invoice else None,
                "status": subscription.latest_invoice.status if subscription.latest_invoice else None
            } if subscription.latest_invoice else None
        )
    
    # ============== Invoices ==============
    
    async def list_invoices(
        self,
        user_id: str,
        limit: int = 10
    ) -> List[Invoice]:
        """List customer's invoices"""
        customer_data = self._customers.get(user_id)
        if not customer_data:
            raise HTTPException(status_code=404, detail="Customer not found")
        
        try:
            invoices = stripe.Invoice.list(
                customer=customer_data["stripe_customer_id"],
                limit=limit
            )
            
            return [
                Invoice(
                    id=inv.id,
                    subscription_id=inv.subscription,
                    status=inv.status,
                    amount_due=inv.amount_due,
                    amount_paid=inv.amount_paid,
                    currency=inv.currency,
                    created=datetime.fromtimestamp(inv.created, tz=timezone.utc),
                    period_start=datetime.fromtimestamp(inv.period_start, tz=timezone.utc) if inv.period_start else None,
                    period_end=datetime.fromtimestamp(inv.period_end, tz=timezone.utc) if inv.period_end else None,
                    pdf_url=inv.invoice_pdf,
                    line_items=[
                        {
                            "description": li.description,
                            "amount": li.amount,
                            "currency": li.currency
                        }
                        for li in inv.lines.data
                    ]
                )
                for inv in invoices.data
            ]
            
        except stripe.error.StripeError as e:
            self.stripe._handle_stripe_error(e)
    
    async def get_upcoming_invoice(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get upcoming invoice preview"""
        customer_data = self._customers.get(user_id)
        if not customer_data:
            raise HTTPException(status_code=404, detail="Customer not found")
        
        try:
            subscription_data = self._subscriptions.get(user_id)
            subscription_id = subscription_data["stripe_subscription_id"] if subscription_data else None
            
            invoice = stripe.Invoice.upcoming(
                customer=customer_data["stripe_customer_id"],
                subscription=subscription_id
            )
            
            return {
                "amount_due": invoice.amount_due,
                "currency": invoice.currency,
                "period_start": invoice.period_start,
                "period_end": invoice.period_end,
                "line_items": [
                    {
                        "description": li.description,
                        "amount": li.amount
                    }
                    for li in invoice.lines.data
                ]
            }
            
        except stripe.error.StripeError:
            return None
    
    # ============== Webhook Handling ==============
    
    async def handle_webhook(
        self,
        request: Request,
        background_tasks: Optional[BackgroundTasks] = None
    ) -> Dict[str, Any]:
        """
        Handle Stripe webhook events.
        
        Args:
            request: FastAPI request object
            background_tasks: Background tasks handler
            
        Returns:
            Event processing result
        """
        payload = await request.body()
        sig_header = request.headers.get("stripe-signature")
        
        try:
            event = stripe.Webhook.construct_event(
                payload,
                sig_header,
                self.webhook_secret
            )
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid payload")
        except stripe.error.SignatureVerificationError:
            raise HTTPException(status_code=400, detail="Invalid signature")
        
        # Process event
        event_type = event["type"]
        data = event["data"]["object"]
        
        handler = getattr(self, f"_handle_{event_type.replace('.', '_')}", None)
        
        if handler:
            if background_tasks:
                background_tasks.add_task(handler, data)
            else:
                await handler(data)
        
        return {"status": "processed", "type": event_type}
    
    async def _handle_invoice_payment_succeeded(self, data: Dict[str, Any]) -> None:
        """Handle successful payment"""
        # Update subscription status, send receipt, etc.
        pass
    
    async def _handle_invoice_payment_failed(self, data: Dict[str, Any]) -> None:
        """Handle failed payment"""
        # Notify user, update subscription status
        pass
    
    async def _handle_customer_subscription_deleted(self, data: Dict[str, Any]) -> None:
        """Handle subscription cancellation"""
        # Downgrade user to free tier
        user_id = data.get("metadata", {}).get("user_id")
        if user_id and user_id in self._subscriptions:
            self._subscriptions[user_id]["status"] = "canceled"
    
    async def _handle_customer_subscription_updated(self, data: Dict[str, Any]) -> None:
        """Handle subscription update"""
        # Update local subscription data
        user_id = data.get("metadata", {}).get("user_id")
        if user_id and user_id in self._subscriptions:
            self._subscriptions[user_id]["status"] = data.get("status")


# Global billing manager instance
_billing_manager: Optional[StripeBillingManager] = None


def get_billing_manager() -> StripeBillingManager:
    """Get or create global billing manager"""
    global _billing_manager
    if _billing_manager is None:
        _billing_manager = StripeBillingManager()
    return _billing_manager
