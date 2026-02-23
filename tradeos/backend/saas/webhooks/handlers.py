"""
TradeOS Webhook Handlers
========================
Webhook handlers for Stripe, exchanges, and external services.
"""

import hmac
import hashlib
import json
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import asyncio

from fastapi import HTTPException, Request, BackgroundTasks, Header

from ..config.saas_config import get_config
from ..billing.stripe_integration import get_billing_manager
from ..notifications.email import get_email_service


class WebhookSource(Enum):
    """Webhook source types"""
    STRIPE = "stripe"
    COINBASE = "coinbase"
    BINANCE = "binance"
    CUSTOM = "custom"


class WebhookEventType(Enum):
    """Common webhook event types"""
    # Payment events
    PAYMENT_SUCCEEDED = "payment.succeeded"
    PAYMENT_FAILED = "payment.failed"
    PAYMENT_REFUNDED = "payment.refunded"
    
    # Subscription events
    SUBSCRIPTION_CREATED = "subscription.created"
    SUBSCRIPTION_UPDATED = "subscription.updated"
    SUBSCRIPTION_CANCELED = "subscription.canceled"
    SUBSCRIPTION_RENEWED = "subscription.renewed"
    
    # Trading events
    ORDER_FILLED = "order.filled"
    ORDER_CANCELED = "order.canceled"
    TRADE_EXECUTED = "trade.executed"
    
    # Account events
    ACCOUNT_UPDATED = "account.updated"
    BALANCE_CHANGED = "balance.changed"
    
    # Security events
    LOGIN_DETECTED = "security.login_detected"
    SUSPICIOUS_ACTIVITY = "security.suspicious_activity"


@dataclass
class WebhookEvent:
    """Standardized webhook event"""
    source: WebhookSource
    event_type: str
    event_id: str
    timestamp: datetime
    payload: Dict[str, Any]
    raw_body: Optional[str] = None
    signature: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source.value,
            "event_type": self.event_type,
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "payload": self.payload
        }


@dataclass
class WebhookEndpoint:
    """Registered webhook endpoint"""
    id: str
    user_id: str
    url: str
    events: List[str]
    secret: str
    is_active: bool = True
    created_at: Optional[datetime] = None
    last_triggered: Optional[datetime] = None
    failure_count: int = 0
    
    def should_trigger(self, event_type: str) -> bool:
        """Check if this endpoint should receive the event"""
        if not self.is_active:
            return False
        return event_type in self.events or "*" in self.events


class WebhookSignatureVerifier:
    """Verify webhook signatures from various sources"""
    
    @staticmethod
    def verify_stripe(
        payload: bytes,
        signature: str,
        secret: str
    ) -> bool:
        """Verify Stripe webhook signature"""
        try:
            from stripe.webhook import WebhookSignature
            WebhookSignature.verify_header(
                payload,
                signature,
                secret,
                tolerance=300  # 5 minute tolerance
            )
            return True
        except Exception:
            return False
    
    @staticmethod
    def verify_coinbase(
        payload: bytes,
        signature: str,
        secret: str
    ) -> bool:
        """Verify Coinbase webhook signature"""
        try:
            expected = hmac.new(
                secret.encode(),
                payload,
                hashlib.sha256
            ).hexdigest()
            return hmac.compare_digest(signature, expected)
        except Exception:
            return False
    
    @staticmethod
    def verify_binance(
        payload: bytes,
        signature: str,
        secret: str
    ) -> bool:
        """Verify Binance webhook signature"""
        try:
            expected = hmac.new(
                secret.encode(),
                payload,
                hashlib.sha256
            ).hexdigest()
            return hmac.compare_digest(signature, expected)
        except Exception:
            return False
    
    @staticmethod
    def verify_custom(
        payload: bytes,
        signature: str,
        secret: str,
        algorithm: str = "sha256"
    ) -> bool:
        """Verify custom webhook signature"""
        try:
            hash_func = getattr(hashlib, algorithm, hashlib.sha256)
            expected = hmac.new(
                secret.encode(),
                payload,
                hash_func
            ).hexdigest()
            return hmac.compare_digest(signature, expected)
        except Exception:
            return False


class StripeWebhookHandler:
    """Handler for Stripe webhooks"""
    
    def __init__(self):
        self.billing_manager = get_billing_manager()
        self.email_service = get_email_service()
    
    async def handle(self, event: WebhookEvent) -> Dict[str, Any]:
        """Handle Stripe webhook event"""
        event_type = event.event_type
        payload = event.payload
        
        handlers = {
            "invoice.payment_succeeded": self._handle_payment_succeeded,
            "invoice.payment_failed": self._handle_payment_failed,
            "customer.subscription.created": self._handle_subscription_created,
            "customer.subscription.updated": self._handle_subscription_updated,
            "customer.subscription.deleted": self._handle_subscription_deleted,
            "customer.created": self._handle_customer_created,
            "payment_intent.succeeded": self._handle_payment_intent_succeeded,
            "payment_intent.payment_failed": self._handle_payment_intent_failed,
        }
        
        handler = handlers.get(event_type)
        if handler:
            return await handler(payload)
        
        return {"status": "ignored", "reason": "Unhandled event type"}
    
    async def _handle_payment_succeeded(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle successful payment"""
        invoice = data.get("data", {}).get("object", {})
        customer_id = invoice.get("customer")
        amount_paid = invoice.get("amount_paid", 0) / 100  # Convert from cents
        
        # TODO: Get user by customer_id and send receipt email
        
        return {
            "status": "processed",
            "action": "payment_succeeded",
            "customer_id": customer_id,
            "amount": amount_paid
        }
    
    async def _handle_payment_failed(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle failed payment"""
        invoice = data.get("data", {}).get("object", {})
        customer_id = invoice.get("customer")
        
        # TODO: Notify user of failed payment
        
        return {
            "status": "processed",
            "action": "payment_failed",
            "customer_id": customer_id
        }
    
    async def _handle_subscription_created(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle new subscription"""
        subscription = data.get("data", {}).get("object", {})
        
        return {
            "status": "processed",
            "action": "subscription_created",
            "subscription_id": subscription.get("id")
        }
    
    async def _handle_subscription_updated(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle subscription update"""
        subscription = data.get("data", {}).get("object", {})
        
        return {
            "status": "processed",
            "action": "subscription_updated",
            "subscription_id": subscription.get("id")
        }
    
    async def _handle_subscription_deleted(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle subscription cancellation"""
        subscription = data.get("data", {}).get("object", {})
        
        # Downgrade user to free tier
        # TODO: Implement user tier downgrade
        
        return {
            "status": "processed",
            "action": "subscription_deleted",
            "subscription_id": subscription.get("id")
        }
    
    async def _handle_customer_created(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle new customer"""
        customer = data.get("data", {}).get("object", {})
        
        return {
            "status": "processed",
            "action": "customer_created",
            "customer_id": customer.get("id")
        }
    
    async def _handle_payment_intent_succeeded(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle successful payment intent"""
        intent = data.get("data", {}).get("object", {})
        
        return {
            "status": "processed",
            "action": "payment_intent_succeeded",
            "payment_intent_id": intent.get("id")
        }
    
    async def _handle_payment_intent_failed(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle failed payment intent"""
        intent = data.get("data", {}).get("object", {})
        
        return {
            "status": "processed",
            "action": "payment_intent_failed",
            "payment_intent_id": intent.get("id")
        }


class ExchangeWebhookHandler:
    """Handler for exchange webhooks (Binance, Coinbase, etc.)"""
    
    def __init__(self):
        self.email_service = get_email_service()
    
    async def handle_binance(self, event: WebhookEvent) -> Dict[str, Any]:
        """Handle Binance webhook"""
        payload = event.payload
        
        # Binance webhooks typically include:
        # - order updates
        # - account updates
        # - trade updates
        
        event_type = payload.get("e")  # Event type in Binance format
        
        if event_type == "executionReport":
            return await self._handle_order_update("binance", payload)
        elif event_type == "accountUpdate":
            return await self._handle_account_update("binance", payload)
        
        return {"status": "ignored", "reason": "Unknown event type"}
    
    async def handle_coinbase(self, event: WebhookEvent) -> Dict[str, Any]:
        """Handle Coinbase webhook"""
        payload = event.payload
        
        # Coinbase webhooks include:
        # - wallet:addresses:new-payment
        # - wallet:orders:create
        # - wallet:orders:charge:confirmed
        
        event_type = payload.get("type")
        
        if "payment" in event_type:
            return await self._handle_payment_notification("coinbase", payload)
        elif "order" in event_type:
            return await self._handle_order_notification("coinbase", payload)
        
        return {"status": "ignored", "reason": "Unknown event type"}
    
    async def _handle_order_update(
        self,
        exchange: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle order update from exchange"""
        # Process order fill, cancel, etc.
        
        return {
            "status": "processed",
            "action": "order_update",
            "exchange": exchange,
            "order_id": data.get("i"),
            "symbol": data.get("s"),
            "status": data.get("X")
        }
    
    async def _handle_account_update(
        self,
        exchange: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle account balance update"""
        
        return {
            "status": "processed",
            "action": "account_update",
            "exchange": exchange
        }
    
    async def _handle_payment_notification(
        self,
        exchange: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle payment notification"""
        
        return {
            "status": "processed",
            "action": "payment_notification",
            "exchange": exchange
        }
    
    async def _handle_order_notification(
        self,
        exchange: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle order notification"""
        
        return {
            "status": "processed",
            "action": "order_notification",
            "exchange": exchange
        }


class UserWebhookManager:
    """
    Manager for user-configured webhooks.
    
    Users can configure webhooks to receive events from TradeOS.
    """
    
    def __init__(self):
        self.endpoints: Dict[str, WebhookEndpoint] = {}
        self.event_handlers: Dict[str, List[Callable]] = defaultdict(list)
    
    def register_endpoint(
        self,
        user_id: str,
        url: str,
        events: List[str],
        secret: Optional[str] = None
    ) -> WebhookEndpoint:
        """
        Register a new webhook endpoint.
        
        Args:
            user_id: User ID
            url: Webhook URL
            events: List of events to subscribe to
            secret: Optional secret for signature verification
            
        Returns:
            Registered endpoint
        """
        import secrets
        
        endpoint_id = f"wh_{secrets.token_urlsafe(16)}"
        
        if not secret:
            secret = secrets.token_urlsafe(32)
        
        endpoint = WebhookEndpoint(
            id=endpoint_id,
            user_id=user_id,
            url=url,
            events=events,
            secret=secret,
            created_at=datetime.now(timezone.utc)
        )
        
        self.endpoints[endpoint_id] = endpoint
        
        return endpoint
    
    def unregister_endpoint(self, endpoint_id: str, user_id: str) -> bool:
        """Unregister a webhook endpoint"""
        endpoint = self.endpoints.get(endpoint_id)
        if not endpoint or endpoint.user_id != user_id:
            return False
        
        del self.endpoints[endpoint_id]
        return True
    
    def list_endpoints(self, user_id: str) -> List[WebhookEndpoint]:
        """List all webhook endpoints for a user"""
        return [
            e for e in self.endpoints.values()
            if e.user_id == user_id
        ]
    
    async def dispatch_event(
        self,
        event_type: str,
        payload: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Dispatch event to all subscribed webhooks.
        
        Args:
            event_type: Event type
            payload: Event payload
            user_id: Optional user ID to filter endpoints
            
        Returns:
            List of dispatch results
        """
        import httpx
        
        results = []
        
        for endpoint in self.endpoints.values():
            # Filter by user if specified
            if user_id and endpoint.user_id != user_id:
                continue
            
            # Check if endpoint should receive this event
            if not endpoint.should_trigger(event_type):
                continue
            
            # Prepare payload
            webhook_payload = {
                "event": event_type,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": payload
            }
            
            # Sign payload
            body = json.dumps(webhook_payload)
            signature = hmac.new(
                endpoint.secret.encode(),
                body.encode(),
                hashlib.sha256
            ).hexdigest()
            
            # Send webhook
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        endpoint.url,
                        content=body,
                        headers={
                            "Content-Type": "application/json",
                            "X-Webhook-Signature": f"sha256={signature}",
                            "X-Webhook-ID": endpoint.id,
                            "X-Event-Type": event_type
                        }
                    )
                
                endpoint.last_triggered = datetime.now(timezone.utc)
                
                results.append({
                    "endpoint_id": endpoint.id,
                    "status": "success",
                    "http_status": response.status_code
                })
                
            except Exception as e:
                endpoint.failure_count += 1
                
                # Deactivate if too many failures
                if endpoint.failure_count >= 10:
                    endpoint.is_active = False
                
                results.append({
                    "endpoint_id": endpoint.id,
                    "status": "failed",
                    "error": str(e)
                })
        
        return results


class WebhookHandler:
    """
    Central webhook handler for TradeOS.
    
    Routes webhooks to appropriate handlers based on source.
    """
    
    def __init__(self):
        self.stripe_handler = StripeWebhookHandler()
        self.exchange_handler = ExchangeWebhookHandler()
        self.user_webhooks = UserWebhookManager()
        self.verifier = WebhookSignatureVerifier()
    
    async def handle_stripe(
        self,
        request: Request,
        stripe_signature: str = Header(None, alias="Stripe-Signature")
    ) -> Dict[str, Any]:
        """
        Handle Stripe webhook.
        
        Usage in FastAPI:
            @app.post("/webhooks/stripe")
            async def stripe_webhook(request: Request):
                return await webhook_handler.handle_stripe(request)
        """
        body = await request.body()
        
        # Verify signature
        config = get_config()
        if not self.verifier.verify_stripe(
            body,
            stripe_signature,
            config.stripe.webhook_secret
        ):
            raise HTTPException(status_code=400, detail="Invalid signature")
        
        # Parse event
        payload = json.loads(body)
        
        event = WebhookEvent(
            source=WebhookSource.STRIPE,
            event_type=payload.get("type", ""),
            event_id=payload.get("id", ""),
            timestamp=datetime.now(timezone.utc),
            payload=payload,
            raw_body=body.decode(),
            signature=stripe_signature
        )
        
        # Handle event
        return await self.stripe_handler.handle(event)
    
    async def handle_binance(
        self,
        request: Request,
        x_mbx_apikey: Optional[str] = Header(None)
    ) -> Dict[str, Any]:
        """Handle Binance webhook"""
        body = await request.body()
        payload = json.loads(body)
        
        event = WebhookEvent(
            source=WebhookSource.BINANCE,
            event_type=payload.get("e", ""),
            event_id=payload.get("E", ""),
            timestamp=datetime.now(timezone.utc),
            payload=payload
        )
        
        return await self.exchange_handler.handle_binance(event)
    
    async def handle_coinbase(
        self,
        request: Request,
        x_cc_webhook_signature: Optional[str] = Header(None)
    ) -> Dict[str, Any]:
        """Handle Coinbase webhook"""
        body = await request.body()
        
        # TODO: Verify signature
        
        payload = json.loads(body)
        
        event = WebhookEvent(
            source=WebhookSource.COINBASE,
            event_type=payload.get("type", ""),
            event_id=payload.get("id", ""),
            timestamp=datetime.now(timezone.utc),
            payload=payload
        )
        
        return await self.exchange_handler.handle_coinbase(event)
    
    async def handle_custom(
        self,
        request: Request,
        source: str,
        signature: Optional[str] = Header(None, alias="X-Webhook-Signature")
    ) -> Dict[str, Any]:
        """Handle custom webhook"""
        body = await request.body()
        payload = json.loads(body)
        
        event = WebhookEvent(
            source=WebhookSource.CUSTOM,
            event_type=payload.get("event", ""),
            event_id=payload.get("id", ""),
            timestamp=datetime.now(timezone.utc),
            payload=payload,
            signature=signature
        )
        
        # Custom handling logic here
        return {
            "status": "received",
            "source": source,
            "event_type": event.event_type
        }
    
    async def dispatch_user_event(
        self,
        event_type: str,
        payload: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Dispatch event to user webhooks"""
        return await self.user_webhooks.dispatch_event(event_type, payload, user_id)


# Global webhook handler instance
_webhook_handler: Optional[WebhookHandler] = None


def get_webhook_handler() -> WebhookHandler:
    """Get or create global webhook handler"""
    global _webhook_handler
    if _webhook_handler is None:
        _webhook_handler = WebhookHandler()
    return _webhook_handler


# Convenience function for dispatching events
async def dispatch_event(
    event_type: str,
    payload: Dict[str, Any],
    user_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Dispatch event to user webhooks"""
    handler = get_webhook_handler()
    return await handler.dispatch_user_event(event_type, payload, user_id)
