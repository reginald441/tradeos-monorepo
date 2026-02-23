"""
TradeOS Order Manager
=====================
Central order lifecycle management system.
Handles order submission, tracking, partial fills, retries, and failover.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Set, Tuple

from .base_exchange import BaseExchange, ExchangeError, NetworkError, RateLimitError
from .models.order import (
    Order, OrderRequest, OrderFill, OrderState, OrderSide,
    OrderType, OrderEvent, OrderEventType
)
from .latency_tracker import LatencyType, global_latency_tracker

logger = logging.getLogger(__name__)


class OrderSubmissionStatus(Enum):
    """Status of order submission attempt."""
    PENDING = "pending"
    SUBMITTING = "submitting"
    SUBMITTED = "submitted"
    FAILED = "failed"
    RETRYING = "retrying"
    MAX_RETRIES_EXCEEDED = "max_retries_exceeded"


@dataclass
class OrderSubmission:
    """Tracks an order submission attempt."""
    order_request: OrderRequest
    exchange_name: str
    status: OrderSubmissionStatus = OrderSubmissionStatus.PENDING
    attempts: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_attempt: Optional[datetime] = None
    error_message: Optional[str] = None
    order: Optional[Order] = None
    retry_delays: List[float] = field(default_factory=lambda: [1.0, 2.0, 4.0])


@dataclass
class RetryPolicy:
    """Configuration for order retry behavior."""
    max_retries: int = 3
    retry_delay_base: float = 1.0
    retry_delay_max: float = 30.0
    retry_multiplier: float = 2.0
    retryable_errors: Set[str] = field(default_factory=lambda: {
        "NETWORK_ERROR", "RATE_LIMIT", "TIMEOUT", "SERVER_ERROR"
    })
    fail_over_exchanges: List[str] = field(default_factory=list)


class OrderManager:
    """
    Central order lifecycle manager.
    
    Features:
    - Order submission queue with retry logic
    - Partial fill handling
    - Failed order recovery
    - Multi-exchange failover
    - Order state synchronization
    - Batch order operations
    
    The OrderManager maintains a queue of pending orders and processes
    them asynchronously, handling retries and failover automatically.
    """
    
    def __init__(self, exchanges: Dict[str, BaseExchange],
                 retry_policy: Optional[RetryPolicy] = None):
        self.exchanges = exchanges
        self.retry_policy = retry_policy or RetryPolicy()
        
        # Order tracking
        self._orders: Dict[str, Order] = {}  # order_id -> Order
        self._submissions: Dict[str, OrderSubmission] = {}  # internal_id -> submission
        self._pending_queue: deque = deque()  # Pending submissions
        
        # Callbacks
        self._order_callbacks: List[Callable[[OrderEvent], None]] = []
        self._fill_callbacks: List[Callable[[OrderFill], None]] = []
        
        # Background tasks
        self._processing_task: Optional[asyncio.Task] = None
        self._sync_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Statistics
        self._stats = {
            "total_submitted": 0,
            "total_filled": 0,
            "total_cancelled": 0,
            "total_rejected": 0,
            "total_failed": 0,
            "total_retries": 0,
        }
        
        # Register callbacks on exchanges
        for exchange in exchanges.values():
            exchange.register_order_callback(self._on_order_update)
            exchange.register_fill_callback(self._on_fill)
        
        logger.info(f"OrderManager initialized with {len(exchanges)} exchanges")
    
    async def start(self):
        """Start the order manager."""
        if self._running:
            return
        
        self._running = True
        self._processing_task = asyncio.create_task(self._process_queue())
        self._sync_task = asyncio.create_task(self._sync_orders_loop())
        
        logger.info("OrderManager started")
    
    async def stop(self):
        """Stop the order manager."""
        self._running = False
        
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
        
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
        
        logger.info("OrderManager stopped")
    
    # ==================== Order Submission ====================
    
    async def submit_order(self, order_request: OrderRequest,
                          exchange_name: str,
                          priority: bool = False) -> OrderSubmission:
        """
        Submit an order for execution.
        
        Args:
            order_request: Order to submit
            exchange_name: Target exchange
            priority: If True, add to front of queue
            
        Returns:
            OrderSubmission tracking object
        """
        submission = OrderSubmission(
            order_request=order_request,
            exchange_name=exchange_name,
            max_retries=self.retry_policy.max_retries,
            retry_delays=self._calculate_retry_delays()
        )
        
        self._submissions[submission.order_request.client_order_id or str(id(submission))] = submission
        
        if priority:
            self._pending_queue.appendleft(submission)
        else:
            self._pending_queue.append(submission)
        
        self._stats["total_submitted"] += 1
        
        logger.info(
            f"Order queued: {order_request.symbol} {order_request.side.value} "
            f"{order_request.quantity} on {exchange_name}"
        )
        
        return submission
    
    async def submit_orders_batch(self, order_requests: List[Tuple[OrderRequest, str]]) -> List[OrderSubmission]:
        """
        Submit multiple orders as a batch.
        
        Args:
            order_requests: List of (order_request, exchange_name) tuples
            
        Returns:
            List of OrderSubmission tracking objects
        """
        submissions = []
        for order_request, exchange_name in order_requests:
            submission = await self.submit_order(order_request, exchange_name)
            submissions.append(submission)
        
        return submissions
    
    def _calculate_retry_delays(self) -> List[float]:
        """Calculate retry delays based on policy."""
        delays = []
        delay = self.retry_policy.retry_delay_base
        for _ in range(self.retry_policy.max_retries):
            delays.append(min(delay, self.retry_policy.retry_delay_max))
            delay *= self.retry_policy.retry_multiplier
        return delays
    
    async def _process_queue(self):
        """Process pending order submissions."""
        while self._running:
            try:
                if self._pending_queue:
                    submission = self._pending_queue.popleft()
                    await self._process_submission(submission)
                else:
                    await asyncio.sleep(0.1)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing order queue: {e}")
                await asyncio.sleep(1)
    
    async def _process_submission(self, submission: OrderSubmission):
        """Process a single order submission."""
        submission.status = OrderSubmissionStatus.SUBMITTING
        submission.attempts += 1
        submission.last_attempt = datetime.utcnow()
        
        exchange = self.exchanges.get(submission.exchange_name)
        if not exchange:
            submission.status = OrderSubmissionStatus.FAILED
            submission.error_message = f"Exchange {submission.exchange_name} not found"
            await self._handle_submission_failure(submission)
            return
        
        try:
            # Submit order to exchange
            order = await exchange.place_order(submission.order_request)
            
            submission.order = order
            submission.status = OrderSubmissionStatus.SUBMITTED
            self._orders[order.order_id] = order
            
            logger.info(f"Order submitted successfully: {order.order_id}")
            
            # Notify
            await self._notify_order_update(OrderEvent(
                event_type=OrderEventType.SUBMITTED,
                order_id=order.order_id,
                data=order.to_dict()
            ))
            
        except Exception as e:
            submission.error_message = str(e)
            
            # Check if error is retryable
            if self._is_retryable_error(e) and submission.attempts <= submission.max_retries:
                submission.status = OrderSubmissionStatus.RETRYING
                self._stats["total_retries"] += 1
                
                # Calculate retry delay
                delay_idx = min(submission.attempts - 1, len(submission.retry_delays) - 1)
                delay = submission.retry_delays[delay_idx]
                
                logger.warning(
                    f"Order submission failed, retrying in {delay}s: {e}"
                )
                
                await asyncio.sleep(delay)
                self._pending_queue.append(submission)
                
            else:
                submission.status = OrderSubmissionStatus.FAILED
                await self._handle_submission_failure(submission)
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if an error is retryable."""
        if isinstance(error, RateLimitError):
            return True
        if isinstance(error, NetworkError):
            return True
        if isinstance(error, ExchangeError):
            return error.error_code in self.retry_policy.retryable_errors
        return False
    
    async def _handle_submission_failure(self, submission: OrderSubmission):
        """Handle a failed order submission."""
        self._stats["total_failed"] += 1
        
        logger.error(
            f"Order submission failed after {submission.attempts} attempts: "
            f"{submission.error_message}"
        )
        
        # Try failover exchanges if configured
        if self.retry_policy.fail_over_exchanges:
            for failover_exchange in self.retry_policy.fail_over_exchanges:
                if failover_exchange != submission.exchange_name:
                    logger.info(f"Attempting failover to {failover_exchange}")
                    submission.exchange_name = failover_exchange
                    submission.attempts = 0
                    submission.status = OrderSubmissionStatus.PENDING
                    self._pending_queue.append(submission)
                    return
        
        # Notify failure
        await self._notify_order_update(OrderEvent(
            event_type=OrderEventType.ERROR,
            order_id=submission.order_request.client_order_id or "unknown",
            error_message=submission.error_message,
            data={"exchange": submission.exchange_name}
        ))
    
    # ==================== Order Management ====================
    
    async def cancel_order(self, order_id: str, 
                          exchange_name: Optional[str] = None) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID to cancel
            exchange_name: Exchange name (optional, will be looked up)
            
        Returns:
            True if cancellation successful
        """
        order = self._orders.get(order_id)
        if not order:
            logger.warning(f"Cancel requested for unknown order: {order_id}")
            return False
        
        exchange_name = exchange_name or order.exchange
        exchange = self.exchanges.get(exchange_name)
        
        if not exchange:
            logger.error(f"Exchange {exchange_name} not found for cancellation")
            return False
        
        try:
            success = await exchange.cancel_order(order_id, order.symbol)
            
            if success:
                order.update_state(OrderState.CANCELLED, "User cancelled")
                self._stats["total_cancelled"] += 1
                
                await self._notify_order_update(OrderEvent(
                    event_type=OrderEventType.CANCELLED,
                    order_id=order_id,
                    data=order.to_dict()
                ))
            
            return success
            
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    async def cancel_all_orders(self, symbol: Optional[str] = None,
                                 exchange_name: Optional[str] = None) -> int:
        """
        Cancel all open orders.
        
        Args:
            symbol: Filter by symbol (optional)
            exchange_name: Filter by exchange (optional)
            
        Returns:
            Number of orders cancelled
        """
        cancelled_count = 0
        
        for order in list(self._orders.values()):
            if not order.is_active:
                continue
            
            if symbol and order.symbol != symbol:
                continue
            
            if exchange_name and order.exchange != exchange_name:
                continue
            
            if await self.cancel_order(order.order_id):
                cancelled_count += 1
        
        return cancelled_count
    
    async def modify_order(self, order_id: str,
                          new_price: Optional[Decimal] = None,
                          new_quantity: Optional[Decimal] = None) -> Optional[Order]:
        """
        Modify an existing order (cancel and replace).
        
        Args:
            order_id: Order ID to modify
            new_price: New price (optional)
            new_quantity: New quantity (optional)
            
        Returns:
            New Order if successful
        """
        order = self._orders.get(order_id)
        if not order:
            return None
        
        if not order.is_active:
            logger.warning(f"Cannot modify inactive order: {order_id}")
            return None
        
        # Cancel existing order
        if not await self.cancel_order(order_id):
            return None
        
        # Create new order request
        new_request = OrderRequest(
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            quantity=new_quantity or order.quantity,
            price=new_price or order.price,
            stop_price=order.stop_price,
            time_in_force=order.time_in_force,
            client_order_id=f"{order.client_order_id}_modified"
        )
        
        # Submit new order
        submission = await self.submit_order(new_request, order.exchange, priority=True)
        
        # Wait for submission to complete
        while submission.status in (OrderSubmissionStatus.PENDING, OrderSubmissionStatus.SUBMITTING, OrderSubmissionStatus.RETRYING):
            await asyncio.sleep(0.1)
        
        return submission.order
    
    # ==================== Order Synchronization ====================
    
    async def _sync_orders_loop(self):
        """Periodically sync order states from exchanges."""
        while self._running:
            try:
                await asyncio.sleep(5)  # Sync every 5 seconds
                await self._sync_orders()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error syncing orders: {e}")
    
    async def _sync_orders(self):
        """Sync order states from exchanges."""
        # Group orders by exchange
        orders_by_exchange: Dict[str, List[Order]] = {}
        for order in self._orders.values():
            if order.is_active:
                if order.exchange not in orders_by_exchange:
                    orders_by_exchange[order.exchange] = []
                orders_by_exchange[order.exchange].append(order)
        
        # Sync each exchange
        for exchange_name, orders in orders_by_exchange.items():
            exchange = self.exchanges.get(exchange_name)
            if not exchange:
                continue
            
            try:
                # Fetch open orders from exchange
                open_orders = await exchange.get_open_orders()
                open_order_ids = {o.order_id for o in open_orders}
                
                for order in orders:
                    if order.order_id not in open_order_ids:
                        # Order not open on exchange, fetch details
                        updated_order = await exchange.get_order(order.order_id, order.symbol)
                        if updated_order:
                            old_state = order.state
                            order.state = updated_order.state
                            order.filled_quantity = updated_order.filled_quantity
                            order.remaining_quantity = updated_order.remaining_quantity
                            order.average_fill_price = updated_order.average_fill_price
                            
                            if old_state != order.state:
                                logger.info(f"Order {order.order_id} state synced: {old_state.value} -> {order.state.value}")
                                
                                await self._notify_order_update(OrderEvent(
                                    event_type=OrderEventType.UPDATE,
                                    order_id=order.order_id,
                                    data=order.to_dict()
                                ))
                                
            except Exception as e:
                logger.error(f"Error syncing orders from {exchange_name}: {e}")
    
    # ==================== Event Handlers ====================
    
    async def _on_order_update(self, event: OrderEvent):
        """Handle order update from exchange."""
        order = self._orders.get(event.order_id)
        if order:
            # Update local order state
            if event.data:
                old_state = order.state
                # Parse updated state from event data
                # This would update the order object
                
                if old_state != order.state:
                    logger.debug(f"Order {order.order_id} updated: {old_state.value} -> {order.state.value}")
        
        # Forward to registered callbacks
        await self._notify_order_update(event)
    
    async def _on_fill(self, fill: OrderFill):
        """Handle fill notification from exchange."""
        order = self._orders.get(fill.order_id)
        if order:
            # Update order with fill
            order.add_fill(fill)
            
            if order.state == OrderState.FILLED:
                self._stats["total_filled"] += 1
        
        # Forward to registered callbacks
        await self._notify_fill(fill)
    
    async def _notify_order_update(self, event: OrderEvent):
        """Notify order update callbacks."""
        for callback in self._order_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(event))
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Error in order callback: {e}")
    
    async def _notify_fill(self, fill: OrderFill):
        """Notify fill callbacks."""
        for callback in self._fill_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(fill))
                else:
                    callback(fill)
            except Exception as e:
                logger.error(f"Error in fill callback: {e}")
    
    # ==================== Callback Registration ====================
    
    def register_order_callback(self, callback: Callable[[OrderEvent], None]):
        """Register callback for order updates."""
        self._order_callbacks.append(callback)
    
    def unregister_order_callback(self, callback: Callable[[OrderEvent], None]):
        """Unregister order callback."""
        if callback in self._order_callbacks:
            self._order_callbacks.remove(callback)
    
    def register_fill_callback(self, callback: Callable[[OrderFill], None]):
        """Register callback for fill updates."""
        self._fill_callbacks.append(callback)
    
    def unregister_fill_callback(self, callback: Callable[[OrderFill], None]):
        """Unregister fill callback."""
        if callback in self._fill_callbacks:
            self._fill_callbacks.remove(callback)
    
    # ==================== Queries ====================
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self._orders.get(order_id)
    
    def get_open_orders(self, symbol: Optional[str] = None,
                        exchange_name: Optional[str] = None) -> List[Order]:
        """Get all open orders."""
        orders = [o for o in self._orders.values() if o.is_active]
        
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        
        if exchange_name:
            orders = [o for o in orders if o.exchange == exchange_name]
        
        return orders
    
    def get_order_history(self, limit: int = 100) -> List[Order]:
        """Get order history."""
        completed = [o for o in self._orders.values() if o.is_complete]
        return sorted(completed, key=lambda o: o.updated_at, reverse=True)[:limit]
    
    def get_stats(self) -> Dict:
        """Get order manager statistics."""
        return {
            **self._stats,
            "active_orders": len(self.get_open_orders()),
            "total_tracked_orders": len(self._orders),
            "pending_submissions": len(self._pending_queue),
        }
    
    # ==================== WebSocket (Delegate to exchanges) ====================
    
    async def subscribe_ticker(self, symbols: List[str], exchange_name: str):
        """Subscribe to ticker updates."""
        exchange = self.exchanges.get(exchange_name)
        if exchange:
            await exchange.subscribe_ticker(symbols)
    
    async def subscribe_orderbook(self, symbols: List[str], exchange_name: str, depth: int = 20):
        """Subscribe to order book updates."""
        exchange = self.exchanges.get(exchange_name)
        if exchange:
            await exchange.subscribe_orderbook(symbols, depth)
    
    async def subscribe_user_data(self, exchange_name: str):
        """Subscribe to user data."""
        exchange = self.exchanges.get(exchange_name)
        if exchange:
            await exchange.subscribe_user_data()
