"""
TradeOS Base Exchange Interface
===============================
Abstract base class for all exchange integrations.
Defines the contract that all exchange implementations must follow.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from datetime import datetime

from .models.order import (
    Order, OrderRequest, OrderFill, OrderState, OrderSide,
    OrderType, TimeInForce, OrderEvent, OrderEventType
)
from .latency_tracker import LatencyTracker, LatencyType, global_latency_tracker

logger = logging.getLogger(__name__)


class ExchangeError(Exception):
    """Base exception for exchange errors."""
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 retryable: bool = False):
        super().__init__(message)
        self.error_code = error_code
        self.retryable = retryable


class AuthenticationError(ExchangeError):
    """API authentication failed."""
    pass


class InsufficientFundsError(ExchangeError):
    """Not enough balance for order."""
    pass


class InvalidOrderError(ExchangeError):
    """Order parameters are invalid."""
    pass


class RateLimitError(ExchangeError):
    """Rate limit exceeded."""
    def __init__(self, message: str, retry_after: float = 1.0):
        super().__init__(message, "RATE_LIMIT", retryable=True)
        self.retry_after = retry_after


class NetworkError(ExchangeError):
    """Network connectivity issue."""
    def __init__(self, message: str):
        super().__init__(message, "NETWORK_ERROR", retryable=True)


class OrderNotFoundError(ExchangeError):
    """Order not found on exchange."""
    pass


@dataclass
class Balance:
    """Account balance for an asset."""
    asset: str
    free: Decimal
    locked: Decimal
    total: Decimal
    
    def __post_init__(self):
        if isinstance(self.free, (int, float, str)):
            self.free = Decimal(str(self.free))
        if isinstance(self.locked, (int, float, str)):
            self.locked = Decimal(str(self.locked))
        if isinstance(self.total, (int, float, str)):
            self.total = Decimal(str(self.total))


@dataclass
class Ticker:
    """Price ticker data."""
    symbol: str
    bid: Decimal
    ask: Decimal
    last: Decimal
    volume_24h: Decimal
    timestamp: datetime
    
    @property
    def spread(self) -> Decimal:
        """Calculate bid-ask spread."""
        return self.ask - self.bid
    
    @property
    def mid_price(self) -> Decimal:
        """Calculate mid price."""
        return (self.bid + self.ask) / 2


@dataclass
class OrderBookLevel:
    """Single level in order book."""
    price: Decimal
    quantity: Decimal


@dataclass
class OrderBook:
    """Order book snapshot."""
    symbol: str
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    timestamp: datetime
    
    def get_best_bid(self) -> Optional[OrderBookLevel]:
        """Get best bid price."""
        return self.bids[0] if self.bids else None
    
    def get_best_ask(self) -> Optional[OrderBookLevel]:
        """Get best ask price."""
        return self.asks[0] if self.asks else None
    
    def get_spread(self) -> Optional[Decimal]:
        """Get bid-ask spread."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if best_bid and best_ask:
            return best_ask.price - best_bid.price
        return None


@dataclass
class Position:
    """Trading position."""
    symbol: str
    side: str  # long, short
    quantity: Decimal
    entry_price: Decimal
    unrealized_pnl: Decimal
    margin: Optional[Decimal] = None
    leverage: Optional[Decimal] = None


class BaseExchange(ABC):
    """
    Abstract base class for all exchange implementations.
    
    All exchange connectors must implement these methods to ensure
    consistent behavior across different trading venues.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.is_connected = False
        self.latency_tracker = global_latency_tracker
        self._order_callbacks: List[Callable[[OrderEvent], None]] = []
        self._fill_callbacks: List[Callable[[OrderFill], None]] = []
        self._ticker_callbacks: List[Callable[[Ticker], None]] = []
        self._lock = asyncio.Lock()
        self._session: Optional[Any] = None
        self._ws_connection: Optional[Any] = None
        self._last_request_time = 0.0
        self._rate_limit_delay = 0.0
        
        logger.info(f"Initialized {name} exchange")
    
    # ==================== Connection Management ====================
    
    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to exchange.
        
        Returns:
            True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """
        Close connection to exchange.
        
        Returns:
            True if disconnection successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def is_connection_healthy(self) -> bool:
        """Check if connection is healthy."""
        pass
    
    # ==================== Order Operations ====================
    
    @abstractmethod
    async def place_order(self, order_request: OrderRequest) -> Order:
        """
        Place a new order on the exchange.
        
        Args:
            order_request: Order request details
            
        Returns:
            Order object with exchange-assigned ID
            
        Raises:
            AuthenticationError: If API credentials invalid
            InsufficientFundsError: If not enough balance
            InvalidOrderError: If order parameters invalid
            RateLimitError: If rate limit exceeded
            NetworkError: If network issue occurs
        """
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> bool:
        """
        Cancel an existing order.
        
        Args:
            order_id: Order ID to cancel
            symbol: Trading symbol (required by some exchanges)
            
        Returns:
            True if cancellation successful
            
        Raises:
            OrderNotFoundError: If order not found
            ExchangeError: If cancellation fails
        """
        pass
    
    @abstractmethod
    async def get_order(self, order_id: str, symbol: Optional[str] = None) -> Optional[Order]:
        """
        Get order details from exchange.
        
        Args:
            order_id: Order ID to query
            symbol: Trading symbol (required by some exchanges)
            
        Returns:
            Order object or None if not found
        """
        pass
    
    @abstractmethod
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """
        Get all open orders.
        
        Args:
            symbol: Filter by symbol (optional)
            
        Returns:
            List of open orders
        """
        pass
    
    @abstractmethod
    async def get_order_history(self, symbol: Optional[str] = None, 
                                 limit: int = 100) -> List[Order]:
        """
        Get order history.
        
        Args:
            symbol: Filter by symbol (optional)
            limit: Maximum number of orders to return
            
        Returns:
            List of historical orders
        """
        pass
    
    # ==================== Account Operations ====================
    
    @abstractmethod
    async def get_balance(self, asset: Optional[str] = None) -> Dict[str, Balance]:
        """
        Get account balance.
        
        Args:
            asset: Specific asset to query (None for all)
            
        Returns:
            Dictionary of asset -> Balance
        """
        pass
    
    @abstractmethod
    async def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """
        Get open positions (for margin/futures trading).
        
        Args:
            symbol: Filter by symbol (optional)
            
        Returns:
            List of positions
        """
        pass
    
    # ==================== Market Data ====================
    
    @abstractmethod
    async def get_ticker(self, symbol: str) -> Ticker:
        """
        Get current price ticker.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Ticker data
        """
        pass
    
    @abstractmethod
    async def get_orderbook(self, symbol: str, depth: int = 20) -> OrderBook:
        """
        Get order book snapshot.
        
        Args:
            symbol: Trading symbol
            depth: Number of levels to fetch
            
        Returns:
            Order book data
        """
        pass
    
    @abstractmethod
    async def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Dict]:
        """
        Get recent trades.
        
        Args:
            symbol: Trading symbol
            limit: Number of trades to fetch
            
        Returns:
            List of recent trades
        """
        pass
    
    # ==================== WebSocket Subscriptions ====================
    
    @abstractmethod
    async def subscribe_ticker(self, symbols: List[str]):
        """Subscribe to ticker updates via WebSocket."""
        pass
    
    @abstractmethod
    async def subscribe_orderbook(self, symbols: List[str], depth: int = 20):
        """Subscribe to order book updates via WebSocket."""
        pass
    
    @abstractmethod
    async def subscribe_trades(self, symbols: List[str]):
        """Subscribe to trade updates via WebSocket."""
        pass
    
    @abstractmethod
    async def subscribe_user_data(self):
        """Subscribe to user data (orders, fills, balances) via WebSocket."""
        pass
    
    # ==================== Utility Methods ====================
    
    def generate_signature(self, payload: str, secret: str) -> str:
        """Generate HMAC signature for API authentication."""
        return hmac.new(
            secret.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def generate_timestamp(self) -> int:
        """Generate current timestamp in milliseconds."""
        return int(time.time() * 1000)
    
    async def _apply_rate_limit(self):
        """Apply rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        
        if time_since_last < self._rate_limit_delay:
            await asyncio.sleep(self._rate_limit_delay - time_since_last)
        
        self._last_request_time = time.time()
    
    def _handle_error_response(self, response: Dict, status_code: int):
        """Handle error response from exchange."""
        error_msg = response.get('msg', response.get('message', 'Unknown error'))
        error_code = response.get('code', 'UNKNOWN')
        
        if status_code == 401 or status_code == 403:
            raise AuthenticationError(f"Authentication failed: {error_msg}", error_code)
        elif status_code == 429:
            raise RateLimitError(f"Rate limit exceeded: {error_msg}", retry_after=1.0)
        elif status_code >= 500:
            raise NetworkError(f"Exchange server error: {error_msg}")
        
        # Check for specific error codes
        if 'insufficient' in error_msg.lower() or 'balance' in error_msg.lower():
            raise InsufficientFundsError(error_msg, error_code)
        elif 'invalid' in error_msg.lower() or 'parameter' in error_msg.lower():
            raise InvalidOrderError(error_msg, error_code)
        elif 'not found' in error_msg.lower() or 'unknown' in error_msg.lower():
            raise OrderNotFoundError(error_msg, error_code)
        
        raise ExchangeError(error_msg, error_code)
    
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
    
    def register_ticker_callback(self, callback: Callable[[Ticker], None]):
        """Register callback for ticker updates."""
        self._ticker_callbacks.append(callback)
    
    def unregister_ticker_callback(self, callback: Callable[[Ticker], None]):
        """Unregister ticker callback."""
        if callback in self._ticker_callbacks:
            self._ticker_callbacks.remove(callback)
    
    async def _notify_order_update(self, event: OrderEvent):
        """Notify all registered order callbacks."""
        for callback in self._order_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(event))
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Error in order callback: {e}")
    
    async def _notify_fill(self, fill: OrderFill):
        """Notify all registered fill callbacks."""
        for callback in self._fill_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(fill))
                else:
                    callback(fill)
            except Exception as e:
                logger.error(f"Error in fill callback: {e}")
    
    async def _notify_ticker(self, ticker: Ticker):
        """Notify all registered ticker callbacks."""
        for callback in self._ticker_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(ticker))
                else:
                    callback(ticker)
            except Exception as e:
                logger.error(f"Error in ticker callback: {e}")
    
    # ==================== Helper Methods ====================
    
    def _map_order_type(self, order_type: OrderType) -> str:
        """Map internal order type to exchange-specific type."""
        mapping = {
            OrderType.MARKET: "MARKET",
            OrderType.LIMIT: "LIMIT",
            OrderType.STOP: "STOP_LOSS",
            OrderType.STOP_LIMIT: "STOP_LOSS_LIMIT",
            OrderType.TAKE_PROFIT: "TAKE_PROFIT",
            OrderType.TAKE_PROFIT_LIMIT: "TAKE_PROFIT_LIMIT",
        }
        return mapping.get(order_type, "MARKET")
    
    def _map_time_in_force(self, tif: TimeInForce) -> str:
        """Map internal time in force to exchange-specific value."""
        mapping = {
            TimeInForce.GTC: "GTC",
            TimeInForce.IOC: "IOC",
            TimeInForce.FOK: "FOK",
        }
        return mapping.get(tif, "GTC")
    
    def _map_order_side(self, side: OrderSide) -> str:
        """Map internal order side to exchange-specific value."""
        return "BUY" if side == OrderSide.BUY else "SELL"
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()


class ExchangeFactory:
    """Factory for creating exchange instances."""
    
    _exchanges: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str, exchange_class: type):
        """Register an exchange class."""
        cls._exchanges[name] = exchange_class
        logger.info(f"Registered exchange: {name}")
    
    @classmethod
    def create(cls, name: str, config: Dict[str, Any]) -> BaseExchange:
        """Create an exchange instance."""
        if name not in cls._exchanges:
            raise ValueError(f"Unknown exchange: {name}")
        
        exchange_class = cls._exchanges[name]
        return exchange_class(config)
    
    @classmethod
    def list_exchanges(cls) -> List[str]:
        """List registered exchanges."""
        return list(cls._exchanges.keys())
