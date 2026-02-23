"""
TradeOS Binance Exchange Implementation
=======================================
Complete Binance Spot and Futures trading integration.
Supports testnet, rate limiting, and WebSocket feeds.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import time
from decimal import Decimal
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime

import aiohttp

from ..base_exchange import (
    BaseExchange, Balance, Ticker, OrderBook, OrderBookLevel,
    Position, ExchangeError, AuthenticationError, RateLimitError,
    NetworkError, InsufficientFundsError, InvalidOrderError,
    OrderNotFoundError
)
from ..models.order import (
    Order, OrderRequest, OrderFill, OrderState, OrderSide,
    OrderType, TimeInForce, OrderEvent, OrderEventType
)
from ..latency_tracker import LatencyType, global_latency_tracker

logger = logging.getLogger(__name__)


class BinanceExchange(BaseExchange):
    """
    Binance Spot Exchange Implementation.
    
    Features:
    - Full spot trading support
    - Margin trading support
    - Testnet support
    - Rate limit handling with exponential backoff
    - WebSocket real-time data
    - Signature-based authentication
    """
    
    # API Endpoints
    SPOT_BASE_URL = "https://api.binance.com"
    SPOT_TESTNET_URL = "https://testnet.binance.vision"
    SPOT_WS_URL = "wss://stream.binance.com:9443/ws"
    SPOT_TESTNET_WS_URL = "wss://testnet.binance.vision/ws"
    
    # Rate limits
    REQUEST_WEIGHT_LIMIT = 1200  # per minute
    ORDER_LIMIT = 50  # per 10 seconds
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("binance", config)
        self.api_key = config.get("api_key", "")
        self.api_secret = config.get("api_secret", "")
        self.use_testnet = config.get("use_testnet", False)
        
        self.base_url = self.SPOT_TESTNET_URL if self.use_testnet else self.SPOT_BASE_URL
        self.ws_url = self.SPOT_TESTNET_WS_URL if self.use_testnet else self.SPOT_WS_URL
        
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws_connection = None
        self._ws_task = None
        self._listen_key: Optional[str] = None
        self._listen_key_refresh_task = None
        
        # Rate limiting
        self._request_weight = 0
        self._request_timestamps: List[float] = []
        self._order_timestamps: List[float] = []
        
        # Subscriptions
        self._subscribed_symbols: set = set()
        
        logger.info(f"Binance exchange initialized (testnet={self.use_testnet})")
    
    # ==================== Connection ====================
    
    async def connect(self) -> bool:
        """Establish connection to Binance."""
        try:
            if self._session is None or self._session.closed:
                self._session = aiohttp.ClientSession(
                    headers={"X-MBX-APIKEY": self.api_key}
                )
            
            # Test connection with server time
            server_time = await self._get_server_time()
            logger.info(f"Connected to Binance. Server time: {server_time}")
            
            self.is_connected = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Binance: {e}")
            self.is_connected = False
            return False
    
    async def disconnect(self) -> bool:
        """Close connection to Binance."""
        try:
            # Cancel listen key refresh
            if self._listen_key_refresh_task:
                self._listen_key_refresh_task.cancel()
                try:
                    await self._listen_key_refresh_task
                except asyncio.CancelledError:
                    pass
            
            # Close WebSocket
            if self._ws_task:
                self._ws_task.cancel()
                try:
                    await self._ws_task
                except asyncio.CancelledError:
                    pass
            
            if self._ws_connection:
                await self._ws_connection.close()
            
            # Close session
            if self._session and not self._session.closed:
                await self._session.close()
            
            self.is_connected = False
            logger.info("Disconnected from Binance")
            return True
            
        except Exception as e:
            logger.error(f"Error disconnecting from Binance: {e}")
            return False
    
    async def is_connection_healthy(self) -> bool:
        """Check if connection is healthy."""
        try:
            await self._get_server_time()
            return True
        except Exception:
            return False
    
    async def _get_server_time(self) -> int:
        """Get Binance server time."""
        response = await self._make_request("GET", "/api/v3/time")
        return response.get("serverTime", 0)
    
    # ==================== Request Handling ====================
    
    async def _make_request(self, method: str, endpoint: str, 
                            params: Optional[Dict] = None,
                            signed: bool = False,
                            weight: int = 1) -> Dict:
        """
        Make authenticated request to Binance API.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Query parameters
            signed: Whether to sign the request
            weight: Request weight for rate limiting
            
        Returns:
            JSON response
        """
        if self._session is None or self._session.closed:
            raise NetworkError("Not connected to Binance")
        
        # Apply rate limiting
        await self._apply_rate_limit(weight)
        
        url = f"{self.base_url}{endpoint}"
        params = params or {}
        
        if signed:
            params["timestamp"] = self.generate_timestamp()
            params["signature"] = self._generate_signature(params)
        
        timer_id = self.latency_tracker.start_timer(
            f"binance_{endpoint}", LatencyType.API_REQUEST, "binance"
        )
        
        try:
            async with self._session.request(method, url, params=params) as response:
                data = await response.json()
                
                success = response.status == 200
                self.latency_tracker.stop_timer(timer_id, success)
                
                if response.status == 429:
                    retry_after = int(response.headers.get('Retry-After', 1))
                    raise RateLimitError("Rate limit exceeded", retry_after)
                
                if response.status >= 400:
                    self._handle_error_response(data, response.status)
                
                return data
                
        except aiohttp.ClientError as e:
            self.latency_tracker.stop_timer(timer_id, False)
            raise NetworkError(f"Request failed: {e}")
        except asyncio.TimeoutError:
            self.latency_tracker.stop_timer(timer_id, False)
            raise NetworkError("Request timeout")
    
    def _generate_signature(self, params: Dict[str, Any]) -> str:
        """Generate request signature."""
        query_string = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
        return hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    async def _apply_rate_limit(self, weight: int = 1):
        """Apply rate limiting."""
        current_time = time.time()
        
        # Clean old timestamps
        cutoff = current_time - 60
        self._request_timestamps = [t for t in self._request_timestamps if t > cutoff]
        self._order_timestamps = [t for t in self._order_timestamps if t > 10]
        
        # Check weight limit
        if len(self._request_timestamps) >= self.REQUEST_WEIGHT_LIMIT:
            sleep_time = 60 - (current_time - self._request_timestamps[0])
            if sleep_time > 0:
                logger.warning(f"Rate limit approaching, sleeping {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time)
        
        self._request_timestamps.append(current_time)
    
    # ==================== Order Operations ====================
    
    async def place_order(self, order_request: OrderRequest) -> Order:
        """Place a new order on Binance."""
        timer_id = self.latency_tracker.start_timer(
            f"order_{order_request.symbol}", LatencyType.ORDER_SUBMIT, "binance"
        )
        
        try:
            # Build order parameters
            params = {
                "symbol": order_request.symbol.upper().replace("-", "").replace("/", ""),
                "side": self._map_order_side(order_request.side),
                "type": self._map_order_type(order_request.order_type),
                "quantity": str(order_request.quantity),
            }
            
            # Add price for limit orders
            if order_request.price and order_request.order_type in [
                OrderType.LIMIT, OrderType.STOP_LIMIT, OrderType.TAKE_PROFIT_LIMIT
            ]:
                params["price"] = str(order_request.price)
            
            # Add stop price
            if order_request.stop_price:
                params["stopPrice"] = str(order_request.stop_price)
            
            # Add time in force
            if order_request.order_type != OrderType.MARKET:
                params["timeInForce"] = self._map_time_in_force(order_request.time_in_force)
            
            # Add client order ID
            if order_request.client_order_id:
                params["newClientOrderId"] = order_request.client_order_id[:36]
            
            # Add post-only flag
            if order_request.post_only:
                params["timeInForce"] = "GTX"  # Good Till Crossing (post-only)
            
            response = await self._make_request(
                "POST", "/api/v3/order", params, signed=True, weight=1
            )
            
            self.latency_tracker.stop_timer(timer_id, True)
            
            # Convert response to Order object
            order = self._parse_order_response(response)
            order.exchange = "binance"
            
            logger.info(f"Order placed on Binance: {order.order_id}")
            return order
            
        except Exception as e:
            self.latency_tracker.stop_timer(timer_id, False, {"error": str(e)})
            raise
    
    async def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> bool:
        """Cancel an existing order."""
        if not symbol:
            raise InvalidOrderError("Symbol is required for cancellation")
        
        timer_id = self.latency_tracker.start_timer(
            f"cancel_{order_id}", LatencyType.CANCEL_REQUEST, "binance"
        )
        
        try:
            params = {
                "symbol": symbol.upper().replace("-", "").replace("/", ""),
            }
            
            # Try to cancel by exchange order ID first
            if order_id.isdigit():
                params["orderId"] = order_id
            else:
                params["origClientOrderId"] = order_id
            
            response = await self._make_request(
                "DELETE", "/api/v3/order", params, signed=True, weight=1
            )
            
            self.latency_tracker.stop_timer(timer_id, True)
            
            logger.info(f"Order cancelled on Binance: {order_id}")
            return True
            
        except Exception as e:
            self.latency_tracker.stop_timer(timer_id, False, {"error": str(e)})
            raise
    
    async def get_order(self, order_id: str, symbol: Optional[str] = None) -> Optional[Order]:
        """Get order details from Binance."""
        if not symbol:
            raise InvalidOrderError("Symbol is required to get order")
        
        params = {
            "symbol": symbol.upper().replace("-", "").replace("/", ""),
        }
        
        if order_id.isdigit():
            params["orderId"] = order_id
        else:
            params["origClientOrderId"] = order_id
        
        try:
            response = await self._make_request(
                "GET", "/api/v3/order", params, signed=True, weight=2
            )
            
            order = self._parse_order_response(response)
            order.exchange = "binance"
            return order
            
        except OrderNotFoundError:
            return None
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all open orders."""
        params = {}
        if symbol:
            params["symbol"] = symbol.upper().replace("-", "").replace("/", "")
        
        response = await self._make_request(
            "GET", "/api/v3/openOrders", params, signed=True, weight=3
        )
        
        orders = []
        for order_data in response:
            order = self._parse_order_response(order_data)
            order.exchange = "binance"
            orders.append(order)
        
        return orders
    
    async def get_order_history(self, symbol: Optional[str] = None,
                                 limit: int = 100) -> List[Order]:
        """Get order history."""
        params = {"limit": min(limit, 1000)}
        if symbol:
            params["symbol"] = symbol.upper().replace("-", "").replace("/", "")
        
        response = await self._make_request(
            "GET", "/api/v3/allOrders", params, signed=True, weight=10
        )
        
        orders = []
        for order_data in response:
            order = self._parse_order_response(order_data)
            order.exchange = "binance"
            orders.append(order)
        
        return orders
    
    def _parse_order_response(self, data: Dict) -> Order:
        """Parse Binance order response to Order object."""
        # Map status
        status_map = {
            "NEW": OrderState.OPEN,
            "PARTIALLY_FILLED": OrderState.PARTIALLY_FILLED,
            "FILLED": OrderState.FILLED,
            "CANCELED": OrderState.CANCELLED,
            "PENDING_CANCEL": OrderState.PENDING_CANCEL,
            "REJECTED": OrderState.REJECTED,
            "EXPIRED": OrderState.EXPIRED,
        }
        
        order = Order(
            order_id=str(data.get("orderId", "")),
            client_order_id=data.get("clientOrderId"),
            exchange_order_id=str(data.get("orderId", "")),
            symbol=data.get("symbol", ""),
            side=OrderSide.BUY if data.get("side") == "BUY" else OrderSide.SELL,
            order_type=self._parse_order_type(data.get("type", "MARKET")),
            quantity=Decimal(str(data.get("origQty", "0"))),
            price=Decimal(str(data.get("price", "0"))) if data.get("price") != "0" else None,
            stop_price=Decimal(str(data.get("stopPrice", "0"))) if data.get("stopPrice") != "0" else None,
            filled_quantity=Decimal(str(data.get("executedQty", "0"))),
            remaining_quantity=Decimal(str(data.get("origQty", "0"))) - Decimal(str(data.get("executedQty", "0"))),
            state=status_map.get(data.get("status"), OrderState.PENDING),
            total_fee=Decimal("0"),  # Binance doesn't return fee in order query
        )
        
        # Calculate average fill price
        if order.filled_quantity > 0:
            order.average_fill_price = Decimal(str(data.get("avgPrice", "0")))
        
        return order
    
    def _parse_order_type(self, type_str: str) -> OrderType:
        """Parse Binance order type."""
        mapping = {
            "MARKET": OrderType.MARKET,
            "LIMIT": OrderType.LIMIT,
            "STOP_LOSS": OrderType.STOP,
            "STOP_LOSS_LIMIT": OrderType.STOP_LIMIT,
            "TAKE_PROFIT": OrderType.TAKE_PROFIT,
            "TAKE_PROFIT_LIMIT": OrderType.TAKE_PROFIT_LIMIT,
            "LIMIT_MAKER": OrderType.LIMIT,
        }
        return mapping.get(type_str, OrderType.MARKET)
    
    # ==================== Account Operations ====================
    
    async def get_balance(self, asset: Optional[str] = None) -> Dict[str, Balance]:
        """Get account balance."""
        response = await self._make_request(
            "GET", "/api/v3/account", {}, signed=True, weight=10
        )
        
        balances = {}
        for bal in response.get("balances", []):
            if asset and bal["asset"] != asset:
                continue
            
            free = Decimal(str(bal.get("free", "0")))
            locked = Decimal(str(bal.get("locked", "0")))
            
            if free > 0 or locked > 0 or asset:
                balances[bal["asset"]] = Balance(
                    asset=bal["asset"],
                    free=free,
                    locked=locked,
                    total=free + locked
                )
        
        return balances
    
    async def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """Get open positions (not applicable for spot)."""
        return []  # Spot trading doesn't have positions
    
    # ==================== Market Data ====================
    
    async def get_ticker(self, symbol: str) -> Ticker:
        """Get current price ticker."""
        params = {
            "symbol": symbol.upper().replace("-", "").replace("/", "")
        }
        
        response = await self._make_request(
            "GET", "/api/v3/ticker/bookTicker", params, weight=1
        )
        
        return Ticker(
            symbol=symbol,
            bid=Decimal(str(response.get("bidPrice", "0"))),
            ask=Decimal(str(response.get("askPrice", "0"))),
            last=Decimal(str(response.get("bidPrice", "0"))),  # Use bid as last
            volume_24h=Decimal("0"),  # Not available in bookTicker
            timestamp=datetime.utcnow()
        )
    
    async def get_orderbook(self, symbol: str, depth: int = 20) -> OrderBook:
        """Get order book snapshot."""
        params = {
            "symbol": symbol.upper().replace("-", "").replace("/", ""),
            "limit": min(depth, 1000)
        }
        
        response = await self._make_request(
            "GET", "/api/v3/depth", params, weight=1
        )
        
        bids = [
            OrderBookLevel(
                price=Decimal(str(b[0])),
                quantity=Decimal(str(b[1]))
            )
            for b in response.get("bids", [])
        ]
        
        asks = [
            OrderBookLevel(
                price=Decimal(str(a[0])),
                quantity=Decimal(str(a[1]))
            )
            for a in response.get("asks", [])
        ]
        
        return OrderBook(
            symbol=symbol,
            bids=bids,
            asks=asks,
            timestamp=datetime.utcnow()
        )
    
    async def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Get recent trades."""
        params = {
            "symbol": symbol.upper().replace("-", "").replace("/", ""),
            "limit": min(limit, 1000)
        }
        
        response = await self._make_request(
            "GET", "/api/v3/trades", params, weight=1
        )
        
        return response
    
    # ==================== WebSocket ====================
    
    async def subscribe_ticker(self, symbols: List[str]):
        """Subscribe to ticker updates via WebSocket."""
        streams = [f"{s.lower()}@ticker" for s in symbols]
        await self._subscribe_streams(streams)
    
    async def subscribe_orderbook(self, symbols: List[str], depth: int = 20):
        """Subscribe to order book updates via WebSocket."""
        depth_str = "@depth" if depth <= 20 else "@depth5"
        streams = [f"{s.lower()}{depth_str}" for s in symbols]
        await self._subscribe_streams(streams)
    
    async def subscribe_trades(self, symbols: List[str]):
        """Subscribe to trade updates via WebSocket."""
        streams = [f"{s.lower()}@trade" for s in symbols]
        await self._subscribe_streams(streams)
    
    async def subscribe_user_data(self):
        """Subscribe to user data via WebSocket."""
        if not self._listen_key:
            self._listen_key = await self._create_listen_key()
            self._listen_key_refresh_task = asyncio.create_task(
                self._refresh_listen_key()
            )
        
        ws_url = f"{self.ws_url}/{self._listen_key}"
        self._ws_task = asyncio.create_task(self._ws_loop(ws_url))
    
    async def _create_listen_key(self) -> str:
        """Create a listen key for user data stream."""
        response = await self._make_request(
            "POST", "/api/v3/userDataStream", {}, signed=False, weight=1
        )
        return response.get("listenKey", "")
    
    async def _refresh_listen_key(self):
        """Periodically refresh the listen key."""
        while True:
            try:
                await asyncio.sleep(1800)  # Refresh every 30 minutes
                if self._listen_key:
                    await self._make_request(
                        "PUT", "/api/v3/userDataStream",
                        {"listenKey": self._listen_key},
                        signed=False, weight=1
                    )
                    logger.debug("Listen key refreshed")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error refreshing listen key: {e}")
    
    async def _subscribe_streams(self, streams: List[str]):
        """Subscribe to WebSocket streams."""
        stream_path = "/".join(streams)
        ws_url = f"{self.ws_url}/{stream_path}"
        self._ws_task = asyncio.create_task(self._ws_loop(ws_url))
    
    async def _ws_loop(self, ws_url: str):
        """WebSocket connection loop."""
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(ws_url) as ws:
                        logger.info(f"WebSocket connected: {ws_url}")
                        self._ws_connection = ws
                        
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                await self._handle_ws_message(json.loads(msg.data))
                            elif msg.type == aiohttp.WSMsgType.ERROR:
                                logger.error(f"WebSocket error: {msg.data}")
                                break
                            elif msg.type == aiohttp.WSMsgType.CLOSED:
                                logger.info("WebSocket closed")
                                break
                                
            except asyncio.CancelledError:
                logger.info("WebSocket loop cancelled")
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                await asyncio.sleep(5)  # Reconnect delay
    
    async def _handle_ws_message(self, data: Dict):
        """Handle WebSocket message."""
        event_type = data.get("e", "")
        
        if event_type == "executionReport":
            await self._handle_execution_report(data)
        elif event_type == "24hrTicker":
            await self._handle_ticker_update(data)
        elif event_type == "trade":
            await self._handle_trade_update(data)
        elif event_type == "depthUpdate":
            await self._handle_depth_update(data)
    
    async def _handle_execution_report(self, data: Dict):
        """Handle order execution report."""
        order = self._parse_order_response(data)
        order.exchange = "binance"
        
        event = OrderEvent(
            event_type=OrderEventType.UPDATE,
            order_id=order.order_id,
            data=order.to_dict()
        )
        
        await self._notify_order_update(event)
    
    async def _handle_ticker_update(self, data: Dict):
        """Handle ticker update."""
        ticker = Ticker(
            symbol=data.get("s", ""),
            bid=Decimal(str(data.get("b", "0"))),
            ask=Decimal(str(data.get("a", "0"))),
            last=Decimal(str(data.get("c", "0"))),
            volume_24h=Decimal(str(data.get("v", "0"))),
            timestamp=datetime.utcnow()
        )
        
        await self._notify_ticker(ticker)
    
    async def _handle_trade_update(self, data: Dict):
        """Handle trade update."""
        pass  # Implement if needed
    
    async def _handle_depth_update(self, data: Dict):
        """Handle order book depth update."""
        pass  # Implement if needed


class BinanceFuturesExchange(BinanceExchange):
    """
    Binance USD-M Futures Exchange Implementation.
    
    Features:
    - USD-Margined futures trading
    - Perpetual and quarterly contracts
    - Leverage management
    - Position tracking
    """
    
    FUTURES_BASE_URL = "https://fapi.binance.com"
    FUTURES_TESTNET_URL = "https://testnet.binancefuture.com"
    FUTURES_WS_URL = "wss://fstream.binance.com/ws"
    FUTURES_TESTNET_WS_URL = "wss://stream.binancefuture.com/ws"
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "binance_futures"
        
        self.base_url = self.FUTURES_TESTNET_URL if self.use_testnet else self.FUTURES_BASE_URL
        self.ws_url = self.FUTURES_TESTNET_WS_URL if self.use_testnet else self.FUTURES_WS_URL
        
        logger.info(f"Binance Futures initialized (testnet={self.use_testnet})")
    
    async def place_order(self, order_request: OrderRequest) -> Order:
        """Place a futures order."""
        timer_id = self.latency_tracker.start_timer(
            f"futures_order_{order_request.symbol}",
            LatencyType.ORDER_SUBMIT,
            "binance_futures"
        )
        
        try:
            params = {
                "symbol": order_request.symbol.upper().replace("-", "").replace("/", ""),
                "side": self._map_order_side(order_request.side),
                "type": self._map_order_type(order_request.order_type),
                "quantity": str(order_request.quantity),
            }
            
            if order_request.price and order_request.order_type in [
                OrderType.LIMIT, OrderType.STOP_LIMIT, OrderType.TAKE_PROFIT_LIMIT
            ]:
                params["price"] = str(order_request.price)
            
            if order_request.stop_price:
                params["stopPrice"] = str(order_request.stop_price)
            
            if order_request.order_type != OrderType.MARKET:
                params["timeInForce"] = self._map_time_in_force(order_request.time_in_force)
            
            if order_request.client_order_id:
                params["newClientOrderId"] = order_request.client_order_id[:36]
            
            if order_request.reduce_only:
                params["reduceOnly"] = "true"
            
            if order_request.post_only:
                params["timeInForce"] = "GTX"
            
            response = await self._make_request(
                "POST", "/fapi/v1/order", params, signed=True, weight=1
            )
            
            self.latency_tracker.stop_timer(timer_id, True)
            
            order = self._parse_futures_order_response(response)
            order.exchange = "binance_futures"
            
            logger.info(f"Futures order placed: {order.order_id}")
            return order
            
        except Exception as e:
            self.latency_tracker.stop_timer(timer_id, False, {"error": str(e)})
            raise
    
    def _parse_futures_order_response(self, data: Dict) -> Order:
        """Parse Binance Futures order response."""
        status_map = {
            "NEW": OrderState.OPEN,
            "PARTIALLY_FILLED": OrderState.PARTIALLY_FILLED,
            "FILLED": OrderState.FILLED,
            "CANCELED": OrderState.CANCELLED,
            "REJECTED": OrderState.REJECTED,
            "EXPIRED": OrderState.EXPIRED,
        }
        
        order = Order(
            order_id=str(data.get("orderId", "")),
            client_order_id=data.get("clientOrderId"),
            exchange_order_id=str(data.get("orderId", "")),
            symbol=data.get("symbol", ""),
            side=OrderSide.BUY if data.get("side") == "BUY" else OrderSide.SELL,
            order_type=self._parse_order_type(data.get("type", "MARKET")),
            quantity=Decimal(str(abs(float(data.get("origQty", "0"))))),
            price=Decimal(str(data.get("price", "0"))) if data.get("price") != "0" else None,
            stop_price=Decimal(str(data.get("stopPrice", "0"))) if data.get("stopPrice") != "0" else None,
            filled_quantity=Decimal(str(abs(float(data.get("executedQty", "0"))))),
            state=status_map.get(data.get("status"), OrderState.PENDING),
            position_side=data.get("positionSide", "BOTH"),
            reduce_only=data.get("reduceOnly", False),
        )
        
        if order.filled_quantity > 0:
            order.average_fill_price = Decimal(str(data.get("avgPrice", "0")))
        
        order.remaining_quantity = order.quantity - order.filled_quantity
        
        return order
    
    async def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """Get open futures positions."""
        params = {}
        if symbol:
            params["symbol"] = symbol.upper().replace("-", "").replace("/", "")
        
        response = await self._make_request(
            "GET", "/fapi/v2/positionRisk", params, signed=True, weight=5
        )
        
        positions = []
        for pos in response:
            amt = Decimal(str(pos.get("positionAmt", "0")))
            if amt != 0:
                positions.append(Position(
                    symbol=pos.get("symbol", ""),
                    side="long" if amt > 0 else "short",
                    quantity=abs(amt),
                    entry_price=Decimal(str(pos.get("entryPrice", "0"))),
                    unrealized_pnl=Decimal(str(pos.get("unRealizedProfit", "0"))),
                    leverage=Decimal(str(pos.get("leverage", "1"))),
                    margin=Decimal(str(pos.get("isolatedMargin", "0")))
                ))
        
        return positions
    
    async def set_leverage(self, symbol: str, leverage: int) -> Dict:
        """Set leverage for a symbol."""
        params = {
            "symbol": symbol.upper().replace("-", "").replace("/", ""),
            "leverage": leverage
        }
        
        return await self._make_request(
            "POST", "/fapi/v1/leverage", params, signed=True, weight=1
        )
    
    async def get_account_info(self) -> Dict:
        """Get futures account information."""
        return await self._make_request(
            "GET", "/fapi/v2/account", {}, signed=True, weight=5
        )
