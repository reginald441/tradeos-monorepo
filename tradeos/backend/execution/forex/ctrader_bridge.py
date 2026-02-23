"""
TradeOS cTrader Integration Framework
=====================================
cTrader Open API integration for forex and CFD trading.
Supports gRPC-based communication with cTrader servers.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
from enum import Enum

from ..base_exchange import (
    BaseExchange, Balance, Ticker, OrderBook, OrderBookLevel,
    Position, ExchangeError, AuthenticationError, NetworkError,
    InvalidOrderError, OrderNotFoundError
)
from ..models.order import (
    Order, OrderRequest, OrderFill, OrderState, OrderSide,
    OrderType, TimeInForce, OrderEvent, OrderEventType
)
from ..latency_tracker import LatencyType

logger = logging.getLogger(__name__)


class cTraderEnvironment(Enum):
    """cTrader environment types."""
    DEMO = "demo"
    LIVE = "live"


@dataclass
class cTraderConfig:
    """cTrader API configuration."""
    client_id: str
    client_secret: str
    access_token: str
    account_id: int
    environment: cTraderEnvironment = cTraderEnvironment.DEMO
    timeout_seconds: float = 30.0


class cTraderBridge(BaseExchange):
    """
    cTrader Open API Bridge.
    
    Features:
    - OAuth 2.0 authentication
    - REST API for trading operations
    - gRPC streaming for real-time data
    - Full forex and CFD support
    
    Note: This is a framework implementation. Full gRPC/protobuf
    integration requires generated client code from cTrader API specs.
    """
    
    DEMO_BASE_URL = "https://demo.ctraderapi.com"
    LIVE_BASE_URL = "https://live.ctraderapi.com"
    AUTH_URL = "https://openapi.ctrader.com/apps/token"
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("ctrader", config)
        
        self.ctrader_config = cTraderConfig(
            client_id=config.get("client_id", ""),
            client_secret=config.get("client_secret", ""),
            access_token=config.get("access_token", ""),
            account_id=config.get("account_id", 0),
            environment=cTraderEnvironment(config.get("environment", "demo")),
            timeout_seconds=config.get("timeout_seconds", 30.0)
        )
        
        self.base_url = (
            self.LIVE_BASE_URL 
            if self.ctrader_config.environment == cTraderEnvironment.LIVE 
            else self.DEMO_BASE_URL
        )
        
        self._session: Optional[Any] = None  # aiohttp session
        self._grpc_channel: Optional[Any] = None  # gRPC channel
        self._auth_token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None
        
        logger.info(f"cTrader bridge initialized ({self.ctrader_config.environment.value})")
    
    # ==================== Connection ====================
    
    async def connect(self) -> bool:
        """Establish connection to cTrader."""
        try:
            import aiohttp
            
            if self._session is None or self._session.closed:
                self._session = aiohttp.ClientSession()
            
            # Authenticate
            await self._authenticate()
            
            # Test connection
            account_info = await self._get_account_info()
            if account_info:
                logger.info(f"Connected to cTrader. Account: {self.ctrader_config.account_id}")
                self.is_connected = True
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to connect to cTrader: {e}")
            self.is_connected = False
            return False
    
    async def disconnect(self) -> bool:
        """Close connection to cTrader."""
        try:
            if self._session and not self._session.closed:
                await self._session.close()
            
            self.is_connected = False
            logger.info("Disconnected from cTrader")
            return True
            
        except Exception as e:
            logger.error(f"Error disconnecting from cTrader: {e}")
            return False
    
    async def is_connection_healthy(self) -> bool:
        """Check if connection is healthy."""
        try:
            await self._get_account_info()
            return True
        except Exception:
            return False
    
    async def _authenticate(self):
        """Authenticate with cTrader OAuth 2.0."""
        import aiohttp
        
        # Check if we have a valid token
        if self._auth_token and self._token_expiry and datetime.utcnow() < self._token_expiry:
            return
        
        # Request new token
        auth_data = {
            "grant_type": "client_credentials",
            "client_id": self.ctrader_config.client_id,
            "client_secret": self.ctrader_config.client_secret,
        }
        
        async with self._session.post(self.AUTH_URL, data=auth_data) as response:
            if response.status != 200:
                raise AuthenticationError("Failed to authenticate with cTrader")
            
            token_data = await response.json()
            self._auth_token = token_data.get("access_token")
            expires_in = token_data.get("expires_in", 3600)
            self._token_expiry = datetime.utcnow() + timedelta(seconds=expires_in - 300)
            
            logger.info("cTrader authentication successful")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get authenticated headers."""
        return {
            "Authorization": f"Bearer {self._auth_token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
    
    async def _make_request(self, method: str, endpoint: str,
                            params: Optional[Dict] = None,
                            body: Optional[Dict] = None) -> Dict:
        """Make authenticated request to cTrader API."""
        import aiohttp
        
        if self._session is None or self._session.closed:
            raise NetworkError("Not connected to cTrader")
        
        # Ensure authenticated
        await self._authenticate()
        
        await self._apply_rate_limit()
        
        url = f"{self.base_url}{endpoint}"
        headers = self._get_headers()
        
        timer_id = self.latency_tracker.start_timer(
            f"ctrader_{endpoint}", LatencyType.API_REQUEST, "ctrader"
        )
        
        try:
            async with self._session.request(
                method, url, headers=headers, params=params, json=body
            ) as response:
                data = await response.json()
                
                success = response.status < 400
                self.latency_tracker.stop_timer(timer_id, success)
                
                if response.status == 401:
                    raise AuthenticationError("cTrader authentication failed")
                elif response.status == 429:
                    raise RateLimitError("Rate limit exceeded", retry_after=1.0)
                elif response.status >= 400:
                    error_msg = data.get("message", "Unknown error")
                    raise ExchangeError(error_msg)
                
                return data
                
        except aiohttp.ClientError as e:
            self.latency_tracker.stop_timer(timer_id, False)
            raise NetworkError(f"Request failed: {e}")
    
    async def _get_account_info(self) -> Dict:
        """Get account information."""
        return await self._make_request(
            "GET", "/v2/account", {"accountId": self.ctrader_config.account_id}
        )
    
    # ==================== Order Operations ====================
    
    async def place_order(self, order_request: OrderRequest) -> Order:
        """Place a new order via cTrader."""
        timer_id = self.latency_tracker.start_timer(
            f"order_{order_request.symbol}", LatencyType.ORDER_SUBMIT, "ctrader"
        )
        
        try:
            body = {
                "accountId": self.ctrader_config.account_id,
                "symbolName": order_request.symbol,
                "orderType": self._map_order_type(order_request.order_type),
                "tradeSide": "BUY" if order_request.side == OrderSide.BUY else "SELL",
                "volume": int(order_request.quantity * 100),  # cTrader uses cents
            }
            
            if order_request.price:
                body["price"] = float(order_request.price)
            
            if order_request.stop_price:
                body["stopPrice"] = float(order_request.stop_price)
            
            response = await self._make_request("POST", "/v2/trading/orders", body=body)
            
            self.latency_tracker.stop_timer(timer_id, True)
            
            order_data = response.get("order", {})
            order = self._parse_order_response(order_data)
            order.exchange = "ctrader"
            
            logger.info(f"Order placed on cTrader: {order.order_id}")
            return order
            
        except Exception as e:
            self.latency_tracker.stop_timer(timer_id, False, {"error": str(e)})
            raise
    
    async def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> bool:
        """Cancel an existing order."""
        timer_id = self.latency_tracker.start_timer(
            f"cancel_{order_id}", LatencyType.CANCEL_REQUEST, "ctrader"
        )
        
        try:
            await self._make_request(
                "DELETE", "/v2/trading/orders", 
                {"accountId": self.ctrader_config.account_id, "orderId": order_id}
            )
            
            self.latency_tracker.stop_timer(timer_id, True)
            logger.info(f"Order cancelled on cTrader: {order_id}")
            return True
            
        except Exception as e:
            self.latency_tracker.stop_timer(timer_id, False, {"error": str(e)})
            raise
    
    async def get_order(self, order_id: str, symbol: Optional[str] = None) -> Optional[Order]:
        """Get order details from cTrader."""
        try:
            response = await self._make_request(
                "GET", "/v2/trading/orders",
                {"accountId": self.ctrader_config.account_id, "orderId": order_id}
            )
            
            order_data = response.get("order", {})
            if order_data:
                order = self._parse_order_response(order_data)
                order.exchange = "ctrader"
                return order
            
            return None
            
        except ExchangeError:
            return None
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all open orders from cTrader."""
        params = {"accountId": self.ctrader_config.account_id}
        if symbol:
            params["symbolName"] = symbol
        
        response = await self._make_request("GET", "/v2/trading/orders", params)
        
        orders = []
        for order_data in response.get("orders", []):
            order = self._parse_order_response(order_data)
            order.exchange = "ctrader"
            orders.append(order)
        
        return orders
    
    async def get_order_history(self, symbol: Optional[str] = None,
                                 limit: int = 100) -> List[Order]:
        """Get order history from cTrader."""
        params = {
            "accountId": self.ctrader_config.account_id,
            "limit": min(limit, 1000)
        }
        if symbol:
            params["symbolName"] = symbol
        
        response = await self._make_request("GET", "/v2/trading/orders/history", params)
        
        orders = []
        for order_data in response.get("orders", []):
            order = self._parse_order_response(order_data)
            order.exchange = "ctrader"
            orders.append(order)
        
        return orders
    
    def _parse_order_response(self, data: Dict) -> Order:
        """Parse cTrader order response to Order object."""
        status_map = {
            "PENDING": OrderState.PENDING,
            "ACCEPTED": OrderState.OPEN,
            "PARTIALLY_FILLED": OrderState.PARTIALLY_FILLED,
            "FILLED": OrderState.FILLED,
            "CANCELLED": OrderState.CANCELLED,
            "REJECTED": OrderState.REJECTED,
            "EXPIRED": OrderState.EXPIRED,
        }
        
        order = Order(
            order_id=str(data.get("orderId", "")),
            client_order_id=data.get("clientOrderId"),
            exchange_order_id=str(data.get("orderId", "")),
            symbol=data.get("symbolName", ""),
            side=OrderSide.BUY if data.get("tradeSide") == "BUY" else OrderSide.SELL,
            order_type=self._parse_order_type(data.get("orderType", "MARKET")),
            quantity=Decimal(str(data.get("volume", "0"))) / 100,  # Convert from cents
            price=Decimal(str(data.get("price", "0"))) if data.get("price") else None,
            stop_price=Decimal(str(data.get("stopPrice", "0"))) if data.get("stopPrice") else None,
            filled_quantity=Decimal(str(data.get("filledVolume", "0"))) / 100,
            state=status_map.get(data.get("status"), OrderState.PENDING),
            exchange="ctrader"
        )
        
        order.remaining_quantity = order.quantity - order.filled_quantity
        
        if data.get("averagePrice"):
            order.average_fill_price = Decimal(str(data.get("averagePrice")))
        
        return order
    
    def _map_order_type(self, order_type: OrderType) -> str:
        """Map internal order type to cTrader type."""
        mapping = {
            OrderType.MARKET: "MARKET",
            OrderType.LIMIT: "LIMIT",
            OrderType.STOP: "STOP",
            OrderType.STOP_LIMIT: "STOP_LIMIT",
        }
        return mapping.get(order_type, "MARKET")
    
    def _parse_order_type(self, type_str: str) -> OrderType:
        """Parse cTrader order type."""
        mapping = {
            "MARKET": OrderType.MARKET,
            "LIMIT": OrderType.LIMIT,
            "STOP": OrderType.STOP,
            "STOP_LIMIT": OrderType.STOP_LIMIT,
        }
        return mapping.get(type_str, OrderType.MARKET)
    
    # ==================== Account Operations ====================
    
    async def get_balance(self, asset: Optional[str] = None) -> Dict[str, Balance]:
        """Get account balance from cTrader."""
        response = await self._get_account_info()
        
        balances = {}
        account_data = response.get("account", {})
        
        currency = account_data.get("currency", "USD")
        balance = Decimal(str(account_data.get("balance", "0")))
        margin_used = Decimal(str(account_data.get("marginUsed", "0")))
        
        balances[currency] = Balance(
            asset=currency,
            free=balance - margin_used,
            locked=margin_used,
            total=balance
        )
        
        return balances
    
    async def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """Get open positions from cTrader."""
        params = {"accountId": self.ctrader_config.account_id}
        if symbol:
            params["symbolName"] = symbol
        
        response = await self._make_request("GET", "/v2/trading/positions", params)
        
        positions = []
        for pos_data in response.get("positions", []):
            volume = Decimal(str(pos_data.get("volume", "0"))) / 100
            if volume > 0:
                positions.append(Position(
                    symbol=pos_data.get("symbolName", ""),
                    side="long" if pos_data.get("tradeSide") == "BUY" else "short",
                    quantity=volume,
                    entry_price=Decimal(str(pos_data.get("entryPrice", "0"))),
                    unrealized_pnl=Decimal(str(pos_data.get("unrealizedProfit", "0"))),
                    leverage=Decimal(str(pos_data.get("leverage", "1"))),
                ))
        
        return positions
    
    # ==================== Market Data ====================
    
    async def get_ticker(self, symbol: str) -> Ticker:
        """Get current price ticker from cTrader."""
        params = {"symbolName": symbol}
        response = await self._make_request("GET", "/v2/public/quote", params)
        
        quote_data = response.get("quote", {})
        
        return Ticker(
            symbol=symbol,
            bid=Decimal(str(quote_data.get("bid", "0"))),
            ask=Decimal(str(quote_data.get("ask", "0"))),
            last=Decimal(str(quote_data.get("last", "0"))),
            volume_24h=Decimal(str(quote_data.get("volume", "0"))),
            timestamp=datetime.utcnow()
        )
    
    async def get_orderbook(self, symbol: str, depth: int = 20) -> OrderBook:
        """Get order book snapshot from cTrader."""
        params = {"symbolName": symbol, "depth": min(depth, 50)}
        response = await self._make_request("GET", "/v2/public/depth", params)
        
        depth_data = response.get("depth", {})
        
        bids = [
            OrderBookLevel(
                price=Decimal(str(b.get("price", "0"))),
                quantity=Decimal(str(b.get("volume", "0"))) / 100
            )
            for b in depth_data.get("bids", [])
        ]
        
        asks = [
            OrderBookLevel(
                price=Decimal(str(a.get("price", "0"))),
                quantity=Decimal(str(a.get("volume", "0"))) / 100
            )
            for a in depth_data.get("asks", [])
        ]
        
        return OrderBook(
            symbol=symbol,
            bids=bids,
            asks=asks,
            timestamp=datetime.utcnow()
        )
    
    async def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Get recent trades from cTrader."""
        params = {"symbolName": symbol, "limit": min(limit, 1000)}
        response = await self._make_request("GET", "/v2/public/ticks", params)
        
        return response.get("ticks", [])
    
    # ==================== WebSocket (gRPC) ====================
    
    async def subscribe_ticker(self, symbols: List[str]):
        """Subscribe to ticker updates via gRPC."""
        # gRPC streaming implementation would go here
        logger.info(f"Subscribing to tickers: {symbols}")
    
    async def subscribe_orderbook(self, symbols: List[str], depth: int = 20):
        """Subscribe to order book updates via gRPC."""
        logger.info(f"Subscribing to order books: {symbols}")
    
    async def subscribe_trades(self, symbols: List[str]):
        """Subscribe to trade updates via gRPC."""
        logger.info(f"Subscribing to trades: {symbols}")
    
    async def subscribe_user_data(self):
        """Subscribe to user data via gRPC."""
        logger.info("Subscribing to user data")


# Import for timedelta
from datetime import timedelta
