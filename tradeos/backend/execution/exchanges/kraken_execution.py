"""
TradeOS Kraken Exchange Implementation
======================================
Complete Kraken spot and futures trading integration.
Supports REST API with robust rate limiting.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import time
import urllib.parse
from decimal import Decimal
from typing import Any, Dict, List, Optional
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
from ..latency_tracker import LatencyType

logger = logging.getLogger(__name__)


class KrakenExchange(BaseExchange):
    """
    Kraken Spot Exchange Implementation.
    
    Features:
    - Full spot trading support
    - Margin trading support
    - Robust rate limiting
    - Signature-based authentication
    """
    
    BASE_URL = "https://api.kraken.com"
    API_VERSION = "/0"
    
    # Rate limiting: Tier 2 (most common)
    # Decay rate: 0.33 per second
    MAX_COUNTER = 60
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("kraken", config)
        self.api_key = config.get("api_key", "")
        self.api_secret = config.get("api_secret", "")
        
        self._session: Optional[aiohttp.ClientSession] = None
        self._rate_limit_counter = 0.0
        self._last_request_time = 0.0
        self._lock = asyncio.Lock()
        
        # Asset pair mapping (Kraken uses different symbols)
        self._asset_pairs: Dict[str, str] = {}
        
        logger.info("Kraken exchange initialized")
    
    # ==================== Connection ====================
    
    async def connect(self) -> bool:
        """Establish connection to Kraken."""
        try:
            if self._session is None or self._session.closed:
                self._session = aiohttp.ClientSession()
            
            # Load asset pairs
            await self._load_asset_pairs()
            
            # Test connection with server time
            server_time = await self._get_server_time()
            logger.info(f"Connected to Kraken. Server time: {server_time}")
            
            self.is_connected = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Kraken: {e}")
            self.is_connected = False
            return False
    
    async def disconnect(self) -> bool:
        """Close connection to Kraken."""
        try:
            if self._session and not self._session.closed:
                await self._session.close()
            
            self.is_connected = False
            logger.info("Disconnected from Kraken")
            return True
            
        except Exception as e:
            logger.error(f"Error disconnecting from Kraken: {e}")
            return False
    
    async def is_connection_healthy(self) -> bool:
        """Check if connection is healthy."""
        try:
            await self._get_server_time()
            return True
        except Exception:
            return False
    
    async def _get_server_time(self) -> Dict:
        """Get Kraken server time."""
        return await self._make_public_request("/public/Time")
    
    async def _load_asset_pairs(self):
        """Load available asset pairs."""
        response = await self._make_public_request("/public/AssetPairs")
        
        for pair_name, pair_info in response.get("result", {}).items():
            wsname = pair_info.get("wsname", "")
            if wsname:
                # Map standard format to Kraken format
                base = pair_info.get("base", "")
                quote = pair_info.get("quote", "")
                std_symbol = f"{base}{quote}"
                self._asset_pairs[std_symbol] = pair_name
    
    # ==================== Authentication ====================
    
    def _generate_signature(self, urlpath: str, data: Dict) -> str:
        """Generate API signature."""
        postdata = urllib.parse.urlencode(data)
        encoded = (str(data["nonce"]) + postdata).encode()
        message = urlpath.encode() + hashlib.sha256(encoded).digest()
        
        signature = hmac.new(
            base64.b64decode(self.api_secret),
            message,
            hashlib.sha512
        )
        return base64.b64encode(signature.digest()).decode()
    
    # ==================== Request Handling ====================
    
    async def _make_public_request(self, endpoint: str, 
                                    params: Optional[Dict] = None) -> Dict:
        """Make public request to Kraken API."""
        if self._session is None or self._session.closed:
            raise NetworkError("Not connected to Kraken")
        
        await self._apply_rate_limit()
        
        url = f"{self.BASE_URL}{self.API_VERSION}{endpoint}"
        
        timer_id = self.latency_tracker.start_timer(
            f"kraken_public_{endpoint}", LatencyType.API_REQUEST, "kraken"
        )
        
        try:
            async with self._session.get(url, params=params) as response:
                data = await response.json()
                
                success = len(data.get("error", [])) == 0
                self.latency_tracker.stop_timer(timer_id, success)
                
                if data.get("error"):
                    raise ExchangeError(f"Kraken error: {data['error']}")
                
                return data
                
        except aiohttp.ClientError as e:
            self.latency_tracker.stop_timer(timer_id, False)
            raise NetworkError(f"Request failed: {e}")
    
    async def _make_private_request(self, endpoint: str,
                                     params: Optional[Dict] = None) -> Dict:
        """Make authenticated request to Kraken API."""
        if self._session is None or self._session.closed:
            raise NetworkError("Not connected to Kraken")
        
        await self._apply_rate_limit()
        
        url = f"{self.BASE_URL}{self.API_VERSION}{endpoint}"
        
        data = params or {}
        data["nonce"] = int(time.time() * 1000)
        
        signature = self._generate_signature(endpoint, data)
        
        headers = {
            "API-Key": self.api_key,
            "API-Sign": signature,
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        timer_id = self.latency_tracker.start_timer(
            f"kraken_private_{endpoint}", LatencyType.API_REQUEST, "kraken"
        )
        
        try:
            async with self._session.post(url, data=data, headers=headers) as response:
                data = await response.json()
                
                errors = data.get("error", [])
                success = len(errors) == 0
                self.latency_tracker.stop_timer(timer_id, success)
                
                if errors:
                    error_str = ", ".join(errors)
                    if "Invalid key" in error_str or "Invalid signature" in error_str:
                        raise AuthenticationError(error_str)
                    elif "Rate limit exceeded" in error_str:
                        raise RateLimitError(error_str, retry_after=3.0)
                    elif "Insufficient funds" in error_str:
                        raise InsufficientFundsError(error_str)
                    elif "Invalid arguments" in error_str:
                        raise InvalidOrderError(error_str)
                    elif "Unknown order" in error_str:
                        raise OrderNotFoundError(error_str)
                    raise ExchangeError(error_str)
                
                return data
                
        except aiohttp.ClientError as e:
            self.latency_tracker.stop_timer(timer_id, False)
            raise NetworkError(f"Request failed: {e}")
    
    async def _apply_rate_limit(self):
        """Apply rate limiting."""
        async with self._lock:
            current_time = time.time()
            time_passed = current_time - self._last_request_time
            
            # Decay counter
            self._rate_limit_counter = max(
                0, 
                self._rate_limit_counter - time_passed * 0.33
            )
            
            # Check if we need to wait
            if self._rate_limit_counter >= self.MAX_COUNTER:
                sleep_time = (self._rate_limit_counter - self.MAX_COUNTER + 1) / 0.33
                logger.warning(f"Kraken rate limit approaching, sleeping {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time)
                self._rate_limit_counter = max(0, self._rate_limit_counter - sleep_time * 0.33)
            
            self._rate_limit_counter += 2  # Each request costs ~2
            self._last_request_time = current_time
    
    def _format_symbol(self, symbol: str) -> str:
        """Format symbol for Kraken."""
        # Check if already in Kraken format
        if symbol in self._asset_pairs.values():
            return symbol
        
        # Check mapping
        if symbol in self._asset_pairs:
            return self._asset_pairs[symbol]
        
        # Try to construct (e.g., BTCUSD -> XBTUSD)
        symbol = symbol.replace("-", "").replace("/", "")
        if symbol.startswith("BTC"):
            symbol = "XBT" + symbol[3:]
        
        # Try to find matching pair
        for std, kraken in self._asset_pairs.items():
            if std == symbol:
                return kraken
        
        return symbol
    
    # ==================== Order Operations ====================
    
    async def place_order(self, order_request: OrderRequest) -> Order:
        """Place a new order on Kraken."""
        timer_id = self.latency_tracker.start_timer(
            f"order_{order_request.symbol}", LatencyType.ORDER_SUBMIT, "kraken"
        )
        
        try:
            params = {
                "pair": self._format_symbol(order_request.symbol),
                "type": "buy" if order_request.side == OrderSide.BUY else "sell",
                "ordertype": self._map_order_type(order_request.order_type),
                "volume": str(order_request.quantity),
            }
            
            if order_request.price:
                params["price"] = str(order_request.price)
            
            if order_request.stop_price:
                params["price2"] = str(order_request.stop_price)
            
            # Add client order ID (userref)
            if order_request.client_order_id:
                params["userref"] = abs(hash(order_request.client_order_id)) % 2147483647
            
            response = await self._make_private_request("/private/AddOrder", params)
            
            self.latency_tracker.stop_timer(timer_id, True)
            
            result = response.get("result", {})
            txid = result.get("txid", [None])[0]
            
            if txid:
                # Fetch the created order
                order = await self.get_order(txid)
                if order:
                    order.exchange = "kraken"
                    logger.info(f"Order placed on Kraken: {order.order_id}")
                    return order
            
            raise ExchangeError("Failed to get order ID from response")
            
        except Exception as e:
            self.latency_tracker.stop_timer(timer_id, False, {"error": str(e)})
            raise
    
    async def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> bool:
        """Cancel an existing order."""
        timer_id = self.latency_tracker.start_timer(
            f"cancel_{order_id}", LatencyType.CANCEL_REQUEST, "kraken"
        )
        
        try:
            params = {"txid": order_id}
            
            response = await self._make_private_request("/private/CancelOrder", params)
            
            self.latency_tracker.stop_timer(timer_id, True)
            
            result = response.get("result", {})
            count = result.get("count", 0)
            
            if count > 0:
                logger.info(f"Order cancelled on Kraken: {order_id}")
                return True
            
            return False
            
        except Exception as e:
            self.latency_tracker.stop_timer(timer_id, False, {"error": str(e)})
            raise
    
    async def get_order(self, order_id: str, symbol: Optional[str] = None) -> Optional[Order]:
        """Get order details from Kraken."""
        params = {"txid": order_id}
        
        try:
            response = await self._make_private_request("/private/QueryOrders", params)
            
            result = response.get("result", {})
            order_data = result.get(order_id)
            
            if order_data:
                order = self._parse_order_response(order_id, order_data)
                order.exchange = "kraken"
                return order
            
            return None
            
        except OrderNotFoundError:
            return None
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all open orders."""
        params = {}
        if symbol:
            params["pair"] = self._format_symbol(symbol)
        
        response = await self._make_private_request("/private/OpenOrders", params)
        
        orders = []
        open_orders = response.get("result", {}).get("open", {})
        
        for order_id, order_data in open_orders.items():
            order = self._parse_order_response(order_id, order_data)
            order.exchange = "kraken"
            orders.append(order)
        
        return orders
    
    async def get_order_history(self, symbol: Optional[str] = None,
                                 limit: int = 100) -> List[Order]:
        """Get order history."""
        params = {}
        if symbol:
            params["pair"] = self._format_symbol(symbol)
        
        response = await self._make_private_request("/private/ClosedOrders", params)
        
        orders = []
        closed_orders = response.get("result", {}).get("closed", {})
        
        for order_id, order_data in list(closed_orders.items())[:limit]:
            order = self._parse_order_response(order_id, order_data)
            order.exchange = "kraken"
            orders.append(order)
        
        return orders
    
    def _parse_order_response(self, order_id: str, data: Dict) -> Order:
        """Parse Kraken order response to Order object."""
        status_map = {
            "pending": OrderState.PENDING,
            "open": OrderState.OPEN,
            "closed": OrderState.FILLED,
            "canceled": OrderState.CANCELLED,
            "expired": OrderState.EXPIRED,
        }
        
        descr = data.get("descr", {})
        
        # Parse order type
        ordertype = descr.get("ordertype", "market")
        order_type = self._parse_order_type(ordertype)
        
        # Parse volumes
        vol = Decimal(str(data.get("vol", "0")))
        vol_exec = Decimal(str(data.get("vol_exec", "0")))
        
        # Parse price
        price = None
        if descr.get("price"):
            price = Decimal(str(descr.get("price", "0")))
        
        # Calculate average price
        avg_price = None
        if vol_exec > 0 and data.get("cost"):
            avg_price = Decimal(str(data.get("cost", "0"))) / vol_exec
        
        order = Order(
            order_id=order_id,
            client_order_id=str(data.get("userref", "")) if data.get("userref") else None,
            exchange_order_id=order_id,
            symbol=descr.get("pair", ""),
            side=OrderSide.BUY if descr.get("type") == "buy" else OrderSide.SELL,
            order_type=order_type,
            quantity=vol,
            price=price,
            filled_quantity=vol_exec,
            remaining_quantity=vol - vol_exec,
            average_fill_price=avg_price,
            state=status_map.get(data.get("status"), OrderState.PENDING),
            total_fee=Decimal(str(data.get("fee", "0"))),
        )
        
        return order
    
    def _map_order_type(self, order_type: OrderType) -> str:
        """Map internal order type to Kraken type."""
        mapping = {
            OrderType.MARKET: "market",
            OrderType.LIMIT: "limit",
            OrderType.STOP: "stop-loss",
            OrderType.STOP_LIMIT: "stop-loss-limit",
            OrderType.TAKE_PROFIT: "take-profit",
            OrderType.TAKE_PROFIT_LIMIT: "take-profit-limit",
        }
        return mapping.get(order_type, "market")
    
    def _parse_order_type(self, type_str: str) -> OrderType:
        """Parse Kraken order type."""
        mapping = {
            "market": OrderType.MARKET,
            "limit": OrderType.LIMIT,
            "stop-loss": OrderType.STOP,
            "stop-loss-limit": OrderType.STOP_LIMIT,
            "take-profit": OrderType.TAKE_PROFIT,
            "take-profit-limit": OrderType.TAKE_PROFIT_LIMIT,
        }
        return mapping.get(type_str, OrderType.MARKET)
    
    # ==================== Account Operations ====================
    
    async def get_balance(self, asset: Optional[str] = None) -> Dict[str, Balance]:
        """Get account balance."""
        response = await self._make_private_request("/private/Balance")
        
        balances = {}
        result = response.get("result", {})
        
        for currency, amount in result.items():
            if asset and currency != asset:
                continue
            
            total = Decimal(str(amount))
            
            balances[currency] = Balance(
                asset=currency,
                free=total,  # Kraken doesn't separate free/locked in basic balance
                locked=Decimal("0"),
                total=total
            )
        
        return balances
    
    async def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """Get open margin positions."""
        params = {}
        if symbol:
            params["pair"] = self._format_symbol(symbol)
        
        try:
            response = await self._make_private_request("/private/OpenPositions", params)
            
            positions = []
            result = response.get("result", {})
            
            for pos_id, pos_data in result.items():
                positions.append(Position(
                    symbol=pos_data.get("pair", ""),
                    side="long" if pos_data.get("type") == "buy" else "short",
                    quantity=Decimal(str(pos_data.get("vol", "0"))),
                    entry_price=Decimal(str(pos_data.get("cost", "0"))) / Decimal(str(pos_data.get("vol", "1"))),
                    unrealized_pnl=Decimal(str(pos_data.get("net", "0"))),
                    leverage=Decimal(str(pos_data.get("leverage", "1"))),
                ))
            
            return positions
            
        except ExchangeError:
            # May not have margin enabled
            return []
    
    # ==================== Market Data ====================
    
    async def get_ticker(self, symbol: str) -> Ticker:
        """Get current price ticker."""
        params = {"pair": self._format_symbol(symbol)}
        
        response = await self._make_public_request("/public/Ticker", params)
        
        result = response.get("result", {})
        ticker_data = list(result.values())[0] if result else {}
        
        # Kraken format: [price, whole lot volume, lot volume]
        bid_data = ticker_data.get("b", ["0", "0", "0"])
        ask_data = ticker_data.get("a", ["0", "0", "0"])
        last_data = ticker_data.get("c", ["0", "0"])
        volume_data = ticker_data.get("v", ["0", "0"])
        
        return Ticker(
            symbol=symbol,
            bid=Decimal(str(bid_data[0])),
            ask=Decimal(str(ask_data[0])),
            last=Decimal(str(last_data[0])),
            volume_24h=Decimal(str(volume_data[1])),  # 24h volume
            timestamp=datetime.utcnow()
        )
    
    async def get_orderbook(self, symbol: str, depth: int = 20) -> OrderBook:
        """Get order book snapshot."""
        params = {
            "pair": self._format_symbol(symbol),
            "count": min(depth, 500)
        }
        
        response = await self._make_public_request("/public/Depth", params)
        
        result = response.get("result", {})
        orderbook_data = list(result.values())[0] if result else {}
        
        bids = [
            OrderBookLevel(price=Decimal(str(b[0])), quantity=Decimal(str(b[1])))
            for b in orderbook_data.get("bids", [])
        ]
        
        asks = [
            OrderBookLevel(price=Decimal(str(a[0])), quantity=Decimal(str(a[1])))
            for a in orderbook_data.get("asks", [])
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
            "pair": self._format_symbol(symbol),
        }
        
        response = await self._make_public_request("/public/Trades", params)
        
        result = response.get("result", {})
        trades_data = list(result.values())[0] if result else []
        
        trades = []
        for trade in trades_data[:limit]:
            trades.append({
                "price": trade[0],
                "volume": trade[1],
                "time": trade[2],
                "side": "buy" if trade[3] == "b" else "sell",
                "order_type": "limit" if trade[4] == "l" else "market",
            })
        
        return trades
    
    # ==================== WebSocket (Not Implemented) ====================
    
    async def subscribe_ticker(self, symbols: List[str]):
        """Subscribe to ticker updates via WebSocket."""
        logger.warning("Kraken WebSocket not yet implemented")
    
    async def subscribe_orderbook(self, symbols: List[str], depth: int = 20):
        """Subscribe to order book updates via WebSocket."""
        logger.warning("Kraken WebSocket not yet implemented")
    
    async def subscribe_trades(self, symbols: List[str]):
        """Subscribe to trade updates via WebSocket."""
        logger.warning("Kraken WebSocket not yet implemented")
    
    async def subscribe_user_data(self):
        """Subscribe to user data via WebSocket."""
        logger.warning("Kraken WebSocket not yet implemented")
