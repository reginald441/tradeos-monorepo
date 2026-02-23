"""
TradeOS Coinbase Client
Coinbase Pro / Advanced Trade API client with WebSocket support.

Features:
- Spot market data
- Real-time WebSocket feeds
- REST API for historical data
- Orderbook Level 2 data
- Trade and ticker feeds
- Candle/ohlc data
- Rate limiting
- Authentication support
"""

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union
from urllib.parse import urlencode

import aiohttp
import websockets
from websockets.exceptions import ConnectionClosed

from ..websocket_manager import WebSocketConfig, WebSocketConnection, get_websocket_manager

logger = logging.getLogger(__name__)


class CoinbaseProduct(Enum):
    """Coinbase product types."""
    SPOT = "spot"
    FUTURE = "future"


@dataclass
class CoinbaseConfig:
    """Coinbase client configuration."""
    # API credentials (optional for public endpoints)
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    passphrase: Optional[str] = None  # For Coinbase Pro
    
    # Use Advanced Trade API (v3) or Coinbase Pro API (v2)
    use_advanced_trade: bool = True
    
    # Testnet (sandbox)
    use_sandbox: bool = False
    
    # Rate limiting
    rate_limit_enabled: bool = True
    requests_per_second: float = 10.0
    
    # WebSocket settings
    ws_reconnect_attempts: int = 10
    ws_reconnect_delay: float = 1.0
    
    def get_rest_base_url(self) -> str:
        """Get REST API base URL."""
        if self.use_advanced_trade:
            if self.use_sandbox:
                return "https://api-public.sandbox.exchange.coinbase.com"
            return "https://api.exchange.coinbase.com"
        else:
            if self.use_sandbox:
                return "https://api-public.sandbox.pro.coinbase.com"
            return "https://api.pro.coinbase.com"
    
    def get_ws_url(self) -> str:
        """Get WebSocket URL."""
        if self.use_advanced_trade:
            if self.use_sandbox:
                return "wss://ws-direct.sandbox.exchange.coinbase.com"
            return "wss://ws-direct.exchange.coinbase.com"
        else:
            if self.use_sandbox:
                return "wss://ws-feed-public.sandbox.pro.coinbase.com"
            return "wss://ws-feed.pro.coinbase.com"


class RateLimiter:
    """Rate limiter for API requests."""
    
    def __init__(self, requests_per_second: float = 10.0):
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = 0.0
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire permission to make a request."""
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_request_time
            
            if elapsed < self.min_interval:
                wait_time = self.min_interval - elapsed
                await asyncio.sleep(wait_time)
            
            self.last_request_time = time.time()


class CoinbaseRESTClient:
    """Coinbase REST API client."""
    
    def __init__(self, config: CoinbaseConfig):
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None
        self._rate_limiter = RateLimiter(config.requests_per_second) if config.rate_limit_enabled else None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            headers = {"Accept": "application/json"}
            if self.config.api_key:
                headers["CB-ACCESS-KEY"] = self.config.api_key
            self._session = aiohttp.ClientSession(headers=headers)
        return self._session
    
    def _generate_signature(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        """Generate signature for authenticated requests."""
        if not self.config.api_secret:
            raise ValueError("API secret required for authenticated requests")
        
        message = timestamp + method.upper() + path + body
        signature = hmac.new(
            base64.b64decode(self.config.api_secret),
            message.encode(),
            hashlib.sha256
        ).digest()
        
        return base64.b64encode(signature).decode()
    
    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict] = None,
        body: Optional[Dict] = None,
        authenticated: bool = False
    ) -> Any:
        """Make HTTP request to Coinbase API."""
        if self._rate_limiter:
            await self._rate_limiter.acquire()
        
        session = await self._get_session()
        
        base_url = self.config.get_rest_base_url()
        url = f"{base_url}{path}"
        
        headers = {}
        
        # Add authentication headers if needed
        if authenticated and self.config.api_key:
            timestamp = str(time.time())
            body_str = json.dumps(body) if body else ""
            signature = self._generate_signature(timestamp, method, path, body_str)
            
            headers.update({
                "CB-ACCESS-KEY": self.config.api_key,
                "CB-ACCESS-SIGN": signature,
                "CB-ACCESS-TIMESTAMP": timestamp,
            })
            
            if self.config.passphrase:
                headers["CB-ACCESS-PASSPHRASE"] = self.config.passphrase
        
        # Build query string
        if params:
            url = f"{url}?{urlencode(params)}"
        
        try:
            request_body = json.dumps(body) if body else None
            
            async with session.request(
                method,
                url,
                headers=headers,
                data=request_body
            ) as response:
                data = await response.json()
                
                # Check for errors
                if "message" in data:
                    raise CoinbaseAPIError(data.get("message", "Unknown error"))
                
                return data
                
        except aiohttp.ClientError as e:
            logger.error(f"HTTP error: {e}")
            raise
    
    async def get(self, path: str, params: Optional[Dict] = None, authenticated: bool = False) -> Any:
        """Make GET request."""
        return await self._request("GET", path, params, None, authenticated)
    
    async def post(self, path: str, body: Optional[Dict] = None, authenticated: bool = False) -> Any:
        """Make POST request."""
        return await self._request("POST", path, None, body, authenticated)
    
    # =========================================================================
    # Public Endpoints
    # =========================================================================
    
    async def get_products(self) -> List[Dict[str, Any]]:
        """Get all available products."""
        return await self.get("/products")
    
    async def get_product(self, product_id: str) -> Dict[str, Any]:
        """Get product details."""
        return await self.get(f"/products/{product_id}")
    
    async def get_product_orderbook(
        self,
        product_id: str,
        level: int = 1
    ) -> Dict[str, Any]:
        """Get product orderbook.
        
        Levels:
        - 1: Only the best bid and ask
        - 2: Top 50 bids and asks (aggregated)
        - 3: Full orderbook (non-aggregated)
        """
        params = {"level": level}
        return await self.get(f"/products/{product_id}/book", params)
    
    async def get_product_ticker(self, product_id: str) -> Dict[str, Any]:
        """Get product ticker."""
        return await self.get(f"/products/{product_id}/ticker")
    
    async def get_product_trades(
        self,
        product_id: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get latest trades for a product."""
        params = {"limit": limit}
        return await self.get(f"/products/{product_id}/trades", params)
    
    async def get_product_candles(
        self,
        product_id: str,
        granularity: int = 3600,
        start: Optional[str] = None,
        end: Optional[str] = None
    ) -> List[List]:
        """Get historic rates (candles) for a product.
        
        Granularity in seconds: 60, 300, 900, 3600, 21600, 86400
        """
        params = {"granularity": granularity}
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        
        return await self.get(f"/products/{product_id}/candles", params)
    
    async def get_product_stats(self, product_id: str) -> Dict[str, Any]:
        """Get 24hr stats for a product."""
        return await self.get(f"/products/{product_id}/stats")
    
    async def get_currencies(self) -> List[Dict[str, Any]]:
        """Get all currencies."""
        return await self.get("/currencies")
    
    async def get_time(self) -> Dict[str, Any]:
        """Get API server time."""
        return await self.get("/time")
    
    # =========================================================================
    # Advanced Trade API Endpoints (v3)
    # =========================================================================
    
    async def list_accounts(self) -> List[Dict[str, Any]]:
        """List accounts (requires authentication)."""
        return await self.get("/api/v3/brokerage/accounts", authenticated=True)
    
    async def get_account(self, account_id: str) -> Dict[str, Any]:
        """Get account details (requires authentication)."""
        return await self.get(f"/api/v3/brokerage/accounts/{account_id}", authenticated=True)
    
    async def get_market_trades(
        self,
        product_id: str,
        limit: int = 100
    ) -> Dict[str, Any]:
        """Get market trades (Advanced Trade API)."""
        params = {"product_id": product_id, "limit": limit}
        return await self.get("/api/v3/brokerage/products/{product_id}/ticker", params)
    
    async def get_candles_v3(
        self,
        product_id: str,
        start: str,
        end: str,
        granularity: str = "ONE_HOUR"
    ) -> Dict[str, Any]:
        """Get candles (Advanced Trade API v3).
        
        Granularity: ONE_MINUTE, FIVE_MINUTE, FIFTEEN_MINUTE, THIRTY_MINUTE,
                     ONE_HOUR, TWO_HOUR, SIX_HOUR, ONE_DAY
        """
        params = {
            "start": start,
            "end": end,
            "granularity": granularity
        }
        return await self.get(f"/api/v3/brokerage/products/{product_id}/candles", params)
    
    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()


class CoinbaseAPIError(Exception):
    """Coinbase API error."""
    
    def __init__(self, message: str):
        self.message = message
        super().__init__(f"Coinbase API Error: {message}")


class CoinbaseWebSocketClient:
    """Coinbase WebSocket client for real-time market data."""
    
    # Channel types
    CHANNEL_HEARTBEAT = "heartbeat"
    CHANNEL_TICKER = "ticker"
    CHANNEL_LEVEL2 = "level2"
    CHANNEL_USER = "user"
    CHANNEL_MATCHES = "matches"
    CHANNEL_FULL = "full"
    CHANNEL_STATUS = "status"
    
    def __init__(self, config: CoinbaseConfig):
        self.config = config
        self._ws_manager = get_websocket_manager()
        self._connections: Dict[str, WebSocketConnection] = {}
        self._handlers: Dict[str, List[Callable]] = {
            "subscriptions": [],
            "heartbeat": [],
            "ticker": [],
            "snapshot": [],
            "l2update": [],
            "match": [],
            "last_match": [],
            "received": [],
            "open": [],
            "done": [],
            "change": [],
            "activate": [],
            "error": [],
        }
        self._running = False
        self._subscribed_products: Set[str] = set()
        self._subscribed_channels: Set[str] = set()
    
    async def start(self):
        """Start the WebSocket client."""
        await self._ws_manager.start()
        self._running = True
        logger.info("Coinbase WebSocket client started")
    
    async def stop(self):
        """Stop all WebSocket connections."""
        self._running = False
        for conn in self._connections.values():
            await conn.stop()
        self._connections.clear()
        logger.info("Coinbase WebSocket client stopped")
    
    def _build_subscribe_message(
        self,
        product_ids: List[str],
        channels: List[str]
    ) -> Dict[str, Any]:
        """Build subscribe message."""
        message = {
            "type": "subscribe",
            "product_ids": product_ids,
            "channels": channels
        }
        
        # Add authentication if credentials available
        if self.config.api_key and self.config.api_secret:
            timestamp = str(time.time())
            message_str = json.dumps(message)
            signature = self._generate_ws_signature(timestamp, message_str)
            
            message.update({
                "signature": signature,
                "key": self.config.api_key,
                "passphrase": self.config.passphrase or "",
                "timestamp": timestamp
            })
        
        return message
    
    def _generate_ws_signature(self, timestamp: str, message: str) -> str:
        """Generate signature for WebSocket authentication."""
        message_to_sign = timestamp + message
        signature = hmac.new(
            base64.b64decode(self.config.api_secret),
            message_to_sign.encode(),
            hashlib.sha256
        ).digest()
        return base64.b64encode(signature).decode()
    
    async def subscribe(
        self,
        product_ids: List[str],
        channels: List[str],
        handler: Callable[[Dict], None]
    ) -> str:
        """Subscribe to channels for products."""
        connection_id = f"coinbase_{','.join(product_ids)}_{','.join(channels)}"
        
        # Track subscriptions
        self._subscribed_products.update(product_ids)
        self._subscribed_channels.update(channels)
        
        # Create WebSocket connection
        ws_config = WebSocketConfig(
            url=self.config.get_ws_url(),
            name=connection_id,
            reconnect_attempts=self.config.ws_reconnect_attempts,
            reconnect_delay_base=self.config.ws_reconnect_delay
        )
        
        connection = await self._ws_manager.add_connection(ws_config, auto_start=False)
        self._connections[connection_id] = connection
        
        # Start connection
        asyncio.create_task(connection.start())
        
        # Wait for connection and send subscribe message
        await asyncio.sleep(1)  # Give time to connect
        
        subscribe_msg = self._build_subscribe_message(product_ids, channels)
        await connection.send(json.dumps(subscribe_msg))
        
        # Register handler
        self._ws_manager.subscribe("*", handler)
        
        logger.info(f"Subscribed to Coinbase channels: {channels} for {product_ids}")
        return connection_id
    
    async def subscribe_ticker(
        self,
        product_ids: List[str],
        handler: Callable[[Dict], None]
    ) -> str:
        """Subscribe to ticker channel."""
        return await self.subscribe(product_ids, [self.CHANNEL_TICKER], handler)
    
    async def subscribe_level2(
        self,
        product_ids: List[str],
        handler: Callable[[Dict], None]
    ) -> str:
        """Subscribe to Level 2 orderbook channel."""
        return await self.subscribe(product_ids, [self.CHANNEL_LEVEL2], handler)
    
    async def subscribe_matches(
        self,
        product_ids: List[str],
        handler: Callable[[Dict], None]
    ) -> str:
        """Subscribe to match/trade channel."""
        return await self.subscribe(product_ids, [self.CHANNEL_MATCHES], handler)
    
    async def subscribe_full(
        self,
        product_ids: List[str],
        handler: Callable[[Dict], None]
    ) -> str:
        """Subscribe to full channel (all updates)."""
        return await self.subscribe(product_ids, [self.CHANNEL_FULL], handler)
    
    async def unsubscribe(self, connection_id: str) -> bool:
        """Unsubscribe from a connection."""
        if connection_id in self._connections:
            connection = self._connections.pop(connection_id)
            await connection.stop()
            logger.info(f"Unsubscribed from: {connection_id}")
            return True
        return False
    
    def on_message(self, msg_type: str, handler: Callable[[Dict], None]):
        """Register a handler for a specific message type."""
        if msg_type not in self._handlers:
            self._handlers[msg_type] = []
        self._handlers[msg_type].append(handler)


class CoinbaseClient:
    """Unified Coinbase client combining REST and WebSocket."""
    
    def __init__(self, config: Optional[CoinbaseConfig] = None):
        self.config = config or CoinbaseConfig()
        self.rest = CoinbaseRESTClient(self.config)
        self.ws = CoinbaseWebSocketClient(self.config)
    
    async def start(self):
        """Start the client."""
        await self.ws.start()
    
    async def stop(self):
        """Stop the client."""
        await self.ws.stop()
        await self.rest.close()
    
    async def get_product_ticker(self, product_id: str) -> Dict[str, Any]:
        """Get current ticker for a product."""
        return await self.rest.get_product_ticker(product_id)
    
    async def get_orderbook_snapshot(
        self,
        product_id: str,
        level: int = 2
    ) -> Dict[str, Any]:
        """Get orderbook snapshot."""
        return await self.rest.get_product_orderbook(product_id, level)
    
    async def get_recent_candles(
        self,
        product_id: str,
        granularity: int = 3600,
        limit: int = 300
    ) -> List[Dict[str, Any]]:
        """Get recent candles with normalized format."""
        candles = await self.rest.get_product_candles(product_id, granularity)
        
        # Normalize candle format
        # Coinbase format: [time, low, high, open, close, volume]
        return [
            {
                "timestamp": int(c[0]) * 1000,  # Convert to milliseconds
                "low": float(c[1]),
                "high": float(c[2]),
                "open": float(c[3]),
                "close": float(c[4]),
                "volume": float(c[5]),
            }
            for c in candles[-limit:]  # Get most recent
        ]


# Factory functions
def create_client(
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    passphrase: Optional[str] = None,
    use_sandbox: bool = False,
    use_advanced_trade: bool = True
) -> CoinbaseClient:
    """Create a Coinbase client."""
    config = CoinbaseConfig(
        api_key=api_key,
        api_secret=api_secret,
        passphrase=passphrase,
        use_sandbox=use_sandbox,
        use_advanced_trade=use_advanced_trade
    )
    return CoinbaseClient(config)


# Singleton instance
_client_instance: Optional[CoinbaseClient] = None


def get_client() -> CoinbaseClient:
    """Get singleton client instance."""
    global _client_instance
    if _client_instance is None:
        _client_instance = create_client()
    return _client_instance


if __name__ == "__main__":
    async def test_coinbase_client():
        """Test Coinbase client."""
        client = create_client()
        
        try:
            await client.start()
            
            # Test REST API
            print("Testing REST API...")
            
            # Get products
            products = await client.rest.get_products()
            print(f"Available products: {len(products)}")
            
            # Get BTC-USD ticker
            ticker = await client.get_product_ticker("BTC-USD")
            print(f"BTC-USD Ticker: {ticker}")
            
            # Get orderbook
            orderbook = await client.get_orderbook_snapshot("BTC-USD", level=2)
            print(f"Orderbook bids: {len(orderbook.get('bids', []))}")
            
            # Get candles
            candles = await client.get_recent_candles("BTC-USD", granularity=3600, limit=5)
            print(f"Recent candles: {candles}")
            
            # Test WebSocket
            print("\nTesting WebSocket...")
            
            def ticker_handler(data):
                if data.get("type") == "ticker":
                    print(f"Ticker: {data.get('product_id')} - {data.get('price')}")
            
            await client.ws.subscribe_ticker(["BTC-USD"], ticker_handler)
            
            # Run for 10 seconds
            await asyncio.sleep(10)
            
        finally:
            await client.stop()
    
    asyncio.run(test_coinbase_client())
