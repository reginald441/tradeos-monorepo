"""
TradeOS Binance Client
Complete Binance WebSocket and REST API client for spot and futures markets.

Features:
- Spot and Futures market data
- Real-time WebSocket streams
- REST API for historical data
- Orderbook management
- Trade and ticker streams
- Kline/candlestick data
- User data streams (with authentication)
- Rate limiting
- Auto-reconnection
"""

import asyncio
import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlencode

import aiohttp
import websockets
from websockets.exceptions import ConnectionClosed

from ..websocket_manager import WebSocketConfig, WebSocketConnection, get_websocket_manager
from ..config.symbols import get_exchange_symbol, Symbol

logger = logging.getLogger(__name__)


class BinanceMarket(Enum):
    """Binance market types."""
    SPOT = "spot"
    USD_M_FUTURES = "usd_m_futures"
    COIN_M_FUTURES = "coin_m_futures"
    EUROPEAN_OPTIONS = "european_options"


class BinanceStreamType(Enum):
    """Binance WebSocket stream types."""
    AGGTRADE = "aggTrade"
    TRADE = "trade"
    TICKER = "ticker"
    MINI_TICKER = "miniTicker"
    BOOK_TICKER = "bookTicker"
    KLINE = "kline"
    DEPTH = "depth"
    DEPTH_UPDATE = "depthUpdate"
    LIQUIDATION = "forceOrder"
    MARK_PRICE = "markPrice"
    FUNDING_RATE = "markPrice"


@dataclass
class BinanceConfig:
    """Binance client configuration."""
    # API credentials (optional for public endpoints)
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    
    # Market type
    market: BinanceMarket = BinanceMarket.SPOT
    
    # Testnet
    use_testnet: bool = False
    
    # Rate limiting
    rate_limit_enabled: bool = True
    requests_per_minute: int = 1200
    
    # WebSocket settings
    ws_reconnect_attempts: int = 10
    ws_reconnect_delay: float = 1.0
    
    def get_base_url(self) -> str:
        """Get REST API base URL."""
        if self.market == BinanceMarket.SPOT:
            if self.use_testnet:
                return "https://testnet.binance.vision"
            return "https://api.binance.com"
        elif self.market == BinanceMarket.USD_M_FUTURES:
            if self.use_testnet:
                return "https://testnet.binancefuture.com"
            return "https://fapi.binance.com"
        elif self.market == BinanceMarket.COIN_M_FUTURES:
            if self.use_testnet:
                return "https://testnet.binancefuture.com"
            return "https://dapi.binance.com"
        return "https://api.binance.com"
    
    def get_ws_url(self) -> str:
        """Get WebSocket base URL."""
        if self.market == BinanceMarket.SPOT:
            if self.use_testnet:
                return "wss://testnet.binance.vision/ws"
            return "wss://stream.binance.com:9443/ws"
        elif self.market == BinanceMarket.USD_M_FUTURES:
            if self.use_testnet:
                return "wss://stream.binancefuture.com/ws"
            return "wss://fstream.binance.com/ws"
        elif self.market == BinanceMarket.COIN_M_FUTURES:
            if self.use_testnet:
                return "wss://dstream.binance.com/ws"
            return "wss://dstream.binance.com/ws"
        return "wss://stream.binance.com:9443/ws"
    
    def get_stream_url(self) -> str:
        """Get combined stream WebSocket URL."""
        if self.market == BinanceMarket.SPOT:
            if self.use_testnet:
                return "wss://testnet.binance.vision/stream"
            return "wss://stream.binance.com:9443/stream"
        elif self.market == BinanceMarket.USD_M_FUTURES:
            if self.use_testnet:
                return "wss://stream.binancefuture.com/stream"
            return "wss://fstream.binance.com/stream"
        elif self.market == BinanceMarket.COIN_M_FUTURES:
            if self.use_testnet:
                return "wss://dstream.binance.com/stream"
            return "wss://dstream.binance.com/stream"
        return "wss://stream.binance.com:9443/stream"


class RateLimiter:
    """Simple rate limiter for API requests."""
    
    def __init__(self, requests_per_minute: int = 1200):
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute
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


class BinanceRESTClient:
    """Binance REST API client."""
    
    def __init__(self, config: BinanceConfig):
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None
        self._rate_limiter = RateLimiter(config.requests_per_minute) if config.rate_limit_enabled else None
        self._recv_window = 5000
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"Accept": "application/json"}
            )
        return self._session
    
    def _generate_signature(self, query_string: str) -> str:
        """Generate HMAC signature for authenticated requests."""
        if not self.config.api_secret:
            raise ValueError("API secret required for signed requests")
        
        return hmac.new(
            self.config.api_secret.encode(),
            query_string.encode(),
            hashlib.sha256
        ).hexdigest()
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        signed: bool = False
    ) -> Any:
        """Make HTTP request to Binance API."""
        if self._rate_limiter:
            await self._rate_limiter.acquire()
        
        session = await self._get_session()
        
        base_url = self.config.get_base_url()
        url = f"{base_url}{endpoint}"
        
        headers = {}
        if self.config.api_key:
            headers["X-MBX-APIKEY"] = self.config.api_key
        
        if params is None:
            params = {}
        
        # Add timestamp for signed requests
        if signed:
            params["timestamp"] = int(time.time() * 1000)
            params["recvWindow"] = self._recv_window
        
        # Build query string
        query_string = urlencode(params)
        
        # Add signature for signed requests
        if signed:
            params["signature"] = self._generate_signature(query_string)
            query_string = urlencode(params)
        
        if method == "GET" and query_string:
            url = f"{url}?{query_string}"
        
        try:
            if method == "GET":
                async with session.get(url, headers=headers) as response:
                    data = await response.json()
            elif method == "POST":
                async with session.post(url, headers=headers, data=params) as response:
                    data = await response.json()
            elif method == "DELETE":
                async with session.delete(url, headers=headers) as response:
                    data = await response.json()
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            # Check for API errors
            if "code" in data and data["code"] < 0:
                raise BinanceAPIError(data["code"], data.get("msg", "Unknown error"))
            
            return data
            
        except aiohttp.ClientError as e:
            logger.error(f"HTTP error: {e}")
            raise
    
    async def get(self, endpoint: str, params: Optional[Dict] = None, signed: bool = False) -> Any:
        """Make GET request."""
        return await self._request("GET", endpoint, params, signed)
    
    async def post(self, endpoint: str, params: Optional[Dict] = None, signed: bool = False) -> Any:
        """Make POST request."""
        return await self._request("POST", endpoint, params, signed)
    
    # =========================================================================
    # Market Data Endpoints
    # =========================================================================
    
    async def ping(self) -> bool:
        """Test connectivity."""
        try:
            await self.get("/api/v3/ping")
            return True
        except Exception:
            return False
    
    async def get_server_time(self) -> int:
        """Get server time."""
        data = await self.get("/api/v3/time")
        return data["serverTime"]
    
    async def get_exchange_info(self) -> Dict[str, Any]:
        """Get exchange information."""
        return await self.get("/api/v3/exchangeInfo")
    
    async def get_orderbook(
        self,
        symbol: str,
        limit: int = 100
    ) -> Dict[str, Any]:
        """Get orderbook for a symbol."""
        params = {"symbol": symbol.upper(), "limit": limit}
        return await self.get("/api/v3/depth", params)
    
    async def get_recent_trades(
        self,
        symbol: str,
        limit: int = 500
    ) -> List[Dict[str, Any]]:
        """Get recent trades."""
        params = {"symbol": symbol.upper(), "limit": limit}
        return await self.get("/api/v3/trades", params)
    
    async def get_historical_trades(
        self,
        symbol: str,
        limit: int = 500,
        from_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get historical trades."""
        params = {"symbol": symbol.upper(), "limit": limit}
        if from_id:
            params["fromId"] = from_id
        return await self.get("/api/v3/historicalTrades", params)
    
    async def get_agg_trades(
        self,
        symbol: str,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        from_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get compressed/aggregate trades."""
        params = {"symbol": symbol.upper(), "limit": limit}
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        if from_id:
            params["fromId"] = from_id
        return await self.get("/api/v3/aggTrades", params)
    
    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[List]:
        """Get klines/candlesticks."""
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": limit
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        return await self.get("/api/v3/klines", params)
    
    async def get_avg_price(self, symbol: str) -> Dict[str, Any]:
        """Get current average price."""
        params = {"symbol": symbol.upper()}
        return await self.get("/api/v3/avgPrice", params)
    
    async def get_24h_ticker(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get 24hr ticker price change statistics."""
        params = {}
        if symbol:
            params["symbol"] = symbol.upper()
        return await self.get("/api/v3/ticker/24hr", params)
    
    async def get_latest_price(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get latest price."""
        params = {}
        if symbol:
            params["symbol"] = symbol.upper()
        return await self.get("/api/v3/ticker/price", params)
    
    async def get_book_ticker(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get best price/qty on the order book."""
        params = {}
        if symbol:
            params["symbol"] = symbol.upper()
        return await self.get("/api/v3/ticker/bookTicker", params)
    
    # =========================================================================
    # Futures Specific Endpoints
    # =========================================================================
    
    async def get_funding_rate(
        self,
        symbol: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get funding rate history (futures only)."""
        params = {"limit": limit}
        if symbol:
            params["symbol"] = symbol.upper()
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        
        endpoint = "/fapi/v1/fundingRate" if self.config.market == BinanceMarket.USD_M_FUTURES else "/dapi/v1/fundingRate"
        return await self.get(endpoint, params)
    
    async def get_mark_price(
        self,
        symbol: Optional[str] = None
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Get mark price (futures only)."""
        params = {}
        if symbol:
            params["symbol"] = symbol.upper()
        
        endpoint = "/fapi/v1/premiumIndex" if self.config.market == BinanceMarket.USD_M_FUTURES else "/dapi/v1/premiumIndex"
        return await self.get(endpoint, params)
    
    async def get_open_interest(self, symbol: str) -> Dict[str, Any]:
        """Get open interest (futures only)."""
        params = {"symbol": symbol.upper()}
        
        endpoint = "/fapi/v1/openInterest" if self.config.market == BinanceMarket.USD_M_FUTURES else "/dapi/v1/openInterest"
        return await self.get(endpoint, params)
    
    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()


class BinanceAPIError(Exception):
    """Binance API error."""
    
    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message
        super().__init__(f"Binance API Error {code}: {message}")


class BinanceWebSocketClient:
    """Binance WebSocket client for real-time market data."""
    
    def __init__(self, config: BinanceConfig):
        self.config = config
        self._ws_manager = get_websocket_manager()
        self._connections: Dict[str, WebSocketConnection] = {}
        self._handlers: Dict[str, List[Callable]] = {}
        self._running = False
    
    def _build_stream_name(
        self,
        symbol: str,
        stream_type: BinanceStreamType,
        interval: Optional[str] = None
    ) -> str:
        """Build stream name for Binance."""
        symbol_lower = symbol.lower()
        
        if stream_type == BinanceStreamType.KLINE and interval:
            return f"{symbol_lower}@{stream_type.value}_{interval}"
        elif stream_type == BinanceStreamType.DEPTH:
            return f"{symbol_lower}@{stream_type.value}@100ms"
        else:
            return f"{symbol_lower}@{stream_type.value}"
    
    async def start(self):
        """Start the WebSocket manager."""
        await self._ws_manager.start()
        self._running = True
        logger.info("Binance WebSocket client started")
    
    async def stop(self):
        """Stop all WebSocket connections."""
        self._running = False
        for conn in self._connections.values():
            await conn.stop()
        self._connections.clear()
        logger.info("Binance WebSocket client stopped")
    
    async def subscribe_ticker(
        self,
        symbol: str,
        handler: Callable[[Dict], None],
        mini: bool = False
    ) -> str:
        """Subscribe to ticker stream."""
        stream_type = BinanceStreamType.MINI_TICKER if mini else BinanceStreamType.TICKER
        return await self._subscribe_stream(symbol, stream_type, handler)
    
    async def subscribe_trades(
        self,
        symbol: str,
        handler: Callable[[Dict], None],
        aggregate: bool = True
    ) -> str:
        """Subscribe to trade stream."""
        stream_type = BinanceStreamType.AGGTRADE if aggregate else BinanceStreamType.TRADE
        return await self._subscribe_stream(symbol, stream_type, handler)
    
    async def subscribe_orderbook(
        self,
        symbol: str,
        handler: Callable[[Dict], None]
    ) -> str:
        """Subscribe to orderbook depth stream."""
        return await self._subscribe_stream(symbol, BinanceStreamType.DEPTH, handler)
    
    async def subscribe_klines(
        self,
        symbol: str,
        interval: str,
        handler: Callable[[Dict], None]
    ) -> str:
        """Subscribe to kline/candlestick stream."""
        return await self._subscribe_stream(symbol, BinanceStreamType.KLINE, handler, interval)
    
    async def subscribe_mark_price(
        self,
        symbol: str,
        handler: Callable[[Dict], None],
        update_speed_ms: int = 1000
    ) -> str:
        """Subscribe to mark price stream (futures only)."""
        stream_name = f"{symbol.lower()}@markPrice@{update_speed_ms}ms"
        return await self._subscribe_custom(stream_name, handler)
    
    async def subscribe_liquidations(
        self,
        symbol: str,
        handler: Callable[[Dict], None]
    ) -> str:
        """Subscribe to liquidation stream (futures only)."""
        return await self._subscribe_stream(symbol, BinanceStreamType.LIQUIDATION, handler)
    
    async def subscribe_book_ticker(
        self,
        symbol: str,
        handler: Callable[[Dict], None]
    ) -> str:
        """Subscribe to book ticker stream."""
        return await self._subscribe_stream(symbol, BinanceStreamType.BOOK_TICKER, handler)
    
    async def _subscribe_stream(
        self,
        symbol: str,
        stream_type: BinanceStreamType,
        handler: Callable[[Dict], None],
        interval: Optional[str] = None
    ) -> str:
        """Subscribe to a single stream."""
        stream_name = self._build_stream_name(symbol, stream_type, interval)
        return await self._subscribe_custom(stream_name, handler)
    
    async def _subscribe_custom(
        self,
        stream_name: str,
        handler: Callable[[Dict], None]
    ) -> str:
        """Subscribe to a custom stream."""
        connection_id = f"binance_{stream_name}"
        
        if connection_id in self._connections:
            # Add handler to existing connection
            self._ws_manager.subscribe(stream_name, handler)
            return connection_id
        
        # Create new WebSocket connection
        ws_url = f"{self.config.get_ws_url()}/{stream_name}"
        
        ws_config = WebSocketConfig(
            url=ws_url,
            name=connection_id,
            reconnect_attempts=self.config.ws_reconnect_attempts,
            reconnect_delay_base=self.config.ws_reconnect_delay
        )
        
        connection = await self._ws_manager.add_connection(ws_config)
        self._connections[connection_id] = connection
        
        # Register handler
        self._ws_manager.subscribe("*", handler)
        
        logger.info(f"Subscribed to Binance stream: {stream_name}")
        return connection_id
    
    async def subscribe_multiple(
        self,
        streams: List[str],
        handler: Callable[[Dict], None]
    ) -> str:
        """Subscribe to multiple streams via combined stream."""
        connection_id = f"binance_combined_{hash(tuple(streams))}"
        
        # Build combined stream URL
        streams_param = "/".join(streams)
        ws_url = f"{self.config.get_stream_url()}?streams={streams_param}"
        
        ws_config = WebSocketConfig(
            url=ws_url,
            name=connection_id,
            reconnect_attempts=self.config.ws_reconnect_attempts,
            reconnect_delay_base=self.config.ws_reconnect_delay
        )
        
        connection = await self._ws_manager.add_connection(ws_config)
        self._connections[connection_id] = connection
        
        # Register handler
        self._ws_manager.subscribe("*", handler)
        
        logger.info(f"Subscribed to {len(streams)} Binance streams")
        return connection_id
    
    async def unsubscribe(self, connection_id: str) -> bool:
        """Unsubscribe from a stream."""
        if connection_id in self._connections:
            connection = self._connections.pop(connection_id)
            await connection.stop()
            logger.info(f"Unsubscribed from: {connection_id}")
            return True
        return False


class BinanceClient:
    """Unified Binance client combining REST and WebSocket."""
    
    def __init__(self, config: Optional[BinanceConfig] = None):
        self.config = config or BinanceConfig()
        self.rest = BinanceRESTClient(self.config)
        self.ws = BinanceWebSocketClient(self.config)
    
    async def start(self):
        """Start the client."""
        await self.ws.start()
    
    async def stop(self):
        """Stop the client."""
        await self.ws.stop()
        await self.rest.close()
    
    async def get_symbol_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get current ticker for a symbol."""
        return await self.rest.get_24h_ticker(symbol)
    
    async def get_orderbook_snapshot(
        self,
        symbol: str,
        limit: int = 100
    ) -> Dict[str, Any]:
        """Get orderbook snapshot."""
        return await self.rest.get_orderbook(symbol, limit)
    
    async def get_recent_klines(
        self,
        symbol: str,
        interval: str = "1m",
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get recent klines with normalized format."""
        klines = await self.rest.get_klines(symbol, interval, limit)
        
        # Normalize kline format
        return [
            {
                "timestamp": k[0],
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "close_time": k[6],
                "quote_volume": float(k[7]),
                "trades": k[8],
                "taker_buy_base": float(k[9]),
                "taker_buy_quote": float(k[10]),
            }
            for k in klines
        ]


# Factory functions
def create_spot_client(
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    testnet: bool = False
) -> BinanceClient:
    """Create a spot market client."""
    config = BinanceConfig(
        api_key=api_key,
        api_secret=api_secret,
        market=BinanceMarket.SPOT,
        use_testnet=testnet
    )
    return BinanceClient(config)


def create_futures_client(
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    coin_margined: bool = False,
    testnet: bool = False
) -> BinanceClient:
    """Create a futures market client."""
    market = BinanceMarket.COIN_M_FUTURES if coin_margined else BinanceMarket.USD_M_FUTURES
    config = BinanceConfig(
        api_key=api_key,
        api_secret=api_secret,
        market=market,
        use_testnet=testnet
    )
    return BinanceClient(config)


# Singleton instances
_spot_client: Optional[BinanceClient] = None
_futures_client: Optional[BinanceClient] = None


def get_spot_client() -> BinanceClient:
    """Get singleton spot client."""
    global _spot_client
    if _spot_client is None:
        _spot_client = create_spot_client()
    return _spot_client


def get_futures_client() -> BinanceClient:
    """Get singleton futures client."""
    global _futures_client
    if _futures_client is None:
        _futures_client = create_futures_client()
    return _futures_client


if __name__ == "__main__":
    async def test_binance_client():
        """Test Binance client."""
        client = create_spot_client()
        
        try:
            await client.start()
            
            # Test REST API
            print("Testing REST API...")
            
            # Get server time
            server_time = await client.rest.get_server_time()
            print(f"Server time: {server_time}")
            
            # Get ticker
            ticker = await client.get_symbol_ticker("BTCUSDT")
            print(f"BTCUSDT 24h ticker: {ticker.get('lastPrice')}")
            
            # Get orderbook
            orderbook = await client.get_orderbook_snapshot("BTCUSDT", limit=5)
            print(f"Orderbook bids: {len(orderbook.get('bids', []))}")
            
            # Get klines
            klines = await client.get_recent_klines("BTCUSDT", "1m", 5)
            print(f"Recent klines: {len(klines)}")
            
            # Test WebSocket
            print("\nTesting WebSocket...")
            
            def trade_handler(data):
                print(f"Trade: {data}")
            
            await client.ws.subscribe_trades("btcusdt", trade_handler)
            
            # Run for 10 seconds
            await asyncio.sleep(10)
            
        finally:
            await client.stop()
    
    asyncio.run(test_binance_client())
