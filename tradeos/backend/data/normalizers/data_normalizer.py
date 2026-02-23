"""
TradeOS Data Normalizer
Normalize market data from different exchanges to a common format.

Features:
- Trade normalization
- Orderbook normalization
- Ticker normalization
- Candle/OHLC normalization
- Funding rate normalization
- Exchange-specific field mapping
- Data validation
"""

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime
import time

from ..config.symbols import get_symbol, get_exchange_symbol

logger = logging.getLogger(__name__)


class ExchangeType(Enum):
    """Supported exchange types."""
    BINANCE = "binance"
    BINANCE_FUTURES = "binance_futures"
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    BYBIT = "bybit"
    OKX = "okx"
    FOREX = "forex"
    CUSTOM = "custom"


class DataType(Enum):
    """Types of market data."""
    TRADE = "trade"
    ORDERBOOK = "orderbook"
    TICKER = "ticker"
    OHLC = "ohlc"
    FUNDING_RATE = "funding_rate"
    MARK_PRICE = "mark_price"
    LIQUIDATION = "liquidation"


@dataclass
class NormalizedTrade:
    """Normalized trade data."""
    symbol: str
    exchange: str
    timestamp: int  # Milliseconds
    price: float
    quantity: float
    side: str  # 'buy' or 'sell'
    trade_id: Optional[str] = None
    quote_quantity: float = 0.0
    is_buyer_maker: bool = False
    raw_data: Dict[str, Any] = field(default_factory=dict, repr=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": "trade",
            "symbol": self.symbol,
            "exchange": self.exchange,
            "timestamp": self.timestamp,
            "price": self.price,
            "quantity": self.quantity,
            "side": self.side,
            "trade_id": self.trade_id,
            "quote_quantity": self.quote_quantity,
            "is_buyer_maker": self.is_buyer_maker,
        }


@dataclass
class NormalizedOrderbook:
    """Normalized orderbook data."""
    symbol: str
    exchange: str
    timestamp: int
    last_update_id: int
    bids: List[List[float]]  # [[price, quantity], ...]
    asks: List[List[float]]
    raw_data: Dict[str, Any] = field(default_factory=dict, repr=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": "orderbook",
            "symbol": self.symbol,
            "exchange": self.exchange,
            "timestamp": self.timestamp,
            "last_update_id": self.last_update_id,
            "bids": self.bids,
            "asks": self.asks,
        }
    
    @property
    def best_bid(self) -> Optional[float]:
        """Get best bid price."""
        return self.bids[0][0] if self.bids else None
    
    @property
    def best_ask(self) -> Optional[float]:
        """Get best ask price."""
        return self.asks[0][0] if self.asks else None
    
    @property
    def spread(self) -> Optional[float]:
        """Get bid-ask spread."""
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None
    
    @property
    def mid_price(self) -> Optional[float]:
        """Get mid price."""
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None


@dataclass
class NormalizedTicker:
    """Normalized ticker data."""
    symbol: str
    exchange: str
    timestamp: int
    price: float
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0
    quote_volume: float = 0.0
    change: float = 0.0
    change_percent: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    bid_volume: float = 0.0
    ask_volume: float = 0.0
    raw_data: Dict[str, Any] = field(default_factory=dict, repr=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": "ticker",
            "symbol": self.symbol,
            "exchange": self.exchange,
            "timestamp": self.timestamp,
            "price": self.price,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "quote_volume": self.quote_volume,
            "change": self.change,
            "change_percent": self.change_percent,
            "bid": self.bid,
            "ask": self.ask,
            "bid_volume": self.bid_volume,
            "ask_volume": self.ask_volume,
        }


@dataclass
class NormalizedOHLC:
    """Normalized OHLC candle data."""
    symbol: str
    exchange: str
    timestamp: int
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float = 0.0
    trades: int = 0
    taker_buy_volume: float = 0.0
    taker_buy_quote_volume: float = 0.0
    vwap: float = 0.0
    closed: bool = False
    raw_data: Dict[str, Any] = field(default_factory=dict, repr=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": "ohlc",
            "symbol": self.symbol,
            "exchange": self.exchange,
            "timestamp": self.timestamp,
            "timeframe": self.timeframe,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "quote_volume": self.quote_volume,
            "trades": self.trades,
            "taker_buy_volume": self.taker_buy_volume,
            "taker_buy_quote_volume": self.taker_buy_quote_volume,
            "vwap": self.vwap,
            "closed": self.closed,
        }


@dataclass
class NormalizedFundingRate:
    """Normalized funding rate data."""
    symbol: str
    exchange: str
    timestamp: int
    funding_rate: float
    mark_price: float = 0.0
    index_price: float = 0.0
    estimated_rate: float = 0.0
    next_funding_time: int = 0
    raw_data: Dict[str, Any] = field(default_factory=dict, repr=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": "funding_rate",
            "symbol": self.symbol,
            "exchange": self.exchange,
            "timestamp": self.timestamp,
            "funding_rate": self.funding_rate,
            "mark_price": self.mark_price,
            "index_price": self.index_price,
            "estimated_rate": self.estimated_rate,
            "next_funding_time": self.next_funding_time,
        }


@dataclass
class NormalizedLiquidation:
    """Normalized liquidation data."""
    symbol: str
    exchange: str
    timestamp: int
    price: float
    quantity: float
    side: str
    liquidation_id: Optional[str] = None
    raw_data: Dict[str, Any] = field(default_factory=dict, repr=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": "liquidation",
            "symbol": self.symbol,
            "exchange": self.exchange,
            "timestamp": self.timestamp,
            "price": self.price,
            "quantity": self.quantity,
            "side": self.side,
            "liquidation_id": self.liquidation_id,
        }


class DataNormalizer:
    """Normalize data from various exchanges to common format."""
    
    def __init__(self):
        self._parsers: Dict[ExchangeType, Dict[DataType, Callable]] = {
            ExchangeType.BINANCE: {
                DataType.TRADE: self._parse_binance_trade,
                DataType.ORDERBOOK: self._parse_binance_orderbook,
                DataType.TICKER: self._parse_binance_ticker,
                DataType.OHLC: self._parse_binance_ohlc,
                DataType.FUNDING_RATE: self._parse_binance_funding_rate,
                DataType.LIQUIDATION: self._parse_binance_liquidation,
            },
            ExchangeType.BINANCE_FUTURES: {
                DataType.TRADE: self._parse_binance_trade,
                DataType.ORDERBOOK: self._parse_binance_orderbook,
                DataType.TICKER: self._parse_binance_ticker,
                DataType.OHLC: self._parse_binance_ohlc,
                DataType.FUNDING_RATE: self._parse_binance_funding_rate,
                DataType.LIQUIDATION: self._parse_binance_liquidation,
            },
            ExchangeType.COINBASE: {
                DataType.TRADE: self._parse_coinbase_trade,
                DataType.ORDERBOOK: self._parse_coinbase_orderbook,
                DataType.TICKER: self._parse_coinbase_ticker,
                DataType.OHLC: self._parse_coinbase_ohlc,
            },
        }
    
    def normalize(
        self,
        data: Dict[str, Any],
        exchange: ExchangeType,
        data_type: DataType
    ) -> Optional[Union[NormalizedTrade, NormalizedOrderbook, NormalizedTicker, NormalizedOHLC]]:
        """Normalize data from an exchange."""
        if exchange not in self._parsers:
            logger.warning(f"No parser for exchange: {exchange}")
            return None
        
        if data_type not in self._parsers[exchange]:
            logger.warning(f"No parser for {data_type} on {exchange}")
            return None
        
        try:
            parser = self._parsers[exchange][data_type]
            return parser(data)
        except Exception as e:
            logger.error(f"Error normalizing {data_type} from {exchange}: {e}")
            return None
    
    # =========================================================================
    # Binance Parsers
    # =========================================================================
    
    def _parse_binance_trade(self, data: Dict[str, Any]) -> NormalizedTrade:
        """Parse Binance trade data."""
        # WebSocket format
        if "e" in data and data["e"] == "aggTrade":
            return NormalizedTrade(
                symbol=data.get("s", ""),
                exchange="binance",
                timestamp=data.get("T", 0),
                price=float(data.get("p", 0)),
                quantity=float(data.get("q", 0)),
                side="sell" if data.get("m", False) else "buy",
                trade_id=str(data.get("a", "")),
                quote_quantity=float(data.get("p", 0)) * float(data.get("q", 0)),
                is_buyer_maker=data.get("m", False),
                raw_data=data
            )
        
        # REST API format
        return NormalizedTrade(
            symbol=data.get("symbol", ""),
            exchange="binance",
            timestamp=data.get("time", int(time.time() * 1000)),
            price=float(data.get("price", 0)),
            quantity=float(data.get("qty", data.get("quantity", 0))),
            side="buy" if data.get("isBuyerMaker", False) else "sell",
            trade_id=str(data.get("id", "")),
            quote_quantity=float(data.get("quoteQty", 0)),
            is_buyer_maker=data.get("isBuyerMaker", False),
            raw_data=data
        )
    
    def _parse_binance_orderbook(self, data: Dict[str, Any]) -> NormalizedOrderbook:
        """Parse Binance orderbook data."""
        # WebSocket format (depth update)
        if "e" in data and data["e"] == "depthUpdate":
            return NormalizedOrderbook(
                symbol=data.get("s", ""),
                exchange="binance",
                timestamp=data.get("E", 0),
                last_update_id=data.get("u", 0),
                bids=[[float(p), float(q)] for p, q in data.get("b", [])],
                asks=[[float(p), float(q)] for p, q in data.get("a", [])],
                raw_data=data
            )
        
        # REST API format (snapshot)
        return NormalizedOrderbook(
            symbol=data.get("symbol", ""),
            exchange="binance",
            timestamp=int(time.time() * 1000),
            last_update_id=data.get("lastUpdateId", 0),
            bids=[[float(p), float(q)] for p, q in data.get("bids", [])],
            asks=[[float(p), float(q)] for p, q in data.get("asks", [])],
            raw_data=data
        )
    
    def _parse_binance_ticker(self, data: Dict[str, Any]) -> NormalizedTicker:
        """Parse Binance ticker data."""
        # WebSocket format (24hr ticker)
        if "e" in data and data["e"] == "24hrTicker":
            return NormalizedTicker(
                symbol=data.get("s", ""),
                exchange="binance",
                timestamp=data.get("E", 0),
                price=float(data.get("c", 0)),
                open=float(data.get("o", 0)),
                high=float(data.get("h", 0)),
                low=float(data.get("l", 0)),
                close=float(data.get("c", 0)),
                volume=float(data.get("v", 0)),
                quote_volume=float(data.get("q", 0)),
                change=float(data.get("p", 0)),
                change_percent=float(data.get("P", 0)),
                bid=float(data.get("b", 0)),
                ask=float(data.get("a", 0)),
                bid_volume=float(data.get("B", 0)),
                ask_volume=float(data.get("A", 0)),
                raw_data=data
            )
        
        # Mini ticker format
        if "e" in data and data["e"] == "24hrMiniTicker":
            return NormalizedTicker(
                symbol=data.get("s", ""),
                exchange="binance",
                timestamp=data.get("E", 0),
                price=float(data.get("c", 0)),
                open=float(data.get("o", 0)),
                high=float(data.get("h", 0)),
                low=float(data.get("l", 0)),
                close=float(data.get("c", 0)),
                volume=float(data.get("v", 0)),
                quote_volume=float(data.get("q", 0)),
                raw_data=data
            )
        
        # REST API format
        return NormalizedTicker(
            symbol=data.get("symbol", ""),
            exchange="binance",
            timestamp=int(time.time() * 1000),
            price=float(data.get("lastPrice", 0)),
            open=float(data.get("openPrice", 0)),
            high=float(data.get("highPrice", 0)),
            low=float(data.get("lowPrice", 0)),
            close=float(data.get("lastPrice", 0)),
            volume=float(data.get("volume", 0)),
            quote_volume=float(data.get("quoteVolume", 0)),
            change=float(data.get("priceChange", 0)),
            change_percent=float(data.get("priceChangePercent", 0)),
            bid=float(data.get("bidPrice", 0)),
            ask=float(data.get("askPrice", 0)),
            bid_volume=float(data.get("bidQty", 0)),
            ask_volume=float(data.get("askQty", 0)),
            raw_data=data
        )
    
    def _parse_binance_ohlc(self, data: Dict[str, Any]) -> NormalizedOHLC:
        """Parse Binance OHLC/kline data."""
        # WebSocket format
        if "e" in data and data["e"] == "kline":
            k = data.get("k", {})
            return NormalizedOHLC(
                symbol=data.get("s", ""),
                exchange="binance",
                timestamp=k.get("t", 0),
                timeframe=k.get("i", "1m"),
                open=float(k.get("o", 0)),
                high=float(k.get("h", 0)),
                low=float(k.get("l", 0)),
                close=float(k.get("c", 0)),
                volume=float(k.get("v", 0)),
                quote_volume=float(k.get("q", 0)),
                trades=k.get("n", 0),
                taker_buy_volume=float(k.get("V", 0)),
                taker_buy_quote_volume=float(k.get("Q", 0)),
                closed=k.get("x", False),
                raw_data=data
            )
        
        # REST API format (single kline)
        if isinstance(data, list) and len(data) >= 6:
            return NormalizedOHLC(
                symbol="",
                exchange="binance",
                timestamp=data[0],
                timeframe="1m",  # Need to infer from context
                open=float(data[1]),
                high=float(data[2]),
                low=float(data[3]),
                close=float(data[4]),
                volume=float(data[5]),
                quote_volume=float(data[7]) if len(data) > 7 else 0,
                trades=data[8] if len(data) > 8 else 0,
                taker_buy_volume=float(data[9]) if len(data) > 9 else 0,
                taker_buy_quote_volume=float(data[10]) if len(data) > 10 else 0,
                raw_data=data
            )
        
        return NormalizedOHLC(
            symbol="",
            exchange="binance",
            timestamp=0,
            timeframe="1m",
            open=0, high=0, low=0, close=0, volume=0,
            raw_data=data
        )
    
    def _parse_binance_funding_rate(self, data: Dict[str, Any]) -> NormalizedFundingRate:
        """Parse Binance funding rate data."""
        # WebSocket format
        if "e" in data and data["e"] == "markPriceUpdate":
            return NormalizedFundingRate(
                symbol=data.get("s", ""),
                exchange="binance_futures",
                timestamp=data.get("E", 0),
                funding_rate=float(data.get("r", 0)),
                mark_price=float(data.get("p", 0)),
                index_price=float(data.get("i", 0)),
                estimated_rate=float(data.get("r", 0)),
                next_funding_time=data.get("T", 0),
                raw_data=data
            )
        
        # REST API format
        return NormalizedFundingRate(
            symbol=data.get("symbol", ""),
            exchange="binance_futures",
            timestamp=int(time.time() * 1000),
            funding_rate=float(data.get("lastFundingRate", 0)),
            mark_price=float(data.get("markPrice", 0)),
            index_price=float(data.get("indexPrice", 0)),
            estimated_rate=float(data.get("lastFundingRate", 0)),
            next_funding_time=data.get("nextFundingTime", 0),
            raw_data=data
        )
    
    def _parse_binance_liquidation(self, data: Dict[str, Any]) -> NormalizedLiquidation:
        """Parse Binance liquidation data."""
        if "e" in data and data["e"] == "forceOrder":
            o = data.get("o", {})
            return NormalizedLiquidation(
                symbol=o.get("s", ""),
                exchange="binance_futures",
                timestamp=data.get("E", 0),
                price=float(o.get("p", 0)),
                quantity=float(o.get("q", 0)),
                side=o.get("S", "SELL").lower(),
                liquidation_id=str(o.get("i", "")),
                raw_data=data
            )
        
        return NormalizedLiquidation(
            symbol=data.get("symbol", ""),
            exchange="binance_futures",
            timestamp=data.get("time", int(time.time() * 1000)),
            price=float(data.get("price", 0)),
            quantity=float(data.get("qty", 0)),
            side=data.get("side", "SELL").lower(),
            liquidation_id=str(data.get("orderId", "")),
            raw_data=data
        )
    
    # =========================================================================
    # Coinbase Parsers
    # =========================================================================
    
    def _parse_coinbase_trade(self, data: Dict[str, Any]) -> NormalizedTrade:
        """Parse Coinbase trade data."""
        # WebSocket format
        if data.get("type") == "match" or data.get("type") == "last_match":
            return NormalizedTrade(
                symbol=data.get("product_id", "").replace("-", ""),
                exchange="coinbase",
                timestamp=int(
                    datetime.fromisoformat(data.get("time", "").replace("Z", "+00:00")).timestamp() * 1000
                ) if data.get("time") else int(time.time() * 1000),
                price=float(data.get("price", 0)),
                quantity=float(data.get("size", 0)),
                side=data.get("side", "buy").lower(),
                trade_id=str(data.get("trade_id", "")),
                quote_quantity=float(data.get("price", 0)) * float(data.get("size", 0)),
                is_buyer_maker=data.get("side", "buy").lower() == "sell",
                raw_data=data
            )
        
        return NormalizedTrade(
            symbol=data.get("product_id", "").replace("-", ""),
            exchange="coinbase",
            timestamp=int(time.time() * 1000),
            price=float(data.get("price", 0)),
            quantity=float(data.get("size", data.get("volume", 0))),
            side="buy",  # Default
            trade_id=str(data.get("trade_id", "")),
            raw_data=data
        )
    
    def _parse_coinbase_orderbook(self, data: Dict[str, Any]) -> NormalizedOrderbook:
        """Parse Coinbase orderbook data."""
        # WebSocket format (snapshot)
        if data.get("type") == "snapshot":
            return NormalizedOrderbook(
                symbol=data.get("product_id", "").replace("-", ""),
                exchange="coinbase",
                timestamp=int(time.time() * 1000),
                last_update_id=0,
                bids=[[float(p), float(q)] for p, q in data.get("bids", [])],
                asks=[[float(p), float(q)] for p, q in data.get("asks", [])],
                raw_data=data
            )
        
        # WebSocket format (l2update)
        if data.get("type") == "l2update":
            changes = data.get("changes", [])
            bids = []
            asks = []
            
            for change in changes:
                side, price, size = change
                if side == "buy":
                    bids.append([float(price), float(size)])
                else:
                    asks.append([float(price), float(size)])
            
            return NormalizedOrderbook(
                symbol=data.get("product_id", "").replace("-", ""),
                exchange="coinbase",
                timestamp=int(
                    datetime.fromisoformat(data.get("time", "").replace("Z", "+00:00")).timestamp() * 1000
                ) if data.get("time") else int(time.time() * 1000),
                last_update_id=0,
                bids=bids,
                asks=asks,
                raw_data=data
            )
        
        # REST API format
        return NormalizedOrderbook(
            symbol=data.get("product_id", "").replace("-", ""),
            exchange="coinbase",
            timestamp=int(time.time() * 1000),
            last_update_id=0,
            bids=[[float(p), float(q)] for p, q in data.get("bids", [])],
            asks=[[float(p), float(q)] for p, q in data.get("asks", [])],
            raw_data=data
        )
    
    def _parse_coinbase_ticker(self, data: Dict[str, Any]) -> NormalizedTicker:
        """Parse Coinbase ticker data."""
        # WebSocket format
        if data.get("type") == "ticker":
            return NormalizedTicker(
                symbol=data.get("product_id", "").replace("-", ""),
                exchange="coinbase",
                timestamp=int(
                    datetime.fromisoformat(data.get("time", "").replace("Z", "+00:00")).timestamp() * 1000
                ) if data.get("time") else int(time.time() * 1000),
                price=float(data.get("price", 0)),
                open=float(data.get("open_24h", 0)),
                high=float(data.get("high_24h", 0)),
                low=float(data.get("low_24h", 0)),
                close=float(data.get("price", 0)),
                volume=float(data.get("volume_24h", 0)),
                quote_volume=float(data.get("volume_30d", 0)),  # Approximation
                change=float(data.get("change_24h", 0)),
                change_percent=float(data.get("change_percent_24h", 0)),
                bid=float(data.get("best_bid", 0)),
                ask=float(data.get("best_ask", 0)),
                raw_data=data
            )
        
        # REST API format
        return NormalizedTicker(
            symbol=data.get("product_id", "").replace("-", ""),
            exchange="coinbase",
            timestamp=int(time.time() * 1000),
            price=float(data.get("price", 0)),
            open=float(data.get("open", 0)),
            high=float(data.get("high", 0)),
            low=float(data.get("low", 0)),
            close=float(data.get("price", 0)),
            volume=float(data.get("volume", 0)),
            raw_data=data
        )
    
    def _parse_coinbase_ohlc(self, data: Dict[str, Any]) -> NormalizedOHLC:
        """Parse Coinbase OHLC data."""
        # REST API format (candles)
        if isinstance(data, list) and len(data) >= 6:
            # Coinbase format: [time, low, high, open, close, volume]
            return NormalizedOHLC(
                symbol="",
                exchange="coinbase",
                timestamp=int(data[0]) * 1000,
                timeframe="1h",  # Need to infer from context
                open=float(data[3]),
                high=float(data[2]),
                low=float(data[1]),
                close=float(data[4]),
                volume=float(data[5]),
                raw_data=data
            )
        
        return NormalizedOHLC(
            symbol="",
            exchange="coinbase",
            timestamp=0,
            timeframe="1h",
            open=0, high=0, low=0, close=0, volume=0,
            raw_data=data
        )


# Factory functions
def create_normalizer() -> DataNormalizer:
    """Create a data normalizer."""
    return DataNormalizer()


# Singleton instance
_normalizer_instance: Optional[DataNormalizer] = None


def get_normalizer() -> DataNormalizer:
    """Get singleton normalizer instance."""
    global _normalizer_instance
    if _normalizer_instance is None:
        _normalizer_instance = create_normalizer()
    return _normalizer_instance


if __name__ == "__main__":
    def test_normalizer():
        """Test data normalizer."""
        normalizer = create_normalizer()
        
        # Test Binance trade
        binance_trade = {
            "e": "aggTrade",
            "E": 123456789,
            "s": "BTCUSDT",
            "a": 12345,
            "p": "50000.00",
            "q": "0.1",
            "f": 100,
            "l": 105,
            "T": 1234567890000,
            "m": False,
        }
        
        normalized = normalizer.normalize(
            binance_trade,
            ExchangeType.BINANCE,
            DataType.TRADE
        )
        print(f"Binance Trade: {normalized.to_dict()}")
        
        # Test Binance orderbook
        binance_orderbook = {
            "e": "depthUpdate",
            "E": 123456789,
            "s": "BTCUSDT",
            "U": 100,
            "u": 200,
            "b": [["50000", "1.0"], ["49999", "0.5"]],
            "a": [["50001", "0.8"], ["50002", "1.2"]],
        }
        
        normalized = normalizer.normalize(
            binance_orderbook,
            ExchangeType.BINANCE,
            DataType.ORDERBOOK
        )
        print(f"Binance Orderbook: {normalized.to_dict()}")
        print(f"Spread: {normalized.spread}, Mid: {normalized.mid_price}")
    
    test_normalizer()
