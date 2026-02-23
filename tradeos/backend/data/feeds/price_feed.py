"""
TradeOS Unified Price Feed
Aggregated price feed combining multiple exchange sources.

Features:
- Multi-source price aggregation
- Price consensus calculation
- Outlier detection and filtering
- Weighted averaging by volume/exchange
- Real-time updates
- Fallback mechanisms
- Spread calculation across exchanges
"""

import asyncio
import logging
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from collections import defaultdict
import statistics
import time

from ..config.symbols import get_symbol, Symbol
from ..normalizers.data_normalizer import (
    DataNormalizer, ExchangeType, DataType,
    NormalizedTrade, NormalizedTicker, NormalizedOrderbook
)
from ..storage.redis_cache import RedisCache, get_cache

logger = logging.getLogger(__name__)


@dataclass
class PricePoint:
    """Price point from a single source."""
    symbol: str
    exchange: str
    timestamp: int
    price: float
    bid: float = 0.0
    ask: float = 0.0
    volume_24h: float = 0.0
    weight: float = 1.0
    latency_ms: float = 0.0
    is_stale: bool = False
    
    @property
    def mid_price(self) -> float:
        """Get mid price."""
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2
        return self.price
    
    @property
    def spread(self) -> float:
        """Get spread."""
        if self.bid > 0 and self.ask > 0:
            return self.ask - self.bid
        return 0.0
    
    @property
    def spread_bps(self) -> float:
        """Get spread in basis points."""
        mid = self.mid_price
        if mid > 0:
            return (self.spread / mid) * 10000
        return 0.0


@dataclass
class AggregatedPrice:
    """Aggregated price from multiple sources."""
    symbol: str
    timestamp: int
    price: float  # Consensus price
    bid: float
    ask: float
    sources: List[PricePoint]
    source_count: int
    
    # Statistics
    min_price: float
    max_price: float
    std_dev: float
    
    # Volume info
    total_volume_24h: float
    
    # Quality metrics
    spread_across_exchanges: float
    confidence_score: float  # 0-100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "price": self.price,
            "bid": self.bid,
            "ask": self.ask,
            "source_count": self.source_count,
            "min_price": self.min_price,
            "max_price": self.max_price,
            "std_dev": self.std_dev,
            "total_volume_24h": self.total_volume_24h,
            "spread_across_exchanges": self.spread_across_exchanges,
            "confidence_score": self.confidence_score,
            "sources": [
                {
                    "exchange": s.exchange,
                    "price": s.price,
                    "bid": s.bid,
                    "ask": s.ask,
                    "weight": s.weight,
                }
                for s in self.sources
            ]
        }


class AggregationMethod(Enum):
    """Price aggregation methods."""
    MEAN = "mean"
    MEDIAN = "median"
    WEIGHTED_BY_VOLUME = "weighted_by_volume"
    WEIGHTED_BY_INVERSE_SPREAD = "weighted_by_inverse_spread"
    VWAP = "vwap"
    BEST_BID_ASK_MID = "best_bid_ask_mid"


class PriceFeedAggregator:
    """Aggregate prices from multiple sources."""
    
    def __init__(
        self,
        method: AggregationMethod = AggregationMethod.WEIGHTED_BY_VOLUME,
        max_sources: int = 10,
        stale_threshold_ms: int = 5000,
        outlier_threshold_std: float = 3.0
    ):
        self.method = method
        self.max_sources = max_sources
        self.stale_threshold_ms = stale_threshold_ms
        self.outlier_threshold_std = outlier_threshold_std
        
        self._price_points: Dict[str, Dict[str, PricePoint]] = defaultdict(dict)
        self._aggregated_prices: Dict[str, AggregatedPrice] = {}
        self._update_handlers: List[Callable[[AggregatedPrice], None]] = []
        self._lock = asyncio.Lock()
    
    def add_update_handler(self, handler: Callable[[AggregatedPrice], None]):
        """Add a handler for price updates."""
        self._update_handlers.append(handler)
    
    async def update_price(self, price_point: PricePoint):
        """Update price from a source."""
        async with self._lock:
            symbol = price_point.symbol
            exchange = price_point.exchange
            
            # Check if price is stale
            current_time = int(time.time() * 1000)
            if current_time - price_point.timestamp > self.stale_threshold_ms:
                price_point.is_stale = True
            
            self._price_points[symbol][exchange] = price_point
            
            # Aggregate and notify
            aggregated = self._aggregate_symbol(symbol)
            if aggregated:
                self._aggregated_prices[symbol] = aggregated
                
                for handler in self._update_handlers:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            asyncio.create_task(handler(aggregated))
                        else:
                            handler(aggregated)
                    except Exception as e:
                        logger.error(f"Update handler error: {e}")
    
    def _aggregate_symbol(self, symbol: str) -> Optional[AggregatedPrice]:
        """Aggregate prices for a symbol."""
        points = list(self._price_points[symbol].values())
        
        # Filter out stale prices
        active_points = [p for p in points if not p.is_stale]
        
        if not active_points:
            return None
        
        # Remove outliers
        filtered_points = self._filter_outliers(active_points)
        
        if not filtered_points:
            filtered_points = active_points
        
        # Calculate weights
        weighted_points = self._calculate_weights(filtered_points)
        
        # Aggregate price
        timestamp = int(time.time() * 1000)
        
        if self.method == AggregationMethod.MEAN:
            price = statistics.mean([p.mid_price for p in weighted_points])
        elif self.method == AggregationMethod.MEDIAN:
            price = statistics.median([p.mid_price for p in weighted_points])
        elif self.method == AggregationMethod.WEIGHTED_BY_VOLUME:
            price = self._weighted_average(weighted_points, "volume_24h")
        elif self.method == AggregationMethod.WEIGHTED_BY_INVERSE_SPREAD:
            price = self._weighted_average_by_inverse_spread(weighted_points)
        elif self.method == AggregationMethod.BEST_BID_ASK_MID:
            price = self._best_bid_ask_mid(weighted_points)
        else:
            price = statistics.median([p.mid_price for p in weighted_points])
        
        # Aggregate bid/ask
        bid = self._aggregate_bid(weighted_points)
        ask = self._aggregate_ask(weighted_points)
        
        # Calculate statistics
        prices = [p.mid_price for p in weighted_points]
        min_price = min(prices)
        max_price = max(prices)
        std_dev = statistics.stdev(prices) if len(prices) > 1 else 0
        
        # Calculate volume
        total_volume = sum(p.volume_24h for p in weighted_points)
        
        # Calculate spread across exchanges
        spread_across = max_price - min_price
        
        # Calculate confidence score
        confidence = self._calculate_confidence(weighted_points, std_dev, price)
        
        return AggregatedPrice(
            symbol=symbol,
            timestamp=timestamp,
            price=price,
            bid=bid,
            ask=ask,
            sources=weighted_points,
            source_count=len(weighted_points),
            min_price=min_price,
            max_price=max_price,
            std_dev=std_dev,
            total_volume_24h=total_volume,
            spread_across_exchanges=spread_across,
            confidence_score=confidence
        )
    
    def _filter_outliers(self, points: List[PricePoint]) -> List[PricePoint]:
        """Filter out outlier prices."""
        if len(points) < 3:
            return points
        
        prices = [p.mid_price for p in points]
        mean = statistics.mean(prices)
        std = statistics.stdev(prices)
        
        if std == 0:
            return points
        
        return [
            p for p in points
            if abs(p.mid_price - mean) <= std * self.outlier_threshold_std
        ]
    
    def _calculate_weights(self, points: List[PricePoint]) -> List[PricePoint]:
        """Calculate weights for each price point."""
        if not points:
            return []
        
        # Base weight by volume
        total_volume = sum(p.volume_24h for p in points)
        
        for p in points:
            if total_volume > 0:
                p.weight = p.volume_24h / total_volume
            else:
                p.weight = 1.0 / len(points)
            
            # Penalize high spread
            if p.spread_bps > 0:
                p.weight *= max(0.1, 1 - (p.spread_bps / 100))
            
            # Penalize high latency
            if p.latency_ms > 100:
                p.weight *= max(0.5, 1 - (p.latency_ms / 1000))
        
        return points
    
    def _weighted_average(self, points: List[PricePoint], weight_field: str) -> float:
        """Calculate weighted average."""
        total_weight = sum(getattr(p, weight_field, 0) for p in points)
        
        if total_weight == 0:
            return statistics.mean([p.mid_price for p in points])
        
        weighted_sum = sum(
            p.mid_price * getattr(p, weight_field, 0)
            for p in points
        )
        
        return weighted_sum / total_weight
    
    def _weighted_average_by_inverse_spread(self, points: List[PricePoint]) -> float:
        """Calculate weighted average by inverse spread."""
        weights = []
        for p in points:
            if p.spread_bps > 0:
                weights.append(1 / p.spread_bps)
            else:
                weights.append(1.0)
        
        total_weight = sum(weights)
        
        if total_weight == 0:
            return statistics.mean([p.mid_price for p in points])
        
        weighted_sum = sum(
            p.mid_price * w
            for p, w in zip(points, weights)
        )
        
        return weighted_sum / total_weight
    
    def _best_bid_ask_mid(self, points: List[PricePoint]) -> float:
        """Calculate mid price from best bid and ask across exchanges."""
        best_bid = max((p.bid for p in points if p.bid > 0), default=0)
        best_ask = min((p.ask for p in points if p.ask > 0), default=0)
        
        if best_bid > 0 and best_ask > 0:
            return (best_bid + best_ask) / 2
        
        return statistics.mean([p.mid_price for p in points])
    
    def _aggregate_bid(self, points: List[PricePoint]) -> float:
        """Aggregate bid prices."""
        bids = [p.bid for p in points if p.bid > 0]
        if not bids:
            return 0.0
        return max(bids)  # Best bid
    
    def _aggregate_ask(self, points: List[PricePoint]) -> float:
        """Aggregate ask prices."""
        asks = [p.ask for p in points if p.ask > 0]
        if not asks:
            return 0.0
        return min(asks)  # Best ask
    
    def _calculate_confidence(
        self,
        points: List[PricePoint],
        std_dev: float,
        price: float
    ) -> float:
        """Calculate confidence score (0-100)."""
        scores = []
        
        # Source count score
        source_score = min(100, len(points) * 20)
        scores.append(source_score)
        
        # Consistency score (lower std dev = higher score)
        if price > 0:
            cv = (std_dev / price) * 100  # Coefficient of variation
            consistency_score = max(0, 100 - cv * 10)
            scores.append(consistency_score)
        
        # Spread score
        spreads = [p.spread_bps for p in points if p.spread_bps > 0]
        if spreads:
            avg_spread = statistics.mean(spreads)
            spread_score = max(0, 100 - avg_spread)
            scores.append(spread_score)
        
        return statistics.mean(scores) if scores else 50
    
    def get_aggregated_price(self, symbol: str) -> Optional[AggregatedPrice]:
        """Get aggregated price for a symbol."""
        return self._aggregated_prices.get(symbol)
    
    def get_all_prices(self) -> Dict[str, AggregatedPrice]:
        """Get all aggregated prices."""
        return self._aggregated_prices.copy()
    
    def remove_source(self, symbol: str, exchange: str):
        """Remove a price source."""
        if symbol in self._price_points:
            self._price_points[symbol].pop(exchange, None)


class UnifiedPriceFeed:
    """Unified price feed manager."""
    
    def __init__(
        self,
        cache: Optional[RedisCache] = None,
        aggregator: Optional[PriceFeedAggregator] = None
    ):
        self.cache = cache or get_cache()
        self.aggregator = aggregator or PriceFeedAggregator()
        self.normalizer = DataNormalizer()
        
        self._running = False
        self._subscribed_symbols: Set[str] = set()
        self._update_callbacks: Dict[str, List[Callable]] = defaultdict(list)
    
    async def start(self):
        """Start the price feed."""
        self._running = True
        logger.info("Unified Price Feed started")
    
    async def stop(self):
        """Stop the price feed."""
        self._running = False
        logger.info("Unified Price Feed stopped")
    
    def subscribe(
        self,
        symbol: str,
        callback: Callable[[AggregatedPrice], None]
    ):
        """Subscribe to price updates for a symbol."""
        self._subscribed_symbols.add(symbol)
        self._update_callbacks[symbol].append(callback)
        
        # Add aggregator handler
        self.aggregator.add_update_handler(
            lambda price: self._on_aggregated_price(price, symbol)
        )
        
        logger.info(f"Subscribed to {symbol}")
    
    def _on_aggregated_price(self, price: AggregatedPrice, symbol: str):
        """Handle aggregated price update."""
        if price.symbol != symbol:
            return
        
        # Cache the price
        asyncio.create_task(self._cache_price(price))
        
        # Notify callbacks
        for callback in self._update_callbacks.get(symbol, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(price))
                else:
                    callback(price)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    async def _cache_price(self, price: AggregatedPrice):
        """Cache the aggregated price."""
        try:
            await self.cache.set_ticker(price.symbol, price.to_dict(), ttl=5)
        except Exception as e:
            logger.error(f"Cache error: {e}")
    
    async def process_ticker(
        self,
        exchange: ExchangeType,
        ticker_data: Dict[str, Any]
    ):
        """Process ticker data from an exchange."""
        # Normalize the data
        normalized = self.normalizer.normalize(exchange, DataType.TICKER, ticker_data)
        
        if not normalized:
            return
        
        # Create price point
        price_point = PricePoint(
            symbol=normalized.symbol,
            exchange=normalized.exchange,
            timestamp=normalized.timestamp,
            price=normalized.price,
            bid=normalized.bid,
            ask=normalized.ask,
            volume_24h=normalized.volume
        )
        
        # Update aggregator
        await self.aggregator.update_price(price_point)
    
    async def process_orderbook(
        self,
        exchange: ExchangeType,
        orderbook_data: Dict[str, Any]
    ):
        """Process orderbook data from an exchange."""
        # Normalize the data
        normalized = self.normalizer.normalize(exchange, DataType.ORDERBOOK, orderbook_data)
        
        if not normalized:
            return
        
        # Create price point
        price_point = PricePoint(
            symbol=normalized.symbol,
            exchange=normalized.exchange,
            timestamp=normalized.timestamp,
            price=normalized.mid_price or 0,
            bid=normalized.best_bid or 0,
            ask=normalized.best_ask or 0
        )
        
        # Update aggregator
        await self.aggregator.update_price(price_point)
    
    async def process_trade(
        self,
        exchange: ExchangeType,
        trade_data: Dict[str, Any]
    ):
        """Process trade data from an exchange."""
        # Normalize the data
        normalized = self.normalizer.normalize(exchange, DataType.TRADE, trade_data)
        
        if not normalized:
            return
        
        # Create price point
        price_point = PricePoint(
            symbol=normalized.symbol,
            exchange=normalized.exchange,
            timestamp=normalized.timestamp,
            price=normalized.price
        )
        
        # Update aggregator
        await self.aggregator.update_price(price_point)
    
    def get_price(self, symbol: str) -> Optional[AggregatedPrice]:
        """Get current aggregated price for a symbol."""
        return self.aggregator.get_aggregated_price(symbol)
    
    async def get_cached_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get cached price for a symbol."""
        return await self.cache.get_ticker(symbol)
    
    def get_all_prices(self) -> Dict[str, AggregatedPrice]:
        """Get all current aggregated prices."""
        return self.aggregator.get_all_prices()


# Factory functions
def create_feed(
    cache: Optional[RedisCache] = None,
    aggregator: Optional[PriceFeedAggregator] = None
) -> UnifiedPriceFeed:
    """Create a unified price feed."""
    return UnifiedPriceFeed(cache, aggregator)


# Singleton instance
_feed_instance: Optional[UnifiedPriceFeed] = None


def get_feed() -> UnifiedPriceFeed:
    """Get singleton feed instance."""
    global _feed_instance
    if _feed_instance is None:
        _feed_instance = create_feed()
    return _feed_instance


if __name__ == "__main__":
    async def test_price_feed():
        """Test price feed."""
        feed = create_feed()
        await feed.start()
        
        def on_price(price):
            print(f"Price update: {price.symbol} = {price.price}")
        
        feed.subscribe("BTCUSDT", on_price)
        
        # Simulate updates from multiple exchanges
        await feed.process_ticker(ExchangeType.BINANCE, {
            "symbol": "BTCUSDT",
            "lastPrice": "50000",
            "bidPrice": "49999",
            "askPrice": "50001",
            "volume": "1000"
        })
        
        await feed.process_ticker(ExchangeType.COINBASE, {
            "product_id": "BTC-USDT",
            "price": "50010",
            "best_bid": "50005",
            "best_ask": "50015",
            "volume_24h": "800"
        })
        
        price = feed.get_price("BTCUSDT")
        if price:
            print(f"\nAggregated Price: {price.to_dict()}")
        
        await feed.stop()
    
    asyncio.run(test_price_feed())
