"""
TradeOS OHLC Builder
Build OHLC (Open, High, Low, Close) candles from tick data.

Features:
- Build candles from trade ticks
- Multiple timeframe support
- Automatic timeframe aggregation
- Volume tracking
- VWAP calculation
- Gap filling
- Real-time and historical modes
"""

import asyncio
import logging
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from collections import defaultdict
import time

from ..config.symbols import get_timeframe_seconds, TIMEFRAMES

logger = logging.getLogger(__name__)


@dataclass
class OHLCV:
    """OHLCV candle data structure."""
    timestamp: int  # Candle open timestamp (milliseconds)
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
    
    # Metadata
    timeframe: str = "1m"
    symbol: str = ""
    closed: bool = False
    
    def __post_init__(self):
        if self.vwap == 0.0 and self.volume > 0:
            self.vwap = self.quote_volume / self.volume if self.volume > 0 else self.close
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
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
            "timeframe": self.timeframe,
            "symbol": self.symbol,
            "closed": self.closed,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OHLCV":
        """Create from dictionary."""
        return cls(
            timestamp=data["timestamp"],
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            volume=data["volume"],
            quote_volume=data.get("quote_volume", 0.0),
            trades=data.get("trades", 0),
            taker_buy_volume=data.get("taker_buy_volume", 0.0),
            taker_buy_quote_volume=data.get("taker_buy_quote_volume", 0.0),
            vwap=data.get("vwap", 0.0),
            timeframe=data.get("timeframe", "1m"),
            symbol=data.get("symbol", ""),
            closed=data.get("closed", False),
        )


@dataclass
class Tick:
    """Trade tick data structure."""
    timestamp: int  # Milliseconds
    price: float
    quantity: float
    side: str = "buy"  # 'buy' or 'sell'
    trade_id: Optional[str] = None
    quote_quantity: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "price": self.price,
            "quantity": self.quantity,
            "side": self.side,
            "trade_id": self.trade_id,
            "quote_quantity": self.quote_quantity,
        }


class CandleBuilder:
    """Build candles from tick data for a single timeframe."""
    
    def __init__(
        self,
        symbol: str,
        timeframe: str,
        on_candle_close: Optional[Callable[[OHLCV], None]] = None
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.interval_ms = get_timeframe_seconds(timeframe) * 1000
        self.on_candle_close = on_candle_close
        
        self._current_candle: Optional[OHLCV] = None
        self._candles: List[OHLCV] = []
        self._last_tick_time: Optional[int] = None
        self._lock = asyncio.Lock()
    
    def _get_candle_timestamp(self, tick_timestamp: int) -> int:
        """Get the candle timestamp for a tick timestamp."""
        return (tick_timestamp // self.interval_ms) * self.interval_ms
    
    async def process_tick(self, tick: Tick) -> Optional[OHLCV]:
        """Process a new tick and update current candle."""
        async with self._lock:
            candle_ts = self._get_candle_timestamp(tick.timestamp)
            
            # Check if we need to start a new candle
            if self._current_candle is None or candle_ts > self._current_candle.timestamp:
                # Close previous candle
                if self._current_candle is not None:
                    self._current_candle.closed = True
                    self._candles.append(self._current_candle)
                    
                    if self.on_candle_close:
                        if asyncio.iscoroutinefunction(self.on_candle_close):
                            asyncio.create_task(self.on_candle_close(self._current_candle))
                        else:
                            self.on_candle_close(self._current_candle)
                
                # Start new candle
                self._current_candle = OHLCV(
                    timestamp=candle_ts,
                    open=tick.price,
                    high=tick.price,
                    low=tick.price,
                    close=tick.price,
                    volume=tick.quantity,
                    quote_volume=tick.quote_quantity,
                    trades=1,
                    taker_buy_volume=tick.quantity if tick.side == "buy" else 0,
                    taker_buy_quote_volume=tick.quote_quantity if tick.side == "buy" else 0,
                    timeframe=self.timeframe,
                    symbol=self.symbol,
                    closed=False
                )
            else:
                # Update current candle
                self._current_candle.high = max(self._current_candle.high, tick.price)
                self._current_candle.low = min(self._current_candle.low, tick.price)
                self._current_candle.close = tick.price
                self._current_candle.volume += tick.quantity
                self._current_candle.quote_volume += tick.quote_quantity
                self._current_candle.trades += 1
                
                if tick.side == "buy":
                    self._current_candle.taker_buy_volume += tick.quantity
                    self._current_candle.taker_buy_quote_volume += tick.quote_quantity
                
                # Recalculate VWAP
                if self._current_candle.volume > 0:
                    self._current_candle.vwap = (
                        self._current_candle.quote_volume / self._current_candle.volume
                    )
            
            self._last_tick_time = tick.timestamp
            return self._current_candle
    
    async def process_ticks(self, ticks: List[Tick]) -> List[OHLCV]:
        """Process multiple ticks."""
        candles = []
        for tick in ticks:
            candle = await self.process_tick(tick)
            if candle:
                candles.append(candle)
        return candles
    
    def get_current_candle(self) -> Optional[OHLCV]:
        """Get the current (incomplete) candle."""
        return self._current_candle
    
    def get_closed_candles(self, count: Optional[int] = None) -> List[OHLCV]:
        """Get closed candles."""
        candles = self._candles.copy()
        if count:
            candles = candles[-count:]
        return candles
    
    def get_all_candles(self, count: Optional[int] = None) -> List[OHLCV]:
        """Get all candles including current."""
        candles = self._candles.copy()
        if self._current_candle:
            candles.append(self._current_candle)
        if count:
            candles = candles[-count:]
        return candles
    
    def clear(self) -> None:
        """Clear all candles."""
        self._candles.clear()
        self._current_candle = None
        self._last_tick_time = None


class MultiTimeframeBuilder:
    """Build candles for multiple timeframes simultaneously."""
    
    def __init__(
        self,
        symbol: str,
        timeframes: List[str],
        on_candle_close: Optional[Callable[[OHLCV], None]] = None
    ):
        self.symbol = symbol
        self.timeframes = timeframes
        self.on_candle_close = on_candle_close
        
        self._builders: Dict[str, CandleBuilder] = {}
        self._init_builders()
    
    def _init_builders(self):
        """Initialize candle builders for each timeframe."""
        for tf in self.timeframes:
            self._builders[tf] = CandleBuilder(
                symbol=self.symbol,
                timeframe=tf,
                on_candle_close=self.on_candle_close
            )
    
    async def process_tick(self, tick: Tick) -> Dict[str, OHLCV]:
        """Process a tick across all timeframes."""
        results = {}
        for tf, builder in self._builders.items():
            candle = await builder.process_tick(tick)
            if candle:
                results[tf] = candle
        return results
    
    async def process_ticks(self, ticks: List[Tick]) -> Dict[str, List[OHLCV]]:
        """Process multiple ticks across all timeframes."""
        results = defaultdict(list)
        for tick in ticks:
            for tf, candle in (await self.process_tick(tick)).items():
                results[tf].append(candle)
        return dict(results)
    
    def get_current_candles(self) -> Dict[str, Optional[OHLCV]]:
        """Get current candles for all timeframes."""
        return {tf: builder.get_current_candle() for tf, builder in self._builders.items()}
    
    def get_closed_candles(self, timeframe: str, count: Optional[int] = None) -> List[OHLCV]:
        """Get closed candles for a specific timeframe."""
        if timeframe in self._builders:
            return self._builders[timeframe].get_closed_candles(count)
        return []
    
    def get_all_candles(self, timeframe: str, count: Optional[int] = None) -> List[OHLCV]:
        """Get all candles for a specific timeframe."""
        if timeframe in self._builders:
            return self._builders[timeframe].get_all_candles(count)
        return []
    
    def clear(self, timeframe: Optional[str] = None) -> None:
        """Clear candles."""
        if timeframe:
            if timeframe in self._builders:
                self._builders[timeframe].clear()
        else:
            for builder in self._builders.values():
                builder.clear()


class OHLCBuilder:
    """Main OHLC builder managing multiple symbols and timeframes."""
    
    def __init__(
        self,
        default_timeframes: Optional[List[str]] = None,
        on_candle_close: Optional[Callable[[OHLCV], None]] = None
    ):
        self.default_timeframes = default_timeframes or ["1m", "5m", "15m", "1h", "4h", "1d"]
        self.on_candle_close = on_candle_close
        
        self._builders: Dict[str, MultiTimeframeBuilder] = {}
        self._lock = asyncio.Lock()
    
    async def add_symbol(
        self,
        symbol: str,
        timeframes: Optional[List[str]] = None
    ) -> MultiTimeframeBuilder:
        """Add a symbol for candle building."""
        async with self._lock:
            if symbol not in self._builders:
                tfs = timeframes or self.default_timeframes
                self._builders[symbol] = MultiTimeframeBuilder(
                    symbol=symbol,
                    timeframes=tfs,
                    on_candle_close=self.on_candle_close
                )
                logger.info(f"Added symbol {symbol} with timeframes: {tfs}")
            
            return self._builders[symbol]
    
    async def remove_symbol(self, symbol: str) -> bool:
        """Remove a symbol."""
        async with self._lock:
            if symbol in self._builders:
                del self._builders[symbol]
                logger.info(f"Removed symbol {symbol}")
                return True
            return False
    
    async def process_trade(
        self,
        symbol: str,
        timestamp: Union[int, float],
        price: float,
        quantity: float,
        side: str = "buy",
        trade_id: Optional[str] = None,
        quote_quantity: float = 0.0
    ) -> Dict[str, OHLCV]:
        """Process a trade for a symbol."""
        # Ensure symbol exists
        if symbol not in self._builders:
            await self.add_symbol(symbol)
        
        # Convert timestamp to milliseconds if needed
        ts = int(timestamp * 1000) if timestamp < 1e12 else int(timestamp)
        
        tick = Tick(
            timestamp=ts,
            price=float(price),
            quantity=float(quantity),
            side=side,
            trade_id=trade_id,
            quote_quantity=float(quote_quantity) if quote_quantity else float(price) * float(quantity)
        )
        
        return await self._builders[symbol].process_tick(tick)
    
    async def process_trades(
        self,
        symbol: str,
        trades: List[Dict[str, Any]]
    ) -> Dict[str, List[OHLCV]]:
        """Process multiple trades for a symbol."""
        if symbol not in self._builders:
            await self.add_symbol(symbol)
        
        ticks = []
        for trade in trades:
            ts = trade.get("timestamp", trade.get("time", time.time() * 1000))
            ts = int(ts * 1000) if ts < 1e12 else int(ts)
            
            ticks.append(Tick(
                timestamp=ts,
                price=float(trade.get("price", 0)),
                quantity=float(trade.get("quantity", trade.get("size", 0))),
                side=trade.get("side", "buy"),
                trade_id=trade.get("trade_id", trade.get("id")),
                quote_quantity=float(trade.get("quote_quantity", trade.get("quoteQty", 0)))
            ))
        
        return await self._builders[symbol].process_ticks(ticks)
    
    def get_current_candles(self, symbol: str) -> Dict[str, Optional[OHLCV]]:
        """Get current candles for a symbol."""
        if symbol in self._builders:
            return self._builders[symbol].get_current_candles()
        return {}
    
    def get_closed_candles(
        self,
        symbol: str,
        timeframe: str,
        count: Optional[int] = None
    ) -> List[OHLCV]:
        """Get closed candles for a symbol and timeframe."""
        if symbol in self._builders:
            return self._builders[symbol].get_closed_candles(timeframe, count)
        return []
    
    def get_all_candles(
        self,
        symbol: str,
        timeframe: str,
        count: Optional[int] = None
    ) -> List[OHLCV]:
        """Get all candles for a symbol and timeframe."""
        if symbol in self._builders:
            return self._builders[symbol].get_all_candles(timeframe, count)
        return []
    
    def clear(self, symbol: Optional[str] = None, timeframe: Optional[str] = None):
        """Clear candles."""
        if symbol:
            if symbol in self._builders:
                self._builders[symbol].clear(timeframe)
        else:
            for builder in self._builders.values():
                builder.clear(timeframe)
    
    def get_symbols(self) -> List[str]:
        """Get all symbols being tracked."""
        return list(self._builders.keys())


class CandleAggregator:
    """Aggregate lower timeframe candles into higher timeframes."""
    
    @staticmethod
    def aggregate(candles: List[OHLCV], target_timeframe: str) -> List[OHLCV]:
        """Aggregate candles to a higher timeframe."""
        if not candles:
            return []
        
        target_interval_ms = get_timeframe_seconds(target_timeframe) * 1000
        source_interval_ms = get_timeframe_seconds(candles[0].timeframe) * 1000
        
        if target_interval_ms < source_interval_ms:
            raise ValueError("Cannot aggregate to a lower timeframe")
        
        if target_interval_ms == source_interval_ms:
            return candles.copy()
        
        aggregated: Dict[int, OHLCV] = {}
        
        for candle in candles:
            # Calculate target timestamp
            target_ts = (candle.timestamp // target_interval_ms) * target_interval_ms
            
            if target_ts not in aggregated:
                aggregated[target_ts] = OHLCV(
                    timestamp=target_ts,
                    open=candle.open,
                    high=candle.high,
                    low=candle.low,
                    close=candle.close,
                    volume=candle.volume,
                    quote_volume=candle.quote_volume,
                    trades=candle.trades,
                    taker_buy_volume=candle.taker_buy_volume,
                    taker_buy_quote_volume=candle.taker_buy_quote_volume,
                    timeframe=target_timeframe,
                    symbol=candle.symbol,
                    closed=candle.closed
                )
            else:
                agg = aggregated[target_ts]
                agg.high = max(agg.high, candle.high)
                agg.low = min(agg.low, candle.low)
                agg.close = candle.close
                agg.volume += candle.volume
                agg.quote_volume += candle.quote_volume
                agg.trades += candle.trades
                agg.taker_buy_volume += candle.taker_buy_volume
                agg.taker_buy_quote_volume += candle.taker_buy_quote_volume
                agg.closed = candle.closed and agg.closed
        
        # Recalculate VWAP for aggregated candles
        for agg in aggregated.values():
            if agg.volume > 0:
                agg.vwap = agg.quote_volume / agg.volume
        
        return sorted(aggregated.values(), key=lambda c: c.timestamp)
    
    @staticmethod
    def resample(
        candles: List[OHLCV],
        target_timeframe: str,
        gap_fill: bool = False
    ) -> List[OHLCV]:
        """Resample candles to a different timeframe with optional gap filling."""
        aggregated = CandleAggregator.aggregate(candles, target_timeframe)
        
        if not gap_fill or len(aggregated) < 2:
            return aggregated
        
        # Fill gaps
        target_interval_ms = get_timeframe_seconds(target_timeframe) * 1000
        filled = []
        
        for i, candle in enumerate(aggregated):
            filled.append(candle)
            
            # Check for gap to next candle
            if i < len(aggregated) - 1:
                next_ts = aggregated[i + 1].timestamp
                expected_next_ts = candle.timestamp + target_interval_ms
                
                while expected_next_ts < next_ts:
                    # Fill gap with previous close
                    filled.append(OHLCV(
                        timestamp=expected_next_ts,
                        open=candle.close,
                        high=candle.close,
                        low=candle.close,
                        close=candle.close,
                        volume=0,
                        quote_volume=0,
                        trades=0,
                        timeframe=target_timeframe,
                        symbol=candle.symbol,
                        closed=True
                    ))
                    expected_next_ts += target_interval_ms
        
        return filled


class RealtimeOHLCManager:
    """Manager for real-time OHLC building across multiple symbols."""
    
    def __init__(
        self,
        timeframes: Optional[List[str]] = None,
        on_candle_close: Optional[Callable[[OHLCV], None]] = None
    ):
        self.builder = OHLCBuilder(timeframes, on_candle_close)
        self._running = False
        self._check_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the manager."""
        self._running = True
        self._check_task = asyncio.create_task(self._candle_check_loop())
        logger.info("Realtime OHLC Manager started")
    
    async def stop(self):
        """Stop the manager."""
        self._running = False
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
        logger.info("Realtime OHLC Manager stopped")
    
    async def _candle_check_loop(self):
        """Periodically check for candle closures."""
        while self._running:
            try:
                # Check every second
                await asyncio.sleep(1)
                
                # Force close candles that should be closed
                current_time_ms = int(time.time() * 1000)
                
                for symbol in self.builder.get_symbols():
                    for tf, candle in self.builder.get_current_candles(symbol).items():
                        if candle and not candle.closed:
                            interval_ms = get_timeframe_seconds(tf) * 1000
                            candle_end_time = candle.timestamp + interval_ms
                            
                            if current_time_ms >= candle_end_time:
                                # Candle should be closed
                                candle.closed = True
                                if self.builder.on_candle_close:
                                    if asyncio.iscoroutinefunction(self.builder.on_candle_close):
                                        asyncio.create_task(
                                            self.builder.on_candle_close(candle)
                                        )
                                    else:
                                        self.builder.on_candle_close(candle)
                                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Candle check loop error: {e}")
    
    async def add_symbol(self, symbol: str, timeframes: Optional[List[str]] = None):
        """Add a symbol."""
        return await self.builder.add_symbol(symbol, timeframes)
    
    async def process_trade(self, **kwargs) -> Dict[str, OHLCV]:
        """Process a trade."""
        return await self.builder.process_trade(**kwargs)


# Factory functions
def create_builder(
    timeframes: Optional[List[str]] = None,
    on_candle_close: Optional[Callable[[OHLCV], None]] = None
) -> OHLCBuilder:
    """Create an OHLC builder."""
    return OHLCBuilder(timeframes, on_candle_close)


def create_realtime_manager(
    timeframes: Optional[List[str]] = None,
    on_candle_close: Optional[Callable[[OHLCV], None]] = None
) -> RealtimeOHLCManager:
    """Create a real-time OHLC manager."""
    return RealtimeOHLCManager(timeframes, on_candle_close)


# Singleton instance
_builder_instance: Optional[OHLCBuilder] = None


def get_builder() -> OHLCBuilder:
    """Get singleton builder instance."""
    global _builder_instance
    if _builder_instance is None:
        _builder_instance = create_builder()
    return _builder_instance


if __name__ == "__main__":
    async def test_ohlc_builder():
        """Test OHLC builder."""
        
        def on_close(candle):
            print(f"Candle closed: {candle.symbol} {candle.timeframe} - O:{candle.open} H:{candle.high} L:{candle.low} C:{candle.close}")
        
        builder = create_realtime_manager(
            timeframes=["1m", "5m"],
            on_candle_close=on_close
        )
        
        await builder.start()
        
        # Simulate trades
        base_time = int(time.time() * 1000)
        base_time = (base_time // 60000) * 60000  # Round to minute
        
        for i in range(10):
            await builder.process_trade(
                symbol="BTCUSDT",
                timestamp=base_time + i * 5000,  # Every 5 seconds
                price=50000 + i * 10,
                quantity=0.1,
                side="buy" if i % 2 == 0 else "sell"
            )
        
        # Get current candles
        candles = builder.builder.get_current_candles("BTCUSDT")
        print(f"\nCurrent candles: {candles}")
        
        await asyncio.sleep(2)
        await builder.stop()
    
    asyncio.run(test_ohlc_builder())
