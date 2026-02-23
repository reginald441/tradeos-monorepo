"""
TradeOS Market Microstructure Processor
Market microstructure signal detection and analysis.

Features:
- Bid-ask spread analysis
- Volume profile
- Order flow analysis
- Trade intensity
- Price impact
- Liquidity metrics
- Market depth analysis
- Imbalance detection
"""

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from collections import defaultdict, deque
import statistics
import time
import math

logger = logging.getLogger(__name__)


@dataclass
class SpreadMetrics:
    """Bid-ask spread metrics."""
    symbol: str
    timestamp: int
    best_bid: float
    best_ask: float
    spread: float
    spread_bps: float  # Spread in basis points
    mid_price: float
    weighted_mid: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "spread": self.spread,
            "spread_bps": self.spread_bps,
            "mid_price": self.mid_price,
            "weighted_mid": self.weighted_mid,
        }


@dataclass
class VolumeProfile:
    """Volume profile metrics."""
    symbol: str
    timestamp: int
    total_volume: float
    buy_volume: float
    sell_volume: float
    buy_ratio: float
    sell_ratio: float
    avg_trade_size: float
    large_trade_threshold: float
    large_trade_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "total_volume": self.total_volume,
            "buy_volume": self.buy_volume,
            "sell_volume": self.sell_volume,
            "buy_ratio": self.buy_ratio,
            "sell_ratio": self.sell_ratio,
            "avg_trade_size": self.avg_trade_size,
            "large_trade_threshold": self.large_trade_threshold,
            "large_trade_count": self.large_trade_count,
        }


@dataclass
class OrderFlowMetrics:
    """Order flow metrics."""
    symbol: str
    timestamp: int
    trade_count: int
    trade_intensity: float  # trades per second
    buy_pressure: float
    sell_pressure: float
    net_flow: float
    cumulative_delta: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "trade_count": self.trade_count,
            "trade_intensity": self.trade_intensity,
            "buy_pressure": self.buy_pressure,
            "sell_pressure": self.sell_pressure,
            "net_flow": self.net_flow,
            "cumulative_delta": self.cumulative_delta,
        }


@dataclass
class LiquidityMetrics:
    """Liquidity metrics."""
    symbol: str
    timestamp: int
    bid_depth: float  # Total bid volume
    ask_depth: float  # Total ask volume
    total_depth: float
    depth_imbalance: float
    bid_depth_5: float  # Depth within 5% of best bid
    ask_depth_5: float  # Depth within 5% of best ask
    depth_ratio_5: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "bid_depth": self.bid_depth,
            "ask_depth": self.ask_depth,
            "total_depth": self.total_depth,
            "depth_imbalance": self.depth_imbalance,
            "bid_depth_5": self.bid_depth_5,
            "ask_depth_5": self.ask_depth_5,
            "depth_ratio_5": self.depth_ratio_5,
        }


@dataclass
class PriceImpactMetrics:
    """Price impact metrics."""
    symbol: str
    timestamp: int
    impact_1pct: float  # Price impact for 1% of volume
    impact_5pct: float  # Price impact for 5% of volume
    impact_10pct: float  # Price impact for 10% of volume
    avg_impact_per_volume: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "impact_1pct": self.impact_1pct,
            "impact_5pct": self.impact_5pct,
            "impact_10pct": self.impact_10pct,
            "avg_impact_per_volume": self.avg_impact_per_volume,
        }


@dataclass
class MicrostructureSignal:
    """Combined microstructure signal."""
    symbol: str
    timestamp: int
    spread_metrics: Optional[SpreadMetrics] = None
    volume_profile: Optional[VolumeProfile] = None
    order_flow: Optional[OrderFlowMetrics] = None
    liquidity: Optional[LiquidityMetrics] = None
    price_impact: Optional[PriceImpactMetrics] = None
    
    # Composite signals
    liquidity_score: float = 0.0  # 0-100, higher is more liquid
    pressure_signal: float = 0.0  # -100 to 100, negative = sell pressure
    volatility_signal: float = 0.0  # 0-100, higher = more volatile
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "spread_metrics": self.spread_metrics.to_dict() if self.spread_metrics else None,
            "volume_profile": self.volume_profile.to_dict() if self.volume_profile else None,
            "order_flow": self.order_flow.to_dict() if self.order_flow else None,
            "liquidity": self.liquidity.to_dict() if self.liquidity else None,
            "price_impact": self.price_impact.to_dict() if self.price_impact else None,
            "liquidity_score": self.liquidity_score,
            "pressure_signal": self.pressure_signal,
            "volatility_signal": self.volatility_signal,
        }


class SpreadAnalyzer:
    """Analyze bid-ask spreads."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._spread_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self._last_update: Dict[str, int] = {}
    
    def analyze(
        self,
        symbol: str,
        timestamp: int,
        bids: List[List[float]],
        asks: List[List[float]]
    ) -> Optional[SpreadMetrics]:
        """Analyze spread from orderbook."""
        if not bids or not asks:
            return None
        
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        bid_qty = float(bids[0][1])
        ask_qty = float(asks[0][1])
        
        spread = best_ask - best_bid
        mid_price = (best_bid + best_ask) / 2
        spread_bps = (spread / mid_price) * 10000 if mid_price > 0 else 0
        
        # Volume-weighted mid price
        total_qty = bid_qty + ask_qty
        weighted_mid = ((best_bid * ask_qty) + (best_ask * bid_qty)) / total_qty if total_qty > 0 else mid_price
        
        metrics = SpreadMetrics(
            symbol=symbol,
            timestamp=timestamp,
            best_bid=best_bid,
            best_ask=best_ask,
            spread=spread,
            spread_bps=spread_bps,
            mid_price=mid_price,
            weighted_mid=weighted_mid
        )
        
        # Store in history
        self._spread_history[symbol].append(metrics)
        self._last_update[symbol] = timestamp
        
        return metrics
    
    def get_average_spread(self, symbol: str) -> Optional[float]:
        """Get average spread for a symbol."""
        history = self._spread_history.get(symbol, [])
        if not history:
            return None
        return statistics.mean([s.spread for s in history])
    
    def get_spread_percentile(self, symbol: str, percentile: float = 50) -> Optional[float]:
        """Get spread percentile."""
        history = self._spread_history.get(symbol, [])
        if not history:
            return None
        
        spreads = sorted([s.spread for s in history])
        index = int(len(spreads) * percentile / 100)
        return spreads[index]


class VolumeProfiler:
    """Analyze volume profile."""
    
    def __init__(self, window_seconds: int = 60):
        self.window_seconds = window_seconds
        self._trades: Dict[str, List[Dict]] = defaultdict(list)
        self._large_trade_multiplier = 3.0  # Trades > 3x average are "large"
    
    def add_trade(self, symbol: str, trade: Dict[str, Any]):
        """Add a trade to the profiler."""
        self._trades[symbol].append({
            "timestamp": trade["timestamp"],
            "quantity": trade["quantity"],
            "side": trade.get("side", "buy"),
        })
        
        # Clean old trades
        cutoff = trade["timestamp"] - (self.window_seconds * 1000)
        self._trades[symbol] = [
            t for t in self._trades[symbol]
            if t["timestamp"] > cutoff
        ]
    
    def analyze(self, symbol: str, timestamp: int) -> Optional[VolumeProfile]:
        """Analyze volume profile."""
        trades = self._trades.get(symbol, [])
        if not trades:
            return None
        
        total_volume = sum(t["quantity"] for t in trades)
        buy_volume = sum(t["quantity"] for t in trades if t["side"] == "buy")
        sell_volume = total_volume - buy_volume
        
        buy_ratio = buy_volume / total_volume if total_volume > 0 else 0.5
        sell_ratio = 1 - buy_ratio
        
        avg_trade_size = total_volume / len(trades) if trades else 0
        large_trade_threshold = avg_trade_size * self._large_trade_multiplier
        large_trade_count = sum(1 for t in trades if t["quantity"] > large_trade_threshold)
        
        return VolumeProfile(
            symbol=symbol,
            timestamp=timestamp,
            total_volume=total_volume,
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            buy_ratio=buy_ratio,
            sell_ratio=sell_ratio,
            avg_trade_size=avg_trade_size,
            large_trade_threshold=large_trade_threshold,
            large_trade_count=large_trade_count
        )


class OrderFlowAnalyzer:
    """Analyze order flow."""
    
    def __init__(self, window_seconds: int = 60):
        self.window_seconds = window_seconds
        self._trades: Dict[str, List[Dict]] = defaultdict(list)
        self._cumulative_delta: Dict[str, float] = defaultdict(float)
    
    def add_trade(self, symbol: str, trade: Dict[str, Any]):
        """Add a trade to the analyzer."""
        self._trades[symbol].append({
            "timestamp": trade["timestamp"],
            "quantity": trade["quantity"],
            "side": trade.get("side", "buy"),
        })
        
        # Update cumulative delta
        delta = trade["quantity"] if trade.get("side") == "buy" else -trade["quantity"]
        self._cumulative_delta[symbol] += delta
        
        # Clean old trades
        cutoff = trade["timestamp"] - (self.window_seconds * 1000)
        self._trades[symbol] = [
            t for t in self._trades[symbol]
            if t["timestamp"] > cutoff
        ]
    
    def analyze(self, symbol: str, timestamp: int) -> Optional[OrderFlowMetrics]:
        """Analyze order flow."""
        trades = self._trades.get(symbol, [])
        if not trades:
            return None
        
        trade_count = len(trades)
        
        # Calculate trade intensity (trades per second)
        if len(trades) >= 2:
            time_span_ms = trades[-1]["timestamp"] - trades[0]["timestamp"]
            time_span_s = time_span_ms / 1000
            trade_intensity = trade_count / time_span_s if time_span_s > 0 else 0
        else:
            trade_intensity = 0
        
        # Calculate buy/sell pressure
        buy_volume = sum(t["quantity"] for t in trades if t["side"] == "buy")
        sell_volume = sum(t["quantity"] for t in trades if t["side"] == "sell")
        total_volume = buy_volume + sell_volume
        
        buy_pressure = buy_volume / total_volume if total_volume > 0 else 0.5
        sell_pressure = sell_volume / total_volume if total_volume > 0 else 0.5
        net_flow = buy_volume - sell_volume
        
        return OrderFlowMetrics(
            symbol=symbol,
            timestamp=timestamp,
            trade_count=trade_count,
            trade_intensity=trade_intensity,
            buy_pressure=buy_pressure,
            sell_pressure=sell_pressure,
            net_flow=net_flow,
            cumulative_delta=self._cumulative_delta.get(symbol, 0)
        )


class LiquidityAnalyzer:
    """Analyze market liquidity."""
    
    def __init__(self):
        self._last_orderbook: Dict[str, Dict] = {}
    
    def update_orderbook(self, symbol: str, orderbook: Dict[str, Any]):
        """Update stored orderbook."""
        self._last_orderbook[symbol] = orderbook
    
    def analyze(
        self,
        symbol: str,
        timestamp: int,
        bids: Optional[List[List[float]]] = None,
        asks: Optional[List[List[float]]] = None
    ) -> Optional[LiquidityMetrics]:
        """Analyze liquidity from orderbook."""
        # Use provided orderbook or stored one
        if bids is None or asks is None:
            orderbook = self._last_orderbook.get(symbol)
            if not orderbook:
                return None
            bids = orderbook.get("bids", [])
            asks = orderbook.get("asks", [])
        
        if not bids or not asks:
            return None
        
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        mid_price = (best_bid + best_ask) / 2
        
        # Calculate total depth
        bid_depth = sum(float(b[1]) for b in bids)
        ask_depth = sum(float(a[1]) for a in asks)
        total_depth = bid_depth + ask_depth
        
        # Calculate depth imbalance
        depth_imbalance = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0
        
        # Calculate depth within 5% of mid price
        depth_threshold = mid_price * 0.05
        
        bid_depth_5 = sum(
            float(b[1]) for b in bids
            if mid_price - float(b[0]) <= depth_threshold
        )
        
        ask_depth_5 = sum(
            float(a[1]) for a in asks
            if float(a[0]) - mid_price <= depth_threshold
        )
        
        depth_ratio_5 = bid_depth_5 / ask_depth_5 if ask_depth_5 > 0 else 1.0
        
        return LiquidityMetrics(
            symbol=symbol,
            timestamp=timestamp,
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            total_depth=total_depth,
            depth_imbalance=depth_imbalance,
            bid_depth_5=bid_depth_5,
            ask_depth_5=ask_depth_5,
            depth_ratio_5=depth_ratio_5
        )


class PriceImpactAnalyzer:
    """Analyze price impact."""
    
    def __init__(self):
        self._orderbook: Dict[str, Dict] = {}
    
    def update_orderbook(self, symbol: str, orderbook: Dict[str, Any]):
        """Update stored orderbook."""
        self._orderbook[symbol] = orderbook
    
    def analyze(
        self,
        symbol: str,
        timestamp: int,
        volume_percentile: float = 0.01
    ) -> Optional[PriceImpactMetrics]:
        """Analyze price impact from orderbook."""
        orderbook = self._orderbook.get(symbol)
        if not orderbook:
            return None
        
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])
        
        if not bids or not asks:
            return None
        
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        mid_price = (best_bid + best_ask) / 2
        
        # Calculate total depth
        total_bid_depth = sum(float(b[1]) for b in bids)
        total_ask_depth = sum(float(a[1]) for a in asks)
        
        # Calculate impact for different volume percentages
        impact_1pct = self._calculate_impact(bids, asks, mid_price, total_bid_depth * 0.01)
        impact_5pct = self._calculate_impact(bids, asks, mid_price, total_bid_depth * 0.05)
        impact_10pct = self._calculate_impact(bids, asks, mid_price, total_bid_depth * 0.10)
        
        # Average impact per unit volume
        avg_impact = impact_1pct / (total_bid_depth * 0.01) if total_bid_depth > 0 else 0
        
        return PriceImpactMetrics(
            symbol=symbol,
            timestamp=timestamp,
            impact_1pct=impact_1pct,
            impact_5pct=impact_5pct,
            impact_10pct=impact_10pct,
            avg_impact_per_volume=avg_impact
        )
    
    def _calculate_impact(
        self,
        bids: List[List[float]],
        asks: List[List[float]],
        mid_price: float,
        volume: float
    ) -> float:
        """Calculate price impact for a given volume."""
        # Simulate buying (walking up the ask side)
        remaining = volume
        weighted_sum = 0
        
        for ask in asks:
            price = float(ask[0])
            qty = float(ask[1])
            
            if remaining <= 0:
                break
            
            fill_qty = min(remaining, qty)
            weighted_sum += price * fill_qty
            remaining -= fill_qty
        
        if volume > remaining:  # Some volume was filled
            avg_fill_price = weighted_sum / (volume - remaining)
            return (avg_fill_price - mid_price) / mid_price * 10000  # In bps
        
        return 0


class MicrostructureProcessor:
    """Main processor for market microstructure analysis."""
    
    def __init__(self):
        self.spread_analyzer = SpreadAnalyzer()
        self.volume_profiler = VolumeProfiler()
        self.order_flow_analyzer = OrderFlowAnalyzer()
        self.liquidity_analyzer = LiquidityAnalyzer()
        self.price_impact_analyzer = PriceImpactAnalyzer()
        
        self._signal_handlers: List[Callable[[MicrostructureSignal], None]] = []
        self._last_signal: Dict[str, MicrostructureSignal] = {}
    
    def add_signal_handler(self, handler: Callable[[MicrostructureSignal], None]):
        """Add a handler for microstructure signals."""
        self._signal_handlers.append(handler)
    
    def process_orderbook(self, symbol: str, timestamp: int, orderbook: Dict[str, Any]):
        """Process orderbook update."""
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])
        
        # Update analyzers
        self.spread_analyzer.analyze(symbol, timestamp, bids, asks)
        self.liquidity_analyzer.update_orderbook(symbol, orderbook)
        self.price_impact_analyzer.update_orderbook(symbol, orderbook)
        
        # Generate signal
        self._generate_signal(symbol, timestamp)
    
    def process_trade(self, symbol: str, trade: Dict[str, Any]):
        """Process trade update."""
        # Update analyzers
        self.volume_profiler.add_trade(symbol, trade)
        self.order_flow_analyzer.add_trade(symbol, trade)
        
        # Generate signal
        self._generate_signal(symbol, trade["timestamp"])
    
    def _generate_signal(self, symbol: str, timestamp: int):
        """Generate microstructure signal."""
        # Get metrics from all analyzers
        spread_history = self.spread_analyzer._spread_history.get(symbol)
        spread_metrics = spread_history[-1] if spread_history else None
        
        volume_profile = self.volume_profiler.analyze(symbol, timestamp)
        order_flow = self.order_flow_analyzer.analyze(symbol, timestamp)
        liquidity = self.liquidity_analyzer.analyze(symbol, timestamp)
        price_impact = self.price_impact_analyzer.analyze(symbol, timestamp)
        
        # Calculate composite signals
        liquidity_score = self._calculate_liquidity_score(spread_metrics, liquidity, price_impact)
        pressure_signal = self._calculate_pressure_signal(volume_profile, order_flow, liquidity)
        volatility_signal = self._calculate_volatility_signal(spread_metrics, order_flow)
        
        signal = MicrostructureSignal(
            symbol=symbol,
            timestamp=timestamp,
            spread_metrics=spread_metrics,
            volume_profile=volume_profile,
            order_flow=order_flow,
            liquidity=liquidity,
            price_impact=price_impact,
            liquidity_score=liquidity_score,
            pressure_signal=pressure_signal,
            volatility_signal=volatility_signal
        )
        
        self._last_signal[symbol] = signal
        
        # Notify handlers
        for handler in self._signal_handlers:
            try:
                handler(signal)
            except Exception as e:
                logger.error(f"Signal handler error: {e}")
    
    def _calculate_liquidity_score(
        self,
        spread: Optional[SpreadMetrics],
        liquidity: Optional[LiquidityMetrics],
        price_impact: Optional[PriceImpactMetrics]
    ) -> float:
        """Calculate overall liquidity score (0-100)."""
        scores = []
        
        # Spread score (lower spread = higher score)
        if spread:
            spread_score = max(0, 100 - spread.spread_bps)
            scores.append(spread_score)
        
        # Depth score
        if liquidity:
            depth_score = min(100, liquidity.total_depth / 100)
            scores.append(depth_score)
        
        # Price impact score (lower impact = higher score)
        if price_impact:
            impact_score = max(0, 100 - price_impact.impact_1pct * 10)
            scores.append(impact_score)
        
        return statistics.mean(scores) if scores else 50
    
    def _calculate_pressure_signal(
        self,
        volume_profile: Optional[VolumeProfile],
        order_flow: Optional[OrderFlowMetrics],
        liquidity: Optional[LiquidityMetrics]
    ) -> float:
        """Calculate buy/sell pressure signal (-100 to 100)."""
        signals = []
        
        # Volume pressure
        if volume_profile:
            volume_signal = (volume_profile.buy_ratio - 0.5) * 200
            signals.append(volume_signal)
        
        # Order flow pressure
        if order_flow:
            flow_signal = (order_flow.buy_pressure - 0.5) * 200
            signals.append(flow_signal)
        
        # Depth imbalance
        if liquidity:
            depth_signal = liquidity.depth_imbalance * 100
            signals.append(depth_signal)
        
        return statistics.mean(signals) if signals else 0
    
    def _calculate_volatility_signal(
        self,
        spread: Optional[SpreadMetrics],
        order_flow: Optional[OrderFlowMetrics]
    ) -> float:
        """Calculate volatility signal (0-100)."""
        signals = []
        
        # Spread volatility
        if spread:
            spread_signal = min(100, spread.spread_bps * 2)
            signals.append(spread_signal)
        
        # Trade intensity
        if order_flow:
            intensity_signal = min(100, order_flow.trade_intensity * 10)
            signals.append(intensity_signal)
        
        return statistics.mean(signals) if signals else 0
    
    def get_last_signal(self, symbol: str) -> Optional[MicrostructureSignal]:
        """Get the last signal for a symbol."""
        return self._last_signal.get(symbol)


# Factory functions
def create_processor() -> MicrostructureProcessor:
    """Create a microstructure processor."""
    return MicrostructureProcessor()


# Singleton instance
_processor_instance: Optional[MicrostructureProcessor] = None


def get_processor() -> MicrostructureProcessor:
    """Get singleton processor instance."""
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = create_processor()
    return _processor_instance


if __name__ == "__main__":
    def test_microstructure():
        """Test microstructure processor."""
        processor = create_processor()
        
        def on_signal(signal):
            print(f"Signal: {signal.to_dict()}")
        
        processor.add_signal_handler(on_signal)
        
        # Simulate orderbook
        orderbook = {
            "bids": [["50000", "1.0"], ["49999", "2.0"], ["49998", "3.0"]],
            "asks": [["50001", "0.8"], ["50002", "1.5"], ["50003", "2.0"]]
        }
        
        processor.process_orderbook("BTCUSDT", int(time.time() * 1000), orderbook)
        
        # Simulate trades
        for i in range(10):
            trade = {
                "timestamp": int(time.time() * 1000),
                "quantity": 0.1 + i * 0.01,
                "side": "buy" if i % 2 == 0 else "sell"
            }
            processor.process_trade("BTCUSDT", trade)
        
        signal = processor.get_last_signal("BTCUSDT")
        if signal:
            print(f"\nFinal Signal:")
            print(f"  Liquidity Score: {signal.liquidity_score}")
            print(f"  Pressure Signal: {signal.pressure_signal}")
            print(f"  Volatility Signal: {signal.volatility_signal}")
    
    test_microstructure()
