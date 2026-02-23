"""
TradeOS Slippage Model
======================
Models and tracks execution slippage for orders.
Provides slippage estimation based on market conditions.
"""

from __future__ import annotations

import asyncio
import logging
import statistics
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Deque, Dict, List, Optional, Tuple

from .models.order import Order, OrderSide, OrderFill, OrderType

logger = logging.getLogger(__name__)


@dataclass
class SlippageRecord:
    """Record of slippage for a single execution."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    expected_price: Decimal
    executed_price: Decimal
    quantity: Decimal
    slippage_bps: Decimal  # Slippage in basis points
    spread_at_execution: Decimal
    volume_24h: Decimal
    timestamp: datetime
    exchange: str
    
    @property
    def slippage_amount(self) -> Decimal:
        """Calculate absolute slippage amount."""
        return abs(self.executed_price - self.expected_price) * self.quantity
    
    @property
    def is_adverse(self) -> bool:
        """Check if slippage was adverse (unfavorable)."""
        if self.side == OrderSide.BUY:
            return self.executed_price > self.expected_price
        return self.executed_price < self.expected_price


@dataclass
class MarketConditions:
    """Market conditions at time of execution."""
    symbol: str
    timestamp: datetime
    spread: Decimal
    bid: Decimal
    ask: Decimal
    volume_24h: Decimal
    volatility_24h: Optional[Decimal] = None
    orderbook_depth: Optional[Decimal] = None  # Depth within 1% of mid
    
    @property
    def mid_price(self) -> Decimal:
        """Calculate mid price."""
        return (self.bid + self.ask) / 2
    
    @property
    def spread_bps(self) -> Decimal:
        """Calculate spread in basis points."""
        return (self.spread / self.mid_price) * Decimal("10000")


@dataclass
class SlippageEstimate:
    """Estimated slippage for an order."""
    expected_slippage_bps: Decimal
    confidence_interval_low: Decimal
    confidence_interval_high: Decimal
    estimated_fill_price: Decimal
    market_conditions: MarketConditions
    factors: Dict[str, Decimal] = field(default_factory=dict)
    
    @property
    def expected_slippage_amount(self, quantity: Decimal = Decimal("1")) -> Decimal:
        """Calculate expected slippage amount."""
        return (self.expected_slippage_bps / Decimal("10000")) * self.market_conditions.mid_price * quantity


class SlippageModel:
    """
    Models and tracks execution slippage.
    
    Features:
    - Historical slippage tracking
    - Slippage estimation based on market conditions
    - Per-symbol slippage profiles
    - Time-of-day patterns
    """
    
    DEFAULT_WINDOW_SIZE = 1000
    MIN_SAMPLES_FOR_ESTIMATE = 10
    
    def __init__(self, window_size: int = DEFAULT_WINDOW_SIZE):
        self.window_size = window_size
        self._records: Dict[str, Deque[SlippageRecord]] = {}  # By symbol
        self._records_by_exchange: Dict[str, Deque[SlippageRecord]] = {}
        self._market_conditions: Dict[str, MarketConditions] = {}
        self._symbol_profiles: Dict[str, Dict] = {}
        self._lock = asyncio.Lock()
        
        logger.info("SlippageModel initialized")
    
    async def record_slippage(self, order: Order, fill: OrderFill,
                               expected_price: Decimal,
                               market_conditions: Optional[MarketConditions] = None):
        """
        Record slippage for an order fill.
        
        Args:
            order: The order that was filled
            fill: The fill details
            expected_price: Expected execution price
            market_conditions: Market conditions at time of fill
        """
        if expected_price == 0:
            logger.warning(f"Cannot calculate slippage: expected price is zero")
            return
        
        # Calculate slippage in basis points
        price_diff = fill.price - expected_price
        slippage_bps = (price_diff / expected_price) * Decimal("10000")
        
        # Adjust for side (adverse slippage is positive)
        if order.side == OrderSide.SELL:
            slippage_bps = -slippage_bps
        
        record = SlippageRecord(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            expected_price=expected_price,
            executed_price=fill.price,
            quantity=fill.quantity,
            slippage_bps=slippage_bps,
            spread_at_execution=market_conditions.spread if market_conditions else Decimal("0"),
            volume_24h=market_conditions.volume_24h if market_conditions else Decimal("0"),
            timestamp=datetime.utcnow(),
            exchange=order.exchange
        )
        
        async with self._lock:
            # Store by symbol
            if order.symbol not in self._records:
                self._records[order.symbol] = deque(maxlen=self.window_size)
            self._records[order.symbol].append(record)
            
            # Store by exchange
            if order.exchange not in self._records_by_exchange:
                self._records_by_exchange[order.exchange] = deque(maxlen=self.window_size)
            self._records_by_exchange[order.exchange].append(record)
        
        logger.debug(
            f"Recorded slippage for {order.symbol}: {slippage_bps:.2f} bps "
            f"(expected: {expected_price}, executed: {fill.price})"
        )
    
    async def update_market_conditions(self, conditions: MarketConditions):
        """Update current market conditions for a symbol."""
        async with self._lock:
            self._market_conditions[conditions.symbol] = conditions
    
    async def estimate_slippage(self, symbol: str, side: OrderSide,
                                 quantity: Decimal,
                                 order_type: OrderType = OrderType.MARKET,
                                 exchange: Optional[str] = None) -> Optional[SlippageEstimate]:
        """
        Estimate slippage for a potential order.
        
        Args:
            symbol: Trading symbol
            side: Order side
            quantity: Order quantity
            order_type: Type of order
            exchange: Exchange name (optional)
            
        Returns:
            SlippageEstimate or None if insufficient data
        """
        async with self._lock:
            records = self._records.get(symbol, deque())
            market_conditions = self._market_conditions.get(symbol)
        
        if len(records) < self.MIN_SAMPLES_FOR_ESTIMATE:
            logger.debug(f"Insufficient slippage data for {symbol}")
            return None
        
        # Filter relevant records
        relevant_records = [
            r for r in records
            if r.side == side and r.order_type == order_type
        ]
        
        if len(relevant_records) < self.MIN_SAMPLES_FOR_ESTIMATE // 2:
            # Fall back to all records for this symbol
            relevant_records = list(records)
        
        # Calculate statistics
        slippage_values = [r.slippage_bps for r in relevant_records]
        mean_slippage = statistics.mean(slippage_values)
        
        # Adjust for current market conditions
        adjustment = Decimal("0")
        if market_conditions:
            # Wider spread = higher slippage
            spread_factor = market_conditions.spread_bps / Decimal("10")
            
            # Lower volume = higher slippage
            avg_volume = statistics.mean([r.volume_24h for r in relevant_records])
            if avg_volume > 0:
                volume_factor = (avg_volume / market_conditions.volume_24h - Decimal("1")) * Decimal("5")
            else:
                volume_factor = Decimal("0")
            
            adjustment = spread_factor + volume_factor
        
        # Adjust for order size (larger orders = more slippage)
        avg_quantity = statistics.mean([r.quantity for r in relevant_records])
        if avg_quantity > 0:
            size_factor = (quantity / avg_quantity - Decimal("1")) * Decimal("2")
        else:
            size_factor = Decimal("0")
        
        expected_slippage = mean_slippage + adjustment + size_factor
        
        # Calculate confidence interval
        if len(slippage_values) > 1:
            std_dev = statistics.stdev(slippage_values)
            confidence_low = expected_slippage - std_dev * Decimal("2")
            confidence_high = expected_slippage + std_dev * Decimal("2")
        else:
            confidence_low = expected_slippage * Decimal("0.5")
            confidence_high = expected_slippage * Decimal("1.5")
        
        # Calculate estimated fill price
        if market_conditions:
            base_price = market_conditions.mid_price
            slippage_factor = expected_slippage / Decimal("10000")
            if side == OrderSide.BUY:
                estimated_fill = base_price * (Decimal("1") + slippage_factor)
            else:
                estimated_fill = base_price * (Decimal("1") - slippage_factor)
        else:
            estimated_fill = Decimal("0")
        
        return SlippageEstimate(
            expected_slippage_bps=max(expected_slippage, Decimal("0")),
            confidence_interval_low=max(confidence_low, Decimal("0")),
            confidence_interval_high=confidence_high,
            estimated_fill_price=estimated_fill,
            market_conditions=market_conditions or MarketConditions(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                spread=Decimal("0"),
                bid=Decimal("0"),
                ask=Decimal("0"),
                volume_24h=Decimal("0")
            ),
            factors={
                "spread_adjustment": adjustment,
                "size_adjustment": size_factor,
                "sample_size": Decimal(len(relevant_records)),
            }
        )
    
    async def get_slippage_stats(self, symbol: str, 
                                  lookback_days: int = 30) -> Dict:
        """
        Get slippage statistics for a symbol.
        
        Args:
            symbol: Trading symbol
            lookback_days: Number of days to look back
            
        Returns:
            Dictionary of statistics
        """
        async with self._lock:
            records = list(self._records.get(symbol, deque()))
        
        if not records:
            return {"error": "No slippage data available"}
        
        cutoff_date = datetime.utcnow() - timedelta(days=lookback_days)
        recent_records = [r for r in records if r.timestamp > cutoff_date]
        
        if not recent_records:
            return {"error": "No recent slippage data available"}
        
        slippage_values = [r.slippage_bps for r in recent_records]
        
        stats = {
            "symbol": symbol,
            "sample_size": len(recent_records),
            "lookback_days": lookback_days,
            "mean_slippage_bps": round(statistics.mean(slippage_values), 4),
            "median_slippage_bps": round(statistics.median(slippage_values), 4),
            "min_slippage_bps": round(min(slippage_values), 4),
            "max_slippage_bps": round(max(slippage_values), 4),
        }
        
        if len(slippage_values) > 1:
            stats["std_dev"] = round(statistics.stdev(slippage_values), 4)
        
        # Breakdown by order type
        by_order_type = {}
        for order_type in OrderType:
            type_records = [r for r in recent_records if r.order_type == order_type]
            if type_records:
                by_order_type[order_type.value] = {
                    "count": len(type_records),
                    "mean_slippage_bps": round(
                        statistics.mean([r.slippage_bps for r in type_records]), 4
                    )
                }
        stats["by_order_type"] = by_order_type
        
        # Breakdown by side
        buy_records = [r for r in recent_records if r.side == OrderSide.BUY]
        sell_records = [r for r in recent_records if r.side == OrderSide.SELL]
        
        stats["by_side"] = {
            "buy": {
                "count": len(buy_records),
                "mean_slippage_bps": round(
                    statistics.mean([r.slippage_bps for r in buy_records]), 4
                ) if buy_records else 0
            },
            "sell": {
                "count": len(sell_records),
                "mean_slippage_bps": round(
                    statistics.mean([r.slippage_bps for r in sell_records]), 4
                ) if sell_records else 0
            }
        }
        
        return stats
    
    async def get_adverse_slippage_rate(self, symbol: str,
                                         lookback_days: int = 30) -> Decimal:
        """
        Calculate the rate of adverse slippage for a symbol.
        
        Args:
            symbol: Trading symbol
            lookback_days: Number of days to look back
            
        Returns:
            Percentage of fills with adverse slippage
        """
        async with self._lock:
            records = list(self._records.get(symbol, deque()))
        
        cutoff_date = datetime.utcnow() - timedelta(days=lookback_days)
        recent_records = [r for r in records if r.timestamp > cutoff_date]
        
        if not recent_records:
            return Decimal("0")
        
        adverse_count = sum(1 for r in recent_records if r.is_adverse)
        return Decimal(adverse_count) / Decimal(len(recent_records)) * Decimal("100")
    
    async def generate_report(self, symbols: Optional[List[str]] = None) -> Dict:
        """
        Generate comprehensive slippage report.
        
        Args:
            symbols: List of symbols to include (None for all)
            
        Returns:
            Report dictionary
        """
        async with self._lock:
            if symbols is None:
                symbols = list(self._records.keys())
        
        report = {
            "generated_at": datetime.utcnow().isoformat(),
            "symbols": {},
            "summary": {
                "total_records": 0,
                "symbols_tracked": len(symbols),
            }
        }
        
        all_slippages = []
        
        for symbol in symbols:
            stats = await self.get_slippage_stats(symbol)
            report["symbols"][symbol] = stats
            
            if "mean_slippage_bps" in stats:
                all_slippages.append(stats["mean_slippage_bps"])
            
            async with self._lock:
                report["summary"]["total_records"] += len(self._records.get(symbol, []))
        
        if all_slippages:
            report["summary"]["overall_mean_slippage_bps"] = round(
                statistics.mean(all_slippages), 4
            )
        
        return report
    
    async def clear_data(self, symbol: Optional[str] = None):
        """Clear slippage data."""
        async with self._lock:
            if symbol:
                if symbol in self._records:
                    del self._records[symbol]
            else:
                self._records.clear()
                self._records_by_exchange.clear()


# Global slippage model instance
global_slippage_model = SlippageModel()
