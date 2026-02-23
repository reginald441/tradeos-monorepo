"""
TradeOS Paper Trading Mode
==========================
Simulated trading environment for backtesting and strategy development.
Provides realistic order fills without risking real capital.
"""

from __future__ import annotations

import asyncio
import logging
import random
from decimal import Decimal, ROUND_DOWN
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime

from .base_exchange import (
    BaseExchange, Balance, Ticker, OrderBook, OrderBookLevel,
    Position, ExchangeError, InvalidOrderError, InsufficientFundsError
)
from .models.order import (
    Order, OrderRequest, OrderFill, OrderState, OrderSide,
    OrderType, TimeInForce, OrderEvent, OrderEventType
)
from .slippage_model import global_slippage_model, SlippageEstimate
from .latency_tracker import LatencyType

logger = logging.getLogger(__name__)


class PaperTradingExchange(BaseExchange):
    """
    Paper trading exchange implementation.
    
    Simulates realistic trading conditions including:
    - Market price fills based on order book
    - Slippage modeling
    - Partial fills for large orders
    - Fee calculation
    - Balance tracking
    - Position management
    
    Uses real market data from a reference exchange while
    simulating order execution.
    """
    
    DEFAULT_INITIAL_BALANCE = Decimal("10000")
    DEFAULT_MAKER_FEE = Decimal("0.001")  # 0.1%
    DEFAULT_TAKER_FEE = Decimal("0.001")  # 0.1%
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("paper", config)
        
        # Configuration
        self.initial_balances = config.get("initial_balances", {
            "USDT": self.DEFAULT_INITIAL_BALANCE,
            "USD": self.DEFAULT_INITIAL_BALANCE,
        })
        self.maker_fee = Decimal(str(config.get("maker_fee", self.DEFAULT_MAKER_FEE)))
        self.taker_fee = Decimal(str(config.get("taker_fee", self.DEFAULT_TAKER_FEE)))
        self.enable_slippage = config.get("enable_slippage", True)
        self.slippage_model = config.get("slippage_model", global_slippage_model)
        self.fill_delay_ms = config.get("fill_delay_ms", 100)
        self.enable_partial_fills = config.get("enable_partial_fills", True)
        self.partial_fill_probability = config.get("partial_fill_probability", 0.1)
        
        # Reference exchange for market data
        self.reference_exchange: Optional[BaseExchange] = config.get("reference_exchange")
        
        # State
        self._balances: Dict[str, Balance] = {}
        self._orders: Dict[str, Order] = {}
        self._positions: Dict[str, Position] = {}
        self._order_history: List[Order] = []
        self._fills: List[OrderFill] = []
        self._trade_id_counter = 0
        
        # Price cache for symbols
        self._price_cache: Dict[str, Ticker] = {}
        self._orderbook_cache: Dict[str, OrderBook] = {}
        
        # Initialize balances
        self._initialize_balances()
        
        # Background tasks
        self._price_update_task: Optional[asyncio.Task] = None
        
        logger.info("Paper trading exchange initialized")
    
    def _initialize_balances(self):
        """Initialize account balances."""
        for asset, amount in self.initial_balances.items():
            self._balances[asset] = Balance(
                asset=asset,
                free=Decimal(str(amount)),
                locked=Decimal("0"),
                total=Decimal(str(amount))
            )
    
    # ==================== Connection ====================
    
    async def connect(self) -> bool:
        """Connect paper trading environment."""
        try:
            # Start price update loop if using reference exchange
            if self.reference_exchange:
                self._price_update_task = asyncio.create_task(self._price_update_loop())
            
            self.is_connected = True
            logger.info("Paper trading environment connected")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect paper trading: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect paper trading environment."""
        try:
            if self._price_update_task:
                self._price_update_task.cancel()
                try:
                    await self._price_update_task
                except asyncio.CancelledError:
                    pass
            
            self.is_connected = False
            logger.info("Paper trading environment disconnected")
            return True
            
        except Exception as e:
            logger.error(f"Error disconnecting paper trading: {e}")
            return False
    
    async def is_connection_healthy(self) -> bool:
        """Check if connection is healthy."""
        return self.is_connected
    
    async def _price_update_loop(self):
        """Update prices from reference exchange."""
        while True:
            try:
                # Get unique symbols from open orders
                symbols = set(order.symbol for order in self._orders.values())
                
                for symbol in symbols:
                    try:
                        ticker = await self.reference_exchange.get_ticker(symbol)
                        self._price_cache[symbol] = ticker
                        
                        orderbook = await self.reference_exchange.get_orderbook(symbol, 10)
                        self._orderbook_cache[symbol] = orderbook
                        
                        # Update slippage model
                        from .slippage_model import MarketConditions
                        await self.slippage_model.update_market_conditions(
                            MarketConditions(
                                symbol=symbol,
                                timestamp=datetime.utcnow(),
                                spread=ticker.spread,
                                bid=ticker.bid,
                                ask=ticker.ask,
                                volume_24h=ticker.volume_24h
                            )
                        )
                        
                    except Exception as e:
                        logger.debug(f"Error updating price for {symbol}: {e}")
                
                await asyncio.sleep(1)  # Update every second
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in price update loop: {e}")
                await asyncio.sleep(5)
    
    # ==================== Order Operations ====================
    
    async def place_order(self, order_request: OrderRequest) -> Order:
        """Place a simulated order."""
        timer_id = self.latency_tracker.start_timer(
            f"paper_order_{order_request.symbol}", LatencyType.ORDER_SUBMIT, "paper"
        )
        
        try:
            # Validate order
            await self._validate_order(order_request)
            
            # Create order
            order = order_request.to_order()
            order.exchange = "paper"
            order.state = OrderState.OPEN
            order.submitted_at = datetime.utcnow()
            
            # Lock balance for the order
            await self._lock_balance_for_order(order)
            
            # Store order
            self._orders[order.order_id] = order
            
            # Simulate fill
            asyncio.create_task(self._simulate_fill(order))
            
            self.latency_tracker.stop_timer(timer_id, True)
            
            logger.info(f"Paper order placed: {order.order_id}")
            return order
            
        except Exception as e:
            self.latency_tracker.stop_timer(timer_id, False, {"error": str(e)})
            raise
    
    async def _validate_order(self, order_request: OrderRequest):
        """Validate order request."""
        # Check symbol
        if not order_request.symbol:
            raise InvalidOrderError("Symbol is required")
        
        # Check quantity
        if order_request.quantity <= 0:
            raise InvalidOrderError("Quantity must be positive")
        
        # Check price for limit orders
        if order_request.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            if not order_request.price or order_request.price <= 0:
                raise InvalidOrderError("Price is required for limit orders")
        
        # Check balance
        await self._check_sufficient_balance(order_request)
    
    async def _check_sufficient_balance(self, order_request: OrderRequest):
        """Check if there's sufficient balance for the order."""
        if order_request.side == OrderSide.BUY:
            # Need quote currency
            quote_asset = self._get_quote_asset(order_request.symbol)
            balance = self._balances.get(quote_asset, Balance(quote_asset, Decimal("0"), Decimal("0"), Decimal("0")))
            
            # Estimate required amount
            if order_request.price:
                required = order_request.quantity * order_request.price
            else:
                # Use a high estimate for market orders
                required = order_request.quantity * Decimal("1000000")
            
            if balance.free < required:
                raise InsufficientFundsError(
                    f"Insufficient {quote_asset} balance. "
                    f"Required: {required}, Available: {balance.free}"
                )
        else:
            # Need base currency
            base_asset = self._get_base_asset(order_request.symbol)
            balance = self._balances.get(base_asset, Balance(base_asset, Decimal("0"), Decimal("0"), Decimal("0")))
            
            if balance.free < order_request.quantity:
                raise InsufficientFundsError(
                    f"Insufficient {base_asset} balance. "
                    f"Required: {order_request.quantity}, Available: {balance.free}"
                )
    
    async def _lock_balance_for_order(self, order: Order):
        """Lock balance for an order."""
        if order.side == OrderSide.BUY:
            quote_asset = self._get_quote_asset(order.symbol)
            balance = self._balances.get(quote_asset)
            if balance:
                # Lock estimated amount
                if order.price:
                    lock_amount = order.quantity * order.price
                else:
                    # For market orders, lock based on current price
                    ticker = self._price_cache.get(order.symbol)
                    if ticker:
                        lock_amount = order.quantity * ticker.ask
                    else:
                        lock_amount = order.quantity * Decimal("1000000")
                
                balance.free -= lock_amount
                balance.locked += lock_amount
    
    async def _unlock_balance_for_order(self, order: Order):
        """Unlock balance for a cancelled/expired order."""
        if order.side == OrderSide.BUY:
            quote_asset = self._get_quote_asset(order.symbol)
            balance = self._balances.get(quote_asset)
            if balance:
                # Unlock remaining amount
                if order.price:
                    unlock_amount = order.remaining_quantity * order.price
                else:
                    unlock_amount = order.remaining_quantity * Decimal("100")
                
                balance.free += unlock_amount
                balance.locked -= unlock_amount
    
    async def _simulate_fill(self, order: Order):
        """Simulate order fill."""
        try:
            # Simulate network delay
            await asyncio.sleep(self.fill_delay_ms / 1000)
            
            # Get current market data
            ticker = self._price_cache.get(order.symbol)
            orderbook = self._orderbook_cache.get(order.symbol)
            
            if not ticker and not orderbook:
                # No market data available, reject order
                order.update_state(OrderState.REJECTED, "No market data available")
                await self._unlock_balance_for_order(order)
                return
            
            # Determine fill price
            fill_price = await self._calculate_fill_price(order, ticker, orderbook)
            
            if not fill_price:
                order.update_state(OrderState.REJECTED, "Could not determine fill price")
                await self._unlock_balance_for_order(order)
                return
            
            # Determine fill quantity
            fill_quantity = await self._calculate_fill_quantity(order, orderbook)
            
            # Apply slippage
            if self.enable_slippage:
                fill_price = self._apply_slippage(order, fill_price, ticker)
            
            # Calculate fee
            is_maker = order.order_type == OrderType.LIMIT
            fee_rate = self.maker_fee if is_maker else self.taker_fee
            fee = fill_quantity * fill_price * fee_rate
            
            # Create fill
            self._trade_id_counter += 1
            fill = OrderFill(
                fill_id=f"paper_fill_{self._trade_id_counter}",
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=fill_quantity,
                price=fill_price,
                fee=fee,
                fee_currency=self._get_quote_asset(order.symbol),
                timestamp=datetime.utcnow(),
                is_maker=is_maker
            )
            
            # Update order
            order.add_fill(fill)
            self._fills.append(fill)
            
            # Update balances
            await self._update_balances_for_fill(order, fill)
            
            # Notify callbacks
            await self._notify_fill(fill)
            
            # Record slippage
            if order.expected_price:
                await self.slippage_model.record_slippage(
                    order, fill, order.expected_price,
                    None  # Market conditions already tracked
                )
            
            # Handle partial fills
            if order.remaining_quantity > 0 and self.enable_partial_fills:
                # Schedule another fill
                if random.random() < self.partial_fill_probability:
                    await asyncio.sleep(random.uniform(0.5, 2.0))
                    asyncio.create_task(self._simulate_fill(order))
            
            logger.info(
                f"Paper fill: {order.symbol} {fill.side.value} "
                f"{fill.quantity} @ {fill.price}"
            )
            
        except Exception as e:
            logger.error(f"Error simulating fill: {e}")
            order.update_state(OrderState.ERROR, str(e))
    
    async def _calculate_fill_price(self, order: Order, 
                                     ticker: Optional[Ticker],
                                     orderbook: Optional[OrderBook]) -> Optional[Decimal]:
        """Calculate fill price based on order type and market data."""
        if order.order_type == OrderType.MARKET:
            # Market order fills at best available price
            if orderbook:
                if order.side == OrderSide.BUY:
                    best_ask = orderbook.get_best_ask()
                    return best_ask.price if best_ask else None
                else:
                    best_bid = orderbook.get_best_bid()
                    return best_bid.price if best_bid else None
            elif ticker:
                return ticker.ask if order.side == OrderSide.BUY else ticker.bid
                
        elif order.order_type == OrderType.LIMIT:
            # Limit order fills at limit price if marketable
            if not order.price:
                return None
            
            if orderbook:
                if order.side == OrderSide.BUY:
                    # Buy limit fills if price >= best ask
                    best_ask = orderbook.get_best_ask()
                    if best_ask and order.price >= best_ask.price:
                        return min(order.price, best_ask.price)
                else:
                    # Sell limit fills if price <= best bid
                    best_bid = orderbook.get_best_bid()
                    if best_bid and order.price <= best_bid.price:
                        return max(order.price, best_bid.price)
            
            # Limit order not marketable yet
            return None
            
        elif order.order_type == OrderType.STOP:
            # Stop order becomes market order when triggered
            if not order.stop_price:
                return None
            
            if ticker:
                if order.side == OrderSide.BUY:
                    # Buy stop triggers when price >= stop price
                    if ticker.last >= order.stop_price:
                        return ticker.ask
                else:
                    # Sell stop triggers when price <= stop price
                    if ticker.last <= order.stop_price:
                        return ticker.bid
        
        return None
    
    async def _calculate_fill_quantity(self, order: Order, 
                                        orderbook: Optional[OrderBook]) -> Decimal:
        """Calculate fill quantity."""
        if not self.enable_partial_fills:
            return order.remaining_quantity
        
        # For market orders, check order book depth
        if order.order_type == OrderType.MARKET and orderbook:
            if order.side == OrderSide.BUY:
                available = sum(a.quantity for a in orderbook.asks[:3])
            else:
                available = sum(b.quantity for b in orderbook.bids[:3])
            
            return min(order.remaining_quantity, available)
        
        # Random partial fill for testing
        if random.random() < self.partial_fill_probability:
            return min(
                order.remaining_quantity,
                order.remaining_quantity * Decimal(str(random.uniform(0.3, 0.7)))
            )
        
        return order.remaining_quantity
    
    def _apply_slippage(self, order: Order, price: Decimal, 
                        ticker: Optional[Ticker]) -> Decimal:
        """Apply realistic slippage to fill price."""
        if not ticker:
            return price
        
        # Get slippage estimate
        slippage_estimate = asyncio.run(
            self.slippage_model.estimate_slippage(
                order.symbol, order.side, order.quantity, order.order_type
            )
        )
        
        if slippage_estimate:
            slippage_bps = slippage_estimate.expected_slippage_bps
        else:
            # Default slippage based on spread
            spread_bps = (ticker.spread / ticker.mid_price) * Decimal("10000")
            slippage_bps = spread_bps * Decimal("0.5")
        
        # Apply slippage (adverse)
        slippage_factor = slippage_bps / Decimal("10000")
        
        if order.side == OrderSide.BUY:
            return price * (Decimal("1") + slippage_factor)
        else:
            return price * (Decimal("1") - slippage_factor)
    
    async def _update_balances_for_fill(self, order: Order, fill: OrderFill):
        """Update balances after a fill."""
        base_asset = self._get_base_asset(order.symbol)
        quote_asset = self._get_quote_asset(order.symbol)
        
        base_balance = self._balances.get(base_asset, Balance(base_asset, Decimal("0"), Decimal("0"), Decimal("0")))
        quote_balance = self._balances.get(quote_asset, Balance(quote_asset, Decimal("0"), Decimal("0"), Decimal("0")))
        
        if order.side == OrderSide.BUY:
            # Buy: spend quote, receive base
            cost = fill.quantity * fill.price + fill.fee
            quote_balance.locked -= cost
            quote_balance.total -= cost
            base_balance.free += fill.quantity
            base_balance.total += fill.quantity
        else:
            # Sell: spend base, receive quote
            proceeds = fill.quantity * fill.price - fill.fee
            base_balance.locked -= fill.quantity
            base_balance.total -= fill.quantity
            quote_balance.free += proceeds
            quote_balance.total += proceeds
    
    def _get_base_asset(self, symbol: str) -> str:
        """Extract base asset from symbol."""
        # Handle formats like BTCUSDT, BTC-USDT, BTC/USDT
        symbol = symbol.replace("-", "").replace("/", "")
        if symbol.endswith("USDT"):
            return symbol[:-4]
        elif symbol.endswith("USD"):
            return symbol[:-3]
        elif symbol.endswith("BTC"):
            return symbol[:-3]
        return symbol[:3]  # Default assumption
    
    def _get_quote_asset(self, symbol: str) -> str:
        """Extract quote asset from symbol."""
        symbol = symbol.replace("-", "").replace("/", "")
        if symbol.endswith("USDT"):
            return "USDT"
        elif symbol.endswith("USD"):
            return "USD"
        elif symbol.endswith("BTC"):
            return "BTC"
        return symbol[3:]  # Default assumption
    
    async def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> bool:
        """Cancel a simulated order."""
        timer_id = self.latency_tracker.start_timer(
            f"paper_cancel_{order_id}", LatencyType.CANCEL_REQUEST, "paper"
        )
        
        try:
            order = self._orders.get(order_id)
            if not order:
                return False
            
            if not order.is_active:
                return False
            
            order.update_state(OrderState.CANCELLED, "User cancelled")
            await self._unlock_balance_for_order(order)
            
            self._order_history.append(order)
            
            self.latency_tracker.stop_timer(timer_id, True)
            logger.info(f"Paper order cancelled: {order_id}")
            return True
            
        except Exception as e:
            self.latency_tracker.stop_timer(timer_id, False, {"error": str(e)})
            raise
    
    async def get_order(self, order_id: str, symbol: Optional[str] = None) -> Optional[Order]:
        """Get order details."""
        return self._orders.get(order_id)
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all open orders."""
        orders = [o for o in self._orders.values() if o.is_active]
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders
    
    async def get_order_history(self, symbol: Optional[str] = None,
                                 limit: int = 100) -> List[Order]:
        """Get order history."""
        history = list(self._order_history)
        if symbol:
            history = [o for o in history if o.symbol == symbol]
        return history[-limit:]
    
    # ==================== Account Operations ====================
    
    async def get_balance(self, asset: Optional[str] = None) -> Dict[str, Balance]:
        """Get account balance."""
        if asset:
            balance = self._balances.get(asset)
            return {asset: balance} if balance else {}
        return dict(self._balances)
    
    async def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """Get open positions."""
        positions = list(self._positions.values())
        if symbol:
            positions = [p for p in positions if p.symbol == symbol]
        return positions
    
    # ==================== Market Data ====================
    
    async def get_ticker(self, symbol: str) -> Ticker:
        """Get current price ticker."""
        if self.reference_exchange:
            return await self.reference_exchange.get_ticker(symbol)
        
        # Return cached or default
        ticker = self._price_cache.get(symbol)
        if ticker:
            return ticker
        
        raise ExchangeError(f"No price data available for {symbol}")
    
    async def get_orderbook(self, symbol: str, depth: int = 20) -> OrderBook:
        """Get order book snapshot."""
        if self.reference_exchange:
            return await self.reference_exchange.get_orderbook(symbol, depth)
        
        orderbook = self._orderbook_cache.get(symbol)
        if orderbook:
            return orderbook
        
        raise ExchangeError(f"No order book data available for {symbol}")
    
    async def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Get recent trades."""
        if self.reference_exchange:
            return await self.reference_exchange.get_recent_trades(symbol, limit)
        return []
    
    # ==================== WebSocket ====================
    
    async def subscribe_ticker(self, symbols: List[str]):
        """Subscribe to ticker updates."""
        if self.reference_exchange:
            await self.reference_exchange.subscribe_ticker(symbols)
    
    async def subscribe_orderbook(self, symbols: List[str], depth: int = 20):
        """Subscribe to order book updates."""
        if self.reference_exchange:
            await self.reference_exchange.subscribe_orderbook(symbols, depth)
    
    async def subscribe_trades(self, symbols: List[str]):
        """Subscribe to trade updates."""
        if self.reference_exchange:
            await self.reference_exchange.subscribe_trades(symbols)
    
    async def subscribe_user_data(self):
        """Subscribe to user data."""
        pass
    
    # ==================== Paper Trading Specific ====================
    
    def set_price(self, symbol: str, bid: Decimal, ask: Decimal):
        """Manually set price for a symbol (for testing)."""
        self._price_cache[symbol] = Ticker(
            symbol=symbol,
            bid=bid,
            ask=ask,
            last=(bid + ask) / 2,
            volume_24h=Decimal("0"),
            timestamp=datetime.utcnow()
        )
    
    def get_performance_summary(self) -> Dict:
        """Get trading performance summary."""
        total_filled_orders = sum(1 for o in self._order_history if o.state == OrderState.FILLED)
        total_cancelled = sum(1 for o in self._order_history if o.state == OrderState.CANCELLED)
        
        total_fees = sum(f.fee for f in self._fills)
        
        # Calculate P&L (simplified)
        pnl = Decimal("0")
        for fill in self._fills:
            if fill.side == OrderSide.BUY:
                pnl -= fill.quantity * fill.price + fill.fee
            else:
                pnl += fill.quantity * fill.price - fill.fee
        
        return {
            "total_orders": len(self._orders) + len(self._order_history),
            "filled_orders": total_filled_orders,
            "cancelled_orders": total_cancelled,
            "total_fills": len(self._fills),
            "total_fees": str(total_fees),
            "estimated_pnl": str(pnl),
            "current_balances": {
                asset: {
                    "free": str(bal.free),
                    "locked": str(bal.locked),
                    "total": str(bal.total)
                }
                for asset, bal in self._balances.items()
            }
        }
    
    def reset(self):
        """Reset paper trading state."""
        self._orders.clear()
        self._order_history.clear()
        self._fills.clear()
        self._positions.clear()
        self._initialize_balances()
        self._trade_id_counter = 0
        logger.info("Paper trading state reset")
