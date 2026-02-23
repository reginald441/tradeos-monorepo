"""
TradeOS MetaTrader 5 Bridge
===========================
ZeroMQ-based bridge for MT5 integration.
Enables automated trading through MT5 platform.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime
from enum import Enum

import zmq
import zmq.asyncio

from ..base_exchange import (
    BaseExchange, Balance, Ticker, OrderBook, OrderBookLevel,
    Position, ExchangeError, NetworkError, InvalidOrderError,
    OrderNotFoundError
)
from ..models.order import (
    Order, OrderRequest, OrderFill, OrderState, OrderSide,
    OrderType, TimeInForce, OrderEvent, OrderEventType
)
from ..latency_tracker import LatencyType

logger = logging.getLogger(__name__)


class MT5Command(Enum):
    """MT5 command types."""
    GET_RATES = "RATES"
    GET_ACCOUNT = "ACCOUNT"
    GET_POSITIONS = "POSITIONS"
    GET_ORDERS = "ORDERS"
    PLACE_ORDER = "TRADE"
    CLOSE_POSITION = "CLOSE"
    CANCEL_ORDER = "CANCEL"
    GET_HISTORY = "HISTORY"
    GET_SYMBOL_INFO = "SYMBOL"


@dataclass
class MT5Config:
    """MT5 ZeroMQ configuration."""
    host: str = "localhost"
    request_port: int = 5555  # PUSH/PULL for commands
    publish_port: int = 5556  # PUB/SUB for data feed
    timeout_ms: int = 30000
    reconnect_delay: float = 5.0


class MT5Bridge(BaseExchange):
    """
    MetaTrader 5 Bridge via ZeroMQ.
    
    Features:
    - ZeroMQ communication with MT5 EA
    - Real-time price feed subscription
    - Order placement and management
    - Position tracking
    - Account information
    
    Requires:
    - MT5 running with ZeroMQ EA (e.g., DWZ MQL5 ZMQ EA)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("mt5", config)
        
        self.mt5_config = MT5Config(
            host=config.get("host", "localhost"),
            request_port=config.get("request_port", 5555),
            publish_port=config.get("publish_port", 5556),
            timeout_ms=config.get("timeout_ms", 30000),
            reconnect_delay=config.get("reconnect_delay", 5.0)
        )
        
        self._context: Optional[zmq.asyncio.Context] = None
        self._request_socket: Optional[zmq.asyncio.Socket] = None
        self._subscribe_socket: Optional[zmq.asyncio.Socket] = None
        self._poll_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        
        # Data cache
        self._price_cache: Dict[str, Dict] = {}
        self._position_cache: Dict[int, Dict] = {}
        self._order_cache: Dict[int, Dict] = {}
        
        # Callbacks
        self._price_callbacks: List[Callable[[str, Dict], None]] = []
        
        logger.info(f"MT5 Bridge initialized: {self.mt5_config.host}:{self.mt5_config.request_port}")
    
    # ==================== Connection ====================
    
    async def connect(self) -> bool:
        """Establish ZeroMQ connection to MT5."""
        try:
            self._context = zmq.asyncio.Context()
            
            # Create request socket (PUSH/PULL pattern)
            self._request_socket = self._context.socket(zmq.REQ)
            self._request_socket.setsockopt(zmq.RCVTIMEO, self.mt5_config.timeout_ms)
            self._request_socket.setsockopt(zmq.LINGER, 0)
            self._request_socket.connect(
                f"tcp://{self.mt5_config.host}:{self.mt5_config.request_port}"
            )
            
            # Create subscription socket (PUB/SUB pattern)
            self._subscribe_socket = self._context.socket(zmq.SUB)
            self._subscribe_socket.setsockopt(zmq.RCVTIMEO, 1000)
            self._subscribe_socket.connect(
                f"tcp://{self.mt5_config.host}:{self.mt5_config.publish_port}"
            )
            self._subscribe_socket.setsockopt_string(zmq.SUBSCRIBE, "")
            
            # Test connection
            account_info = await self._send_command(MT5Command.GET_ACCOUNT)
            if account_info:
                logger.info(f"Connected to MT5. Account: {account_info.get('login')}")
                self.is_connected = True
                
                # Start background tasks
                self._poll_task = asyncio.create_task(self._poll_subscriptions())
                self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to connect to MT5: {e}")
            self.is_connected = False
            return False
    
    async def disconnect(self) -> bool:
        """Close ZeroMQ connection."""
        try:
            # Cancel background tasks
            if self._poll_task:
                self._poll_task.cancel()
                try:
                    await self._poll_task
                except asyncio.CancelledError:
                    pass
            
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
                try:
                    await self._heartbeat_task
                except asyncio.CancelledError:
                    pass
            
            # Close sockets
            if self._request_socket:
                self._request_socket.close()
            if self._subscribe_socket:
                self._subscribe_socket.close()
            if self._context:
                self._context.term()
            
            self.is_connected = False
            logger.info("Disconnected from MT5")
            return True
            
        except Exception as e:
            logger.error(f"Error disconnecting from MT5: {e}")
            return False
    
    async def is_connection_healthy(self) -> bool:
        """Check if connection is healthy."""
        try:
            response = await self._send_command(MT5Command.GET_ACCOUNT)
            return response is not None
        except Exception:
            return False
    
    async def _send_command(self, command: MT5Command, 
                            params: Optional[Dict] = None) -> Optional[Dict]:
        """Send command to MT5 and wait for response."""
        if not self._request_socket:
            raise NetworkError("Not connected to MT5")
        
        request = {
            "action": command.value,
            "timestamp": int(time.time() * 1000)
        }
        if params:
            request.update(params)
        
        timer_id = self.latency_tracker.start_timer(
            f"mt5_{command.value}", LatencyType.API_REQUEST, "mt5"
        )
        
        try:
            # Send request
            await self._request_socket.send_json(request)
            
            # Wait for response
            response = await self._request_socket.recv_json()
            
            success = response.get("error") is None
            self.latency_tracker.stop_timer(timer_id, success)
            
            if response.get("error"):
                raise ExchangeError(f"MT5 error: {response['error']}")
            
            return response
            
        except zmq.Again:
            self.latency_tracker.stop_timer(timer_id, False, {"error": "timeout"})
            raise NetworkError("MT5 request timeout")
        except Exception as e:
            self.latency_tracker.stop_timer(timer_id, False, {"error": str(e)})
            raise
    
    async def _poll_subscriptions(self):
        """Poll subscription socket for price updates."""
        while True:
            try:
                if self._subscribe_socket:
                    message = await self._subscribe_socket.recv_json()
                    await self._handle_subscription_message(message)
            except zmq.Again:
                # Timeout, continue polling
                await asyncio.sleep(0.001)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error polling subscriptions: {e}")
                await asyncio.sleep(1)
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats to check connection."""
        while True:
            try:
                await asyncio.sleep(30)
                if not await self.is_connection_healthy():
                    logger.warning("MT5 heartbeat failed")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
    
    async def _handle_subscription_message(self, message: Dict):
        """Handle incoming subscription message."""
        msg_type = message.get("type", "")
        
        if msg_type == "PRICE":
            symbol = message.get("symbol", "")
            self._price_cache[symbol] = message
            
            # Notify callbacks
            for callback in self._price_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        asyncio.create_task(callback(symbol, message))
                    else:
                        callback(symbol, message)
                except Exception as e:
                    logger.error(f"Error in price callback: {e}")
        
        elif msg_type == "POSITION":
            ticket = message.get("ticket")
            if ticket:
                self._position_cache[ticket] = message
        
        elif msg_type == "ORDER":
            ticket = message.get("ticket")
            if ticket:
                self._order_cache[ticket] = message
    
    # ==================== Order Operations ====================
    
    async def place_order(self, order_request: OrderRequest) -> Order:
        """Place a new order via MT5."""
        timer_id = self.latency_tracker.start_timer(
            f"order_{order_request.symbol}", LatencyType.ORDER_SUBMIT, "mt5"
        )
        
        try:
            # Map order type to MT5 order type
            mt5_order_type = self._map_to_mt5_order_type(order_request.order_type)
            
            params = {
                "symbol": order_request.symbol,
                "type": mt5_order_type,
                "side": 0 if order_request.side == OrderSide.BUY else 1,
                "volume": float(order_request.quantity),
                "price": float(order_request.price) if order_request.price else 0.0,
                "sl": 0.0,  # Stop loss
                "tp": 0.0,  # Take profit
                "deviation": 10,  # Slippage in points
                "magic": 0,  # Magic number
                "comment": order_request.client_order_id or "TradeOS",
            }
            
            response = await self._send_command(MT5Command.PLACE_ORDER, params)
            
            self.latency_tracker.stop_timer(timer_id, True)
            
            if response and response.get("ticket"):
                order = Order(
                    order_id=str(response.get("ticket")),
                    client_order_id=order_request.client_order_id,
                    exchange_order_id=str(response.get("ticket")),
                    symbol=order_request.symbol,
                    side=order_request.side,
                    order_type=order_request.order_type,
                    quantity=order_request.quantity,
                    price=order_request.price,
                    state=OrderState.OPEN,
                    exchange="mt5"
                )
                logger.info(f"Order placed on MT5: {order.order_id}")
                return order
            else:
                raise InvalidOrderError("MT5 did not return order ticket")
                
        except Exception as e:
            self.latency_tracker.stop_timer(timer_id, False, {"error": str(e)})
            raise
    
    async def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> bool:
        """Cancel a pending order."""
        timer_id = self.latency_tracker.start_timer(
            f"cancel_{order_id}", LatencyType.CANCEL_REQUEST, "mt5"
        )
        
        try:
            params = {"ticket": int(order_id)}
            
            response = await self._send_command(MT5Command.CANCEL_ORDER, params)
            
            self.latency_tracker.stop_timer(timer_id, True)
            
            if response and response.get("success"):
                logger.info(f"Order cancelled on MT5: {order_id}")
                return True
            
            return False
            
        except Exception as e:
            self.latency_tracker.stop_timer(timer_id, False, {"error": str(e)})
            raise
    
    async def get_order(self, order_id: str, symbol: Optional[str] = None) -> Optional[Order]:
        """Get order details from MT5."""
        try:
            # Check cache first
            ticket = int(order_id)
            if ticket in self._order_cache:
                return self._parse_order(self._order_cache[ticket])
            
            # Fetch from MT5
            response = await self._send_command(MT5Command.GET_ORDERS)
            
            for order_data in response.get("orders", []):
                if order_data.get("ticket") == ticket:
                    return self._parse_order(order_data)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting order: {e}")
            return None
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all open orders from MT5."""
        response = await self._send_command(MT5Command.GET_ORDERS)
        
        orders = []
        for order_data in response.get("orders", []):
            if symbol and order_data.get("symbol") != symbol:
                continue
            
            order = self._parse_order(order_data)
            if order.state in (OrderState.PENDING, OrderState.OPEN):
                orders.append(order)
        
        return orders
    
    async def get_order_history(self, symbol: Optional[str] = None,
                                 limit: int = 100) -> List[Order]:
        """Get order history from MT5."""
        response = await self._send_command(MT5Command.GET_HISTORY)
        
        orders = []
        for order_data in response.get("history", [])[:limit]:
            if symbol and order_data.get("symbol") != symbol:
                continue
            
            order = self._parse_order(order_data)
            orders.append(order)
        
        return orders
    
    def _parse_order(self, data: Dict) -> Order:
        """Parse MT5 order data to Order object."""
        # MT5 order state mapping
        mt5_state = data.get("state", 0)
        state_map = {
            0: OrderState.PENDING,   # ORDER_STATE_STARTED
            1: OrderState.PENDING,   # ORDER_STATE_PLACED
            2: OrderState.CANCELLED, # ORDER_STATE_CANCELED
            3: OrderState.PARTIALLY_FILLED,  # ORDER_STATE_PARTIAL
            4: OrderState.FILLED,    # ORDER_STATE_FILLED
            5: OrderState.REJECTED,  # ORDER_STATE_REJECTED
        }
        
        # MT5 order type mapping
        mt5_type = data.get("type", 0)
        type_map = {
            0: OrderType.MARKET,     # ORDER_TYPE_BUY
            1: OrderType.MARKET,     # ORDER_TYPE_SELL
            2: OrderType.LIMIT,      # ORDER_TYPE_BUY_LIMIT
            3: OrderType.LIMIT,      # ORDER_TYPE_SELL_LIMIT
            4: OrderType.STOP,       # ORDER_TYPE_BUY_STOP
            5: OrderType.STOP,       # ORDER_TYPE_SELL_STOP
        }
        
        order = Order(
            order_id=str(data.get("ticket", "")),
            client_order_id=data.get("comment"),
            exchange_order_id=str(data.get("ticket", "")),
            symbol=data.get("symbol", ""),
            side=OrderSide.BUY if data.get("side") == 0 else OrderSide.SELL,
            order_type=type_map.get(mt5_type, OrderType.MARKET),
            quantity=Decimal(str(data.get("volume", "0"))),
            price=Decimal(str(data.get("price", "0"))) if data.get("price") else None,
            filled_quantity=Decimal(str(data.get("volume_current", "0"))),
            state=state_map.get(mt5_state, OrderState.PENDING),
            exchange="mt5"
        )
        
        order.remaining_quantity = order.quantity - order.filled_quantity
        
        return order
    
    def _map_to_mt5_order_type(self, order_type: OrderType) -> int:
        """Map internal order type to MT5 order type."""
        mapping = {
            OrderType.MARKET: 0,   # ORDER_TYPE_BUY
            OrderType.LIMIT: 2,    # ORDER_TYPE_BUY_LIMIT
            OrderType.STOP: 4,     # ORDER_TYPE_BUY_STOP
        }
        return mapping.get(order_type, 0)
    
    # ==================== Account Operations ====================
    
    async def get_balance(self, asset: Optional[str] = None) -> Dict[str, Balance]:
        """Get account balance from MT5."""
        response = await self._send_command(MT5Command.GET_ACCOUNT)
        
        balances = {}
        
        if response:
            # MT5 account info
            currency = response.get("currency", "USD")
            balance = Decimal(str(response.get("balance", "0")))
            equity = Decimal(str(response.get("equity", "0")))
            margin = Decimal(str(response.get("margin", "0")))
            
            balances[currency] = Balance(
                asset=currency,
                free=equity - margin,
                locked=margin,
                total=balance
            )
        
        return balances
    
    async def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """Get open positions from MT5."""
        response = await self._send_command(MT5Command.GET_POSITIONS)
        
        positions = []
        for pos_data in response.get("positions", []):
            if symbol and pos_data.get("symbol") != symbol:
                continue
            
            volume = Decimal(str(pos_data.get("volume", "0")))
            if volume > 0:
                positions.append(Position(
                    symbol=pos_data.get("symbol", ""),
                    side="long" if pos_data.get("type") == 0 else "short",
                    quantity=volume,
                    entry_price=Decimal(str(pos_data.get("price_open", "0"))),
                    unrealized_pnl=Decimal(str(pos_data.get("profit", "0"))),
                    leverage=Decimal("1"),  # MT5 doesn't expose leverage directly
                ))
        
        return positions
    
    async def close_position(self, symbol: str, position_id: Optional[int] = None) -> bool:
        """Close a position."""
        params = {"symbol": symbol}
        if position_id:
            params["ticket"] = position_id
        
        response = await self._send_command(MT5Command.CLOSE_POSITION, params)
        return response and response.get("success", False)
    
    # ==================== Market Data ====================
    
    async def get_ticker(self, symbol: str) -> Ticker:
        """Get current price ticker from MT5."""
        # Check cache first
        if symbol in self._price_cache:
            price_data = self._price_cache[symbol]
            return Ticker(
                symbol=symbol,
                bid=Decimal(str(price_data.get("bid", "0"))),
                ask=Decimal(str(price_data.get("ask", "0"))),
                last=Decimal(str(price_data.get("last", "0"))),
                volume_24h=Decimal("0"),  # Not available from MT5 directly
                timestamp=datetime.utcnow()
            )
        
        # Fetch from MT5
        params = {"symbol": symbol}
        response = await self._send_command(MT5Command.GET_RATES, params)
        
        if response and "rates" in response:
            rate = response["rates"][0]
            return Ticker(
                symbol=symbol,
                bid=Decimal(str(rate.get("bid", "0"))),
                ask=Decimal(str(rate.get("ask", "0"))),
                last=Decimal(str(rate.get("close", "0"))),
                volume_24h=Decimal("0"),
                timestamp=datetime.utcnow()
            )
        
        raise ExchangeError(f"Could not get ticker for {symbol}")
    
    async def get_orderbook(self, symbol: str, depth: int = 20) -> OrderBook:
        """Get order book snapshot from MT5."""
        # MT5 doesn't provide order book depth via ZeroMQ
        # Return empty order book with current price
        ticker = await self.get_ticker(symbol)
        
        return OrderBook(
            symbol=symbol,
            bids=[OrderBookLevel(price=ticker.bid, quantity=Decimal("0"))],
            asks=[OrderBookLevel(price=ticker.ask, quantity=Decimal("0"))],
            timestamp=datetime.utcnow()
        )
    
    async def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Get recent trades from MT5."""
        params = {"symbol": symbol, "count": limit}
        response = await self._send_command(MT5Command.GET_HISTORY, params)
        
        trades = []
        for deal in response.get("deals", []):
            trades.append({
                "ticket": deal.get("ticket"),
                "symbol": deal.get("symbol"),
                "type": deal.get("type"),
                "volume": deal.get("volume"),
                "price": deal.get("price"),
                "profit": deal.get("profit"),
                "time": deal.get("time"),
            })
        
        return trades
    
    # ==================== Symbol Information ====================
    
    async def get_symbol_info(self, symbol: str) -> Dict:
        """Get symbol information from MT5."""
        params = {"symbol": symbol}
        return await self._send_command(MT5Command.GET_SYMBOL_INFO, params)
    
    # ==================== WebSocket (Not Applicable) ====================
    
    async def subscribe_ticker(self, symbols: List[str]):
        """Subscribe to ticker updates (handled via ZeroMQ)."""
        # Already subscribed to all via ZeroMQ
        pass
    
    async def subscribe_orderbook(self, symbols: List[str], depth: int = 20):
        """Subscribe to order book updates (not supported)."""
        logger.warning("MT5 does not support order book subscriptions via ZeroMQ")
    
    async def subscribe_trades(self, symbols: List[str]):
        """Subscribe to trade updates (handled via ZeroMQ)."""
        pass
    
    async def subscribe_user_data(self):
        """Subscribe to user data (handled via ZeroMQ)."""
        pass
    
    # ==================== Callback Registration ====================
    
    def register_price_callback(self, callback: Callable[[str, Dict], None]):
        """Register callback for price updates."""
        self._price_callbacks.append(callback)
    
    def unregister_price_callback(self, callback: Callable[[str, Dict], None]):
        """Unregister price callback."""
        if callback in self._price_callbacks:
            self._price_callbacks.remove(callback)
