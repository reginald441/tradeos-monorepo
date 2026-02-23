"""
TradeOS WebSocket Manager
Central WebSocket manager for all market data feeds with auto-reconnection.

Features:
- Async WebSocket connection management
- Automatic reconnection with exponential backoff
- Connection pooling and load balancing
- Heartbeat/ping monitoring
- Message routing and distribution
- Comprehensive error handling and logging
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar, Generic
from collections import defaultdict
import uuid

import websockets
from websockets.exceptions import ConnectionClosed, InvalidStatusCode

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """WebSocket connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"
    CLOSING = "closing"


class FeedType(Enum):
    """Types of market data feeds."""
    TRADE = "trade"
    TICKER = "ticker"
    ORDERBOOK = "orderbook"
    KLINE = "kline"
    LIQUIDATION = "liquidation"
    FUNDING_RATE = "funding_rate"


@dataclass
class WebSocketConfig:
    """Configuration for WebSocket connections."""
    url: str
    name: str
    reconnect_attempts: int = 10
    reconnect_delay_base: float = 1.0
    reconnect_delay_max: float = 60.0
    reconnect_delay_multiplier: float = 2.0
    heartbeat_interval: float = 30.0
    heartbeat_timeout: float = 10.0
    connection_timeout: float = 10.0
    message_timeout: float = 60.0
    max_message_size: int = 10 * 1024 * 1024  # 10MB
    compression: bool = True
    ssl_verify: bool = True
    headers: Dict[str, str] = field(default_factory=dict)
    subscriptions: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ConnectionMetrics:
    """Metrics for WebSocket connections."""
    connection_id: str
    connected_at: Optional[float] = None
    disconnected_at: Optional[float] = None
    messages_received: int = 0
    messages_sent: int = 0
    bytes_received: int = 0
    bytes_sent: int = 0
    reconnections: int = 0
    errors: int = 0
    last_message_at: Optional[float] = None
    latency_ms: Optional[float] = None


T = TypeVar('T')


class MessageHandler(ABC, Generic[T]):
    """Abstract base class for message handlers."""
    
    @abstractmethod
    async def handle(self, message: T, source: str, timestamp: float) -> None:
        """Handle an incoming message."""
        pass
    
    @abstractmethod
    def can_handle(self, message: T) -> bool:
        """Check if this handler can process the message."""
        pass


class WebSocketConnection:
    """Manages a single WebSocket connection with auto-reconnection."""
    
    def __init__(
        self,
        config: WebSocketConfig,
        message_callback: Callable[[str, str, float], None],
        error_callback: Optional[Callable[[Exception, str], None]] = None
    ):
        self.config = config
        self.message_callback = message_callback
        self.error_callback = error_callback
        self.connection_id = str(uuid.uuid4())[:8]
        
        self._websocket: Optional[websockets.WebSocketClientProtocol] = None
        self._state = ConnectionState.DISCONNECTED
        self._metrics = ConnectionMetrics(connection_id=self.connection_id)
        self._reconnect_count = 0
        self._should_run = False
        self._tasks: Set[asyncio.Task] = set()
        self._lock = asyncio.Lock()
        self._last_pong = time.time()
        self._subscribed_channels: Set[str] = set()
        
    @property
    def state(self) -> ConnectionState:
        """Get current connection state."""
        return self._state
    
    @property
    def metrics(self) -> ConnectionMetrics:
        """Get connection metrics."""
        return self._metrics
    
    @property
    def is_connected(self) -> bool:
        """Check if connection is established."""
        return self._state == ConnectionState.CONNECTED and self._websocket is not None
    
    async def start(self) -> None:
        """Start the WebSocket connection."""
        self._should_run = True
        await self._connect_with_retry()
    
    async def stop(self) -> None:
        """Stop the WebSocket connection."""
        self._should_run = False
        self._state = ConnectionState.CLOSING
        
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
        
        if self._websocket:
            try:
                await self._websocket.close()
            except Exception as e:
                logger.debug(f"Error closing websocket: {e}")
        
        self._state = ConnectionState.DISCONNECTED
        logger.info(f"[{self.config.name}] Connection stopped")
    
    async def send(self, message: str) -> bool:
        """Send a message through the WebSocket."""
        if not self.is_connected or not self._websocket:
            logger.warning(f"[{self.config.name}] Cannot send, not connected")
            return False
        
        try:
            await self._websocket.send(message)
            self._metrics.messages_sent += 1
            self._metrics.bytes_sent += len(message.encode())
            return True
        except Exception as e:
            logger.error(f"[{self.config.name}] Send error: {e}")
            return False
    
    async def subscribe(self, subscription: Dict[str, Any]) -> bool:
        """Subscribe to a channel/topic."""
        channel_id = json.dumps(subscription, sort_keys=True)
        self._subscribed_channels.add(channel_id)
        
        if self.is_connected:
            return await self.send(json.dumps(subscription))
        return False
    
    async def unsubscribe(self, subscription: Dict[str, Any]) -> bool:
        """Unsubscribe from a channel/topic."""
        channel_id = json.dumps(subscription, sort_keys=True)
        self._subscribed_channels.discard(channel_id)
        
        # Create unsubscribe message (exchange-specific)
        unsub = subscription.copy()
        unsub['action'] = 'unsubscribe'
        
        if self.is_connected:
            return await self.send(json.dumps(unsub))
        return False
    
    async def _connect_with_retry(self) -> None:
        """Connect with exponential backoff retry."""
        while self._should_run:
            try:
                await self._connect()
                self._reconnect_count = 0  # Reset on successful connection
                await self._run_connection_loop()
            except Exception as e:
                logger.error(f"[{self.config.name}] Connection error: {e}")
                self._metrics.errors += 1
            
            if not self._should_run:
                break
            
            # Calculate retry delay with exponential backoff
            delay = min(
                self.config.reconnect_delay_base * (self.config.reconnect_delay_multiplier ** self._reconnect_count),
                self.config.reconnect_delay_max
            )
            self._reconnect_count += 1
            
            if self._reconnect_count > self.config.reconnect_attempts:
                logger.error(f"[{self.config.name}] Max reconnection attempts reached")
                break
            
            logger.info(f"[{self.config.name}] Reconnecting in {delay:.1f}s (attempt {self._reconnect_count})")
            self._state = ConnectionState.RECONNECTING
            self._metrics.reconnections += 1
            await asyncio.sleep(delay)
    
    async def _connect(self) -> None:
        """Establish WebSocket connection."""
        async with self._lock:
            self._state = ConnectionState.CONNECTING
            logger.info(f"[{self.config.name}] Connecting to {self.config.url}")
            
            try:
                self._websocket = await websockets.connect(
                    self.config.url,
                    extra_headers=self.config.headers,
                    ping_interval=self.config.heartbeat_interval,
                    ping_timeout=self.config.heartbeat_timeout,
                    close_timeout=self.config.connection_timeout,
                    max_size=self.config.max_message_size,
                    compression=self.config.compression,
                    ssl=self.config.ssl_verify
                )
                
                self._state = ConnectionState.CONNECTED
                self._metrics.connected_at = time.time()
                logger.info(f"[{self.config.name}] Connected successfully")
                
                # Resubscribe to previous channels
                for channel in self._subscribed_channels:
                    try:
                        sub = json.loads(channel)
                        await self.send(json.dumps(sub))
                    except Exception as e:
                        logger.warning(f"[{self.config.name}] Failed to resubscribe: {e}")
                
            except InvalidStatusCode as e:
                logger.error(f"[{self.config.name}] HTTP {e.status_code} error")
                raise
            except Exception as e:
                logger.error(f"[{self.config.name}] Connection failed: {e}")
                raise
    
    async def _run_connection_loop(self) -> None:
        """Main connection loop for receiving messages."""
        if not self._websocket:
            return
        
        # Start heartbeat task
        heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._tasks.add(heartbeat_task)
        heartbeat_task.add_done_callback(self._tasks.discard)
        
        try:
            async for message in self._websocket:
                if not self._should_run:
                    break
                
                receive_time = time.time()
                self._metrics.last_message_at = receive_time
                self._metrics.messages_received += 1
                self._metrics.bytes_received += len(message.encode())
                
                # Call the message callback
                try:
                    await self.message_callback(message, self.config.name, receive_time)
                except Exception as e:
                    logger.error(f"[{self.config.name}] Message handler error: {e}")
        
        except ConnectionClosed as e:
            logger.warning(f"[{self.config.name}] Connection closed: {e}")
        except asyncio.CancelledError:
            logger.info(f"[{self.config.name}] Connection loop cancelled")
        except Exception as e:
            logger.error(f"[{self.config.name}] Connection loop error: {e}")
            if self.error_callback:
                try:
                    self.error_callback(e, self.config.name)
                except Exception:
                    pass
        finally:
            self._metrics.disconnected_at = time.time()
            self._state = ConnectionState.DISCONNECTED
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
    
    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats to keep connection alive."""
        while self._should_run and self.is_connected:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)
                if self._websocket and self.is_connected:
                    # Send ping
                    pong_waiter = await self._websocket.ping()
                    start_time = time.time()
                    await asyncio.wait_for(pong_waiter, timeout=self.config.heartbeat_timeout)
                    self._metrics.latency_ms = (time.time() - start_time) * 1000
            except asyncio.TimeoutError:
                logger.warning(f"[{self.config.name}] Heartbeat timeout")
                break
            except Exception as e:
                logger.debug(f"[{self.config.name}] Heartbeat error: {e}")
                break


class WebSocketManager:
    """Central manager for all WebSocket connections."""
    
    def __init__(self):
        self._connections: Dict[str, WebSocketConnection] = {}
        self._handlers: List[MessageHandler] = []
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._running = False
        self._dispatch_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        
    async def start(self) -> None:
        """Start the WebSocket manager."""
        self._running = True
        self._dispatch_task = asyncio.create_task(self._dispatch_loop())
        logger.info("WebSocket Manager started")
    
    async def stop(self) -> None:
        """Stop the WebSocket manager and all connections."""
        self._running = False
        
        # Stop all connections
        stop_tasks = [conn.stop() for conn in self._connections.values()]
        if stop_tasks:
            await asyncio.gather(*stop_tasks, return_exceptions=True)
        
        if self._dispatch_task:
            self._dispatch_task.cancel()
            try:
                await self._dispatch_task
            except asyncio.CancelledError:
                pass
        
        logger.info("WebSocket Manager stopped")
    
    async def add_connection(
        self,
        config: WebSocketConfig,
        auto_start: bool = True
    ) -> WebSocketConnection:
        """Add a new WebSocket connection."""
        async with self._lock:
            if config.name in self._connections:
                logger.warning(f"Connection '{config.name}' already exists, replacing")
                await self._connections[config.name].stop()
            
            connection = WebSocketConnection(
                config=config,
                message_callback=self._on_message,
                error_callback=self._on_error
            )
            self._connections[config.name] = connection
            
            if auto_start:
                asyncio.create_task(connection.start())
            
            logger.info(f"Added connection: {config.name}")
            return connection
    
    async def remove_connection(self, name: str) -> bool:
        """Remove a WebSocket connection."""
        async with self._lock:
            if name not in self._connections:
                return False
            
            connection = self._connections.pop(name)
            await connection.stop()
            logger.info(f"Removed connection: {name}")
            return True
    
    def get_connection(self, name: str) -> Optional[WebSocketConnection]:
        """Get a connection by name."""
        return self._connections.get(name)
    
    def get_all_connections(self) -> Dict[str, WebSocketConnection]:
        """Get all connections."""
        return self._connections.copy()
    
    def get_connection_states(self) -> Dict[str, ConnectionState]:
        """Get states of all connections."""
        return {name: conn.state for name, conn in self._connections.items()}
    
    def get_all_metrics(self) -> Dict[str, ConnectionMetrics]:
        """Get metrics for all connections."""
        return {name: conn.metrics for name, conn in self._connections.items()}
    
    def add_handler(self, handler: MessageHandler) -> None:
        """Add a message handler."""
        self._handlers.append(handler)
        logger.info(f"Added message handler: {handler.__class__.__name__}")
    
    def remove_handler(self, handler: MessageHandler) -> bool:
        """Remove a message handler."""
        if handler in self._handlers:
            self._handlers.remove(handler)
            return True
        return False
    
    def subscribe(self, event_type: str, callback: Callable) -> None:
        """Subscribe to a specific event type."""
        self._subscribers[event_type].append(callback)
    
    def unsubscribe(self, event_type: str, callback: Callable) -> bool:
        """Unsubscribe from an event type."""
        if event_type in self._subscribers and callback in self._subscribers[event_type]:
            self._subscribers[event_type].remove(callback)
            return True
        return False
    
    async def broadcast(self, message: str) -> Dict[str, bool]:
        """Broadcast a message to all connections."""
        results = {}
        for name, connection in self._connections.items():
            results[name] = await connection.send(message)
        return results
    
    async def _on_message(self, message: str, source: str, timestamp: float) -> None:
        """Handle incoming message from any connection."""
        await self._message_queue.put((message, source, timestamp))
    
    def _on_error(self, error: Exception, source: str) -> None:
        """Handle connection error."""
        logger.error(f"Connection error from {source}: {error}")
    
    async def _dispatch_loop(self) -> None:
        """Dispatch messages to handlers and subscribers."""
        while self._running:
            try:
                message, source, timestamp = await self._message_queue.get()
                
                # Parse message to determine type
                try:
                    data = json.loads(message)
                    event_type = data.get('e') or data.get('type') or 'unknown'
                except json.JSONDecodeError:
                    data = message
                    event_type = 'raw'
                
                # Call registered handlers
                handler_tasks = []
                for handler in self._handlers:
                    try:
                        if handler.can_handle(data):
                            task = asyncio.create_task(
                                handler.handle(data, source, timestamp)
                            )
                            handler_tasks.append(task)
                    except Exception as e:
                        logger.error(f"Handler error: {e}")
                
                if handler_tasks:
                    await asyncio.gather(*handler_tasks, return_exceptions=True)
                
                # Notify subscribers
                for callback in self._subscribers.get(event_type, []):
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            asyncio.create_task(callback(data, source, timestamp))
                        else:
                            callback(data, source, timestamp)
                    except Exception as e:
                        logger.error(f"Subscriber callback error: {e}")
                
                # Notify wildcards
                for callback in self._subscribers.get('*', []):
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            asyncio.create_task(callback(data, source, timestamp))
                        else:
                            callback(data, source, timestamp)
                    except Exception as e:
                        logger.error(f"Wildcard subscriber error: {e}")
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Dispatch loop error: {e}")


# Singleton instance
_manager_instance: Optional[WebSocketManager] = None


def get_websocket_manager() -> WebSocketManager:
    """Get the singleton WebSocket manager instance."""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = WebSocketManager()
    return _manager_instance


async def create_default_manager() -> WebSocketManager:
    """Create and start a default WebSocket manager."""
    manager = get_websocket_manager()
    await manager.start()
    return manager


# Example usage and testing
if __name__ == "__main__":
    async def test_websocket_manager():
        """Test the WebSocket manager."""
        manager = await create_default_manager()
        
        # Example: Add Binance connection
        binance_config = WebSocketConfig(
            url="wss://stream.binance.com:9443/ws/btcusdt@trade",
            name="binance_btcusdt",
            reconnect_attempts=5
        )
        
        await manager.add_connection(binance_config)
        
        # Add a simple message handler
        class PrintHandler(MessageHandler):
            def can_handle(self, message: Any) -> bool:
                return True
            
            async def handle(self, message: Any, source: str, timestamp: float) -> None:
                print(f"[{source}] {message}")
        
        manager.add_handler(PrintHandler())
        
        # Run for 30 seconds
        await asyncio.sleep(30)
        
        await manager.stop()
    
    asyncio.run(test_websocket_manager())
