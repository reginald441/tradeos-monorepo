"""
TradeOS Execution Latency Tracker
=================================
Comprehensive latency monitoring for order execution.
Tracks API latency, network latency, and end-to-end execution time.
"""

from __future__ import annotations

import asyncio
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Deque, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class LatencyType(Enum):
    """Types of latency measurements."""
    API_REQUEST = "api_request"           # HTTP API request/response
    WEBSOCKET_MESSAGE = "websocket"       # WebSocket message latency
    ORDER_SUBMIT = "order_submit"         # Order submission time
    ORDER_CONFIRM = "order_confirm"       # Order confirmation time
    FILL_NOTIFICATION = "fill_notify"     # Fill notification latency
    CANCEL_REQUEST = "cancel_request"     # Cancel request time
    HEARTBEAT = "heartbeat"               # Connection heartbeat
    RECONNECT = "reconnect"               # Reconnection time
    END_TO_END = "end_to_end"             # Total execution time


@dataclass
class LatencyMeasurement:
    """Single latency measurement."""
    latency_type: LatencyType
    timestamp: datetime
    latency_ms: float
    exchange: str
    operation: str
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class LatencyStats:
    """Statistics for latency measurements."""
    latency_type: LatencyType
    exchange: str
    count: int = 0
    min_ms: float = float('inf')
    max_ms: float = 0.0
    mean_ms: float = 0.0
    median_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    std_dev: float = 0.0
    last_updated: Optional[datetime] = None
    
    def update(self, measurements: List[float]):
        """Update statistics from measurements."""
        if not measurements:
            return
        
        self.count = len(measurements)
        self.min_ms = min(measurements)
        self.max_ms = max(measurements)
        self.mean_ms = statistics.mean(measurements)
        self.median_ms = statistics.median(measurements)
        self.std_dev = statistics.stdev(measurements) if len(measurements) > 1 else 0.0
        
        # Calculate percentiles
        sorted_measurements = sorted(measurements)
        self.p95_ms = self._percentile(sorted_measurements, 0.95)
        self.p99_ms = self._percentile(sorted_measurements, 0.99)
        self.last_updated = datetime.utcnow()
    
    @staticmethod
    def _percentile(sorted_data: List[float], percentile: float) -> float:
        """Calculate percentile from sorted data."""
        if not sorted_data:
            return 0.0
        index = int(len(sorted_data) * percentile)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "latency_type": self.latency_type.value,
            "exchange": self.exchange,
            "count": self.count,
            "min_ms": round(self.min_ms, 3),
            "max_ms": round(self.max_ms, 3),
            "mean_ms": round(self.mean_ms, 3),
            "median_ms": round(self.median_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "p99_ms": round(self.p99_ms, 3),
            "std_dev": round(self.std_dev, 3),
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
        }


class LatencyTracker:
    """
    Tracks and analyzes execution latency.
    
    Features:
    - Real-time latency measurement
    - Statistical analysis (mean, median, percentiles)
    - Alerting on high latency
    - Historical data retention
    """
    
    DEFAULT_WINDOW_SIZE = 1000  # Keep last 1000 measurements per type
    ALERT_THRESHOLD_MS = 1000   # Alert if latency exceeds 1 second
    
    def __init__(self, window_size: int = DEFAULT_WINDOW_SIZE):
        self.window_size = window_size
        self._measurements: Dict[Tuple[LatencyType, str], Deque[LatencyMeasurement]] = {}
        self._pending_timers: Dict[str, Tuple[LatencyType, str, float]] = {}
        self._callbacks: List[Callable[[LatencyMeasurement], None]] = []
        self._alert_callbacks: List[Callable[[LatencyMeasurement], None]] = []
        self._lock = asyncio.Lock()
        self._alert_threshold_ms = self.ALERT_THRESHOLD_MS
        
        logger.info("LatencyTracker initialized")
    
    def start_timer(self, timer_id: str, latency_type: LatencyType, 
                    exchange: str, operation: str = "") -> str:
        """
        Start a latency timer.
        
        Args:
            timer_id: Unique identifier for this timer
            latency_type: Type of latency being measured
            exchange: Exchange name
            operation: Specific operation being timed
            
        Returns:
            timer_id for stopping the timer
        """
        start_time = time.perf_counter()
        self._pending_timers[timer_id] = (latency_type, exchange, start_time)
        return timer_id
    
    def stop_timer(self, timer_id: str, success: bool = True, 
                   metadata: Optional[Dict] = None) -> Optional[LatencyMeasurement]:
        """
        Stop a latency timer and record the measurement.
        
        Args:
            timer_id: Timer ID from start_timer
            success: Whether the operation succeeded
            metadata: Additional metadata
            
        Returns:
            LatencyMeasurement if timer found, None otherwise
        """
        if timer_id not in self._pending_timers:
            logger.warning(f"Timer {timer_id} not found")
            return None
        
        latency_type, exchange, start_time = self._pending_timers.pop(timer_id)
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        
        measurement = LatencyMeasurement(
            latency_type=latency_type,
            timestamp=datetime.utcnow(),
            latency_ms=latency_ms,
            exchange=exchange,
            operation="",
            success=success,
            metadata=metadata or {}
        )
        
        asyncio.create_task(self._record_measurement(measurement))
        return measurement
    
    async def _record_measurement(self, measurement: LatencyMeasurement):
        """Record a latency measurement."""
        key = (measurement.latency_type, measurement.exchange)
        
        async with self._lock:
            if key not in self._measurements:
                self._measurements[key] = deque(maxlen=self.window_size)
            
            self._measurements[key].append(measurement)
        
        # Notify callbacks
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(measurement))
                else:
                    callback(measurement)
            except Exception as e:
                logger.error(f"Error in latency callback: {e}")
        
        # Check alert threshold
        if measurement.latency_ms > self._alert_threshold_ms:
            await self._trigger_alert(measurement)
    
    async def _trigger_alert(self, measurement: LatencyMeasurement):
        """Trigger latency alert."""
        logger.warning(
            f"High latency alert: {measurement.latency_type.value} "
            f"on {measurement.exchange}: {measurement.latency_ms:.2f}ms"
        )
        
        for callback in self._alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(measurement))
                else:
                    callback(measurement)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def record_latency(self, latency_type: LatencyType, exchange: str,
                       latency_ms: float, operation: str = "",
                       success: bool = True, metadata: Optional[Dict] = None):
        """Record a latency measurement directly."""
        measurement = LatencyMeasurement(
            latency_type=latency_type,
            timestamp=datetime.utcnow(),
            latency_ms=latency_ms,
            exchange=exchange,
            operation=operation,
            success=success,
            metadata=metadata or {}
        )
        asyncio.create_task(self._record_measurement(measurement))
    
    async def get_stats(self, latency_type: Optional[LatencyType] = None,
                        exchange: Optional[str] = None) -> Dict:
        """
        Get latency statistics.
        
        Args:
            latency_type: Filter by type (optional)
            exchange: Filter by exchange (optional)
            
        Returns:
            Dictionary of statistics
        """
        async with self._lock:
            results = {}
            
            for key, measurements in self._measurements.items():
                m_type, m_exchange = key
                
                if latency_type and m_type != latency_type:
                    continue
                if exchange and m_exchange != exchange:
                    continue
                
                latency_values = [m.latency_ms for m in measurements]
                stats = LatencyStats(latency_type=m_type, exchange=m_exchange)
                stats.update(latency_values)
                
                results[f"{m_type.value}_{m_exchange}"] = stats.to_dict()
            
            return results
    
    def get_recent_measurements(self, latency_type: LatencyType, 
                                 exchange: str, count: int = 100) -> List[LatencyMeasurement]:
        """Get recent measurements for a type/exchange."""
        key = (latency_type, exchange)
        if key not in self._measurements:
            return []
        
        measurements = list(self._measurements[key])
        return measurements[-count:]
    
    def register_callback(self, callback: Callable[[LatencyMeasurement], None]):
        """Register a callback for new measurements."""
        self._callbacks.append(callback)
    
    def unregister_callback(self, callback: Callable[[LatencyMeasurement], None]):
        """Unregister a callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def register_alert_callback(self, callback: Callable[[LatencyMeasurement], None]):
        """Register a callback for latency alerts."""
        self._alert_callbacks.append(callback)
    
    def set_alert_threshold(self, threshold_ms: float):
        """Set the alert threshold in milliseconds."""
        self._alert_threshold_ms = threshold_ms
    
    async def clear_measurements(self, latency_type: Optional[LatencyType] = None,
                                  exchange: Optional[str] = None):
        """Clear stored measurements."""
        async with self._lock:
            keys_to_remove = []
            
            for key in self._measurements:
                m_type, m_exchange = key
                
                if latency_type and m_type != latency_type:
                    continue
                if exchange and m_exchange != exchange:
                    continue
                
                keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self._measurements[key]
    
    def get_summary(self) -> Dict:
        """Get a summary of all latency statistics."""
        summary = {
            "total_measurements": sum(len(m) for m in self._measurements.values()),
            "tracked_types": list(set(k[0].value for k in self._measurements.keys())),
            "tracked_exchanges": list(set(k[1] for k in self._measurements.keys())),
            "alert_threshold_ms": self._alert_threshold_ms,
        }
        
        # Add per-exchange summary
        exchange_stats = {}
        for (m_type, m_exchange), measurements in self._measurements.items():
            if m_exchange not in exchange_stats:
                exchange_stats[m_exchange] = {
                    "total_measurements": 0,
                    "types": []
                }
            exchange_stats[m_exchange]["total_measurements"] += len(measurements)
            exchange_stats[m_exchange]["types"].append(m_type.value)
        
        summary["exchange_stats"] = exchange_stats
        return summary


class AsyncLatencyContext:
    """Async context manager for measuring latency."""
    
    def __init__(self, tracker: LatencyTracker, latency_type: LatencyType,
                 exchange: str, operation: str = ""):
        self.tracker = tracker
        self.latency_type = latency_type
        self.exchange = exchange
        self.operation = operation
        self.timer_id: Optional[str] = None
        self.measurement: Optional[LatencyMeasurement] = None
    
    async def __aenter__(self):
        self.timer_id = self.tracker.start_timer(
            f"{self.exchange}_{self.latency_type.value}_{id(self)}",
            self.latency_type,
            self.exchange,
            self.operation
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        success = exc_type is None
        self.measurement = self.tracker.stop_timer(self.timer_id, success)


def measure_latency(latency_type: LatencyType, exchange: str, operation: str = ""):
    """Decorator for measuring function latency."""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            tracker = kwargs.get('latency_tracker') or (
                args[0].latency_tracker if args and hasattr(args[0], 'latency_tracker') else None
            )
            
            if tracker is None:
                return await func(*args, **kwargs)
            
            async with AsyncLatencyContext(tracker, latency_type, exchange, operation):
                return await func(*args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            tracker = kwargs.get('latency_tracker') or (
                args[0].latency_tracker if args and hasattr(args[0], 'latency_tracker') else None
            )
            
            if tracker is None:
                return func(*args, **kwargs)
            
            timer_id = tracker.start_timer(
                f"{exchange}_{latency_type.value}_{id(func)}",
                latency_type,
                exchange,
                operation
            )
            try:
                result = func(*args, **kwargs)
                tracker.stop_timer(timer_id, True)
                return result
            except Exception as e:
                tracker.stop_timer(timer_id, False, {"error": str(e)})
                raise
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator


# Global latency tracker instance
global_latency_tracker = LatencyTracker()
