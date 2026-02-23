"""
Kill Switch Module for TradeOS Risk Engine

Emergency controls for extreme risk events:
- Manual kill switch
- Automatic kill on extreme events
- Kill switch recovery procedures
- Alert notifications

CRITICAL: Once triggered, all trading stops immediately.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Callable, Any, Set
from enum import Enum, auto
import threading

from .models.risk_profile import (
    KillSwitchState,
    RiskLevel,
)
from .config.risk_limits import get_risk_limits, SubscriptionTier


logger = logging.getLogger(__name__)


class KillSwitchTrigger(Enum):
    """Reasons for kill switch activation"""
    MANUAL = auto()
    DAILY_LOSS_LIMIT = auto()
    DRAWDOWN_LIMIT = auto()
    VAR_BREACH = auto()
    EXPOSURE_LIMIT = auto()
    LEVERAGE_LIMIT = auto()
    MARGIN_CALL = auto()
    CORRELATION_SPIKE = auto()
    PORTFOLIO_HEAT = auto()
    SYSTEM_ERROR = auto()
    EXTERNAL_SIGNAL = auto()
    CONSECUTIVE_VIOLATIONS = auto()


class KillSwitch:
    """
    Emergency kill switch for trading system.
    Provides immediate halt of all trading activity.
    """
    
    def __init__(
        self,
        tier: SubscriptionTier = SubscriptionTier.PRO,
        on_kill: Optional[Callable[[KillSwitchTrigger, str], None]] = None,
        on_recover: Optional[Callable[[], None]] = None,
        notification_callbacks: Optional[List[Callable[[str, Dict], None]]] = None,
    ):
        self.risk_limits = get_risk_limits(tier)
        self.state = KillSwitchState()
        self.on_kill = on_kill
        self.on_recover = on_recover
        self.notification_callbacks = notification_callbacks or []
        
        # Trigger history
        self._trigger_history: List[Dict[str, Any]] = []
        self._max_history_size = 100
        
        # Consecutive violation tracking
        self._violation_counts: Dict[str, int] = {}
        self._max_consecutive_violations = 3
        
        # Recovery settings
        self._recovery_cooldown_minutes = 30
        self._required_confirmations = 2
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info("KillSwitch initialized")
    
    def trigger(
        self,
        reason: KillSwitchTrigger,
        details: Optional[Dict[str, Any]] = None,
        message: Optional[str] = None,
    ) -> bool:
        """
        Trigger the kill switch.
        
        Args:
            reason: Trigger reason
            details: Additional details
            message: Human-readable message
            
        Returns:
            True if kill switch was activated
        """
        with self._lock:
            if self.state.is_active:
                logger.warning(f"Kill switch already active, trigger ignored: {reason.name}")
                return False
            
            current_time = datetime.utcnow()
            
            # Update state
            self.state.is_active = True
            self.state.triggered_at = current_time
            self.state.triggered_by = reason.name
            self.state.reason = message or f"Kill switch triggered: {reason.name}"
            self.state.can_resume_at = current_time + timedelta(
                minutes=self._recovery_cooldown_minutes
            )
            self.state.manual_override = False
            
            # Update daily trigger count
            if self.state.last_trigger_date != current_time.date():
                self.state.last_trigger_date = current_time.date()
                self.state.trigger_count_today = 1
            else:
                self.state.trigger_count_today += 1
            
            # Log to history
            self._trigger_history.append({
                "timestamp": current_time.isoformat(),
                "reason": reason.name,
                "details": details or {},
                "message": message,
            })
            
            # Trim history if needed
            if len(self._trigger_history) > self._max_history_size:
                self._trigger_history = self._trigger_history[-self._max_history_size:]
            
            # Execute kill callback
            if self.on_kill:
                try:
                    self.on_kill(reason, self.state.reason)
                except Exception as e:
                    logger.error(f"Error in kill callback: {e}")
            
            # Send notifications
            self._send_notifications("KILL_SWITCH_TRIGGERED", {
                "reason": reason.name,
                "message": self.state.reason,
                "timestamp": current_time.isoformat(),
                "details": details or {},
            })
            
            logger.critical(f"KILL SWITCH ACTIVATED: {reason.name} - {self.state.reason}")
            
            return True
    
    def manual_trigger(self, message: str, user_id: Optional[str] = None) -> bool:
        """
        Manually trigger the kill switch.
        
        Args:
            message: Reason for manual trigger
            user_id: ID of user triggering
            
        Returns:
            True if kill switch was activated
        """
        return self.trigger(
            reason=KillSwitchTrigger.MANUAL,
            message=f"Manual trigger by {user_id or 'unknown'}: {message}",
            details={"user_id": user_id, "manual_message": message},
        )
    
    def release(self, user_id: Optional[str] = None, force: bool = False) -> bool:
        """
        Release the kill switch.
        
        Args:
            user_id: ID of user releasing
            force: Force release even if cooldown not expired
            
        Returns:
            True if kill switch was released
        """
        with self._lock:
            if not self.state.is_active:
                logger.warning("Kill switch not active, release ignored")
                return False
            
            current_time = datetime.utcnow()
            
            # Check cooldown
            if not force and self.state.can_resume_at:
                if current_time < self.state.can_resume_at:
                    remaining = (self.state.can_resume_at - current_time).total_seconds() / 60
                    logger.warning(
                        f"Cannot release kill switch: cooldown active ({remaining:.1f} minutes remaining)"
                    )
                    return False
            
            # Check daily trigger limit
            if self.state.trigger_count_today >= 3:
                logger.warning(
                    f"Kill switch triggered {self.state.trigger_count_today} times today, "
                    "manual override required"
                )
                if not force:
                    return False
            
            # Update state
            self.state.is_active = False
            self.state.manual_override = True
            
            # Log release
            release_message = f"Kill switch released by {user_id or 'system'}"
            if force:
                release_message += " (forced)"
            
            logger.critical(release_message)
            
            # Execute recover callback
            if self.on_recover:
                try:
                    self.on_recover()
                except Exception as e:
                    logger.error(f"Error in recover callback: {e}")
            
            # Send notifications
            self._send_notifications("KILL_SWITCH_RELEASED", {
                "message": release_message,
                "timestamp": current_time.isoformat(),
                "previous_trigger": self.state.triggered_by,
                "released_by": user_id,
                "forced": force,
            })
            
            return True
    
    def check_auto_trigger(
        self,
        daily_loss_pct: Decimal,
        drawdown_pct: Decimal,
        var_pct: Optional[Decimal] = None,
        violations: Optional[List[Dict]] = None,
    ) -> Optional[KillSwitchTrigger]:
        """
        Check if auto-trigger conditions are met.
        
        Args:
            daily_loss_pct: Current daily loss percentage
            drawdown_pct: Current drawdown percentage
            var_pct: Current VaR percentage
            violations: List of current violations
            
        Returns:
            KillSwitchTrigger if conditions met, None otherwise
        """
        with self._lock:
            # Check daily loss limit
            if daily_loss_pct >= self.risk_limits.kill_switch_daily_loss_pct:
                logger.critical(
                    f"Auto-trigger: Daily loss {daily_loss_pct:.2%} >= "
                    f"limit {self.risk_limits.kill_switch_daily_loss_pct:.2%}"
                )
                return KillSwitchTrigger.DAILY_LOSS_LIMIT
            
            # Check drawdown limit
            if drawdown_pct >= self.risk_limits.kill_switch_drawdown_pct:
                logger.critical(
                    f"Auto-trigger: Drawdown {drawdown_pct:.2%} >= "
                    f"limit {self.risk_limits.kill_switch_drawdown_pct:.2%}"
                )
                return KillSwitchTrigger.DRAWDOWN_LIMIT
            
            # Check VaR
            if var_pct is not None:
                if var_pct >= self.risk_limits.max_daily_var_pct * Decimal("2"):
                    logger.critical(f"Auto-trigger: VaR {var_pct:.2%} exceeds critical threshold")
                    return KillSwitchTrigger.VAR_BREACH
            
            # Check consecutive violations
            if violations:
                critical_violations = [v for v in violations if v.get("severity") == "critical"]
                if len(critical_violations) >= 3:
                    logger.critical(f"Auto-trigger: {len(critical_violations)} critical violations")
                    return KillSwitchTrigger.CONSECUTIVE_VIOLATIONS
            
            return None
    
    def record_violation(self, violation_type: str, severity: str = "medium"):
        """Record a risk violation for consecutive tracking"""
        with self._lock:
            key = f"{violation_type}_{severity}"
            self._violation_counts[key] = self._violation_counts.get(key, 0) + 1
            
            # Check if we've hit the consecutive limit
            if self._violation_counts[key] >= self._max_consecutive_violations:
                logger.warning(
                    f"Max consecutive violations reached for {key}: "
                    f"{self._violation_counts[key]}"
                )
                self.trigger(
                    KillSwitchTrigger.CONSECUTIVE_VIOLATIONS,
                    message=f"Max consecutive violations: {violation_type}",
                    details={"violation_type": violation_type, "count": self._violation_counts[key]},
                )
    
    def reset_violation_count(self, violation_type: Optional[str] = None):
        """Reset violation counts"""
        with self._lock:
            if violation_type:
                keys_to_remove = [k for k in self._violation_counts if k.startswith(violation_type)]
                for key in keys_to_remove:
                    del self._violation_counts[key]
            else:
                self._violation_counts.clear()
    
    def is_active(self) -> bool:
        """Check if kill switch is currently active"""
        with self._lock:
            return self.state.is_active
    
    def can_trade(self) -> bool:
        """Check if trading is allowed"""
        with self._lock:
            return not self.state.is_active
    
    def get_status(self) -> Dict[str, Any]:
        """Get kill switch status"""
        with self._lock:
            return {
                "is_active": self.state.is_active,
                "triggered_at": self.state.triggered_at.isoformat() if self.state.triggered_at else None,
                "triggered_by": self.state.triggered_by,
                "reason": self.state.reason,
                "can_resume_at": self.state.can_resume_at.isoformat() if self.state.can_resume_at else None,
                "manual_override": self.state.manual_override,
                "trigger_count_today": self.state.trigger_count_today,
                "last_trigger_date": self.state.last_trigger_date.isoformat() if self.state.last_trigger_date else None,
                "time_until_resume": self._get_time_until_resume(),
            }
    
    def _get_time_until_resume(self) -> Optional[float]:
        """Get minutes until trading can resume"""
        if not self.state.is_active or not self.state.can_resume_at:
            return None
        
        remaining = (self.state.can_resume_at - datetime.utcnow()).total_seconds() / 60
        return max(0, remaining)
    
    def get_trigger_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get kill switch trigger history"""
        with self._lock:
            return self._trigger_history[-limit:]
    
    def _send_notifications(self, event_type: str, data: Dict[str, Any]):
        """Send notifications to all registered callbacks"""
        for callback in self.notification_callbacks:
            try:
                callback(event_type, data)
            except Exception as e:
                logger.error(f"Error in notification callback: {e}")
    
    def add_notification_callback(self, callback: Callable[[str, Dict], None]):
        """Add a notification callback"""
        self.notification_callbacks.append(callback)
    
    def remove_notification_callback(self, callback: Callable[[str, Dict], None]):
        """Remove a notification callback"""
        if callback in self.notification_callbacks:
            self.notification_callbacks.remove(callback)
    
    def reset(self):
        """Reset kill switch state (use with extreme caution)"""
        with self._lock:
            was_active = self.state.is_active
            self.state = KillSwitchState()
            self._violation_counts.clear()
            
            if was_active and self.on_recover:
                try:
                    self.on_recover()
                except Exception as e:
                    logger.error(f"Error in recover callback during reset: {e}")
            
            logger.critical("Kill switch state reset")


class CircuitBreaker:
    """
    Circuit breaker for specific risk conditions.
    Less severe than kill switch - allows for automatic recovery.
    """
    
    def __init__(
        self,
        name: str,
        cooldown_minutes: int = 15,
        auto_reset: bool = True,
    ):
        self.name = name
        self.cooldown_minutes = cooldown_minutes
        self.auto_reset = auto_reset
        
        self._is_tripped = False
        self._tripped_at: Optional[datetime] = None
        self._trip_count = 0
        self._last_trip_date: Optional[datetime.date] = None
        
        self._lock = threading.RLock()
    
    def trip(self, reason: str) -> bool:
        """Trip the circuit breaker"""
        with self._lock:
            if self._is_tripped:
                return False
            
            current_time = datetime.utcnow()
            
            self._is_tripped = True
            self._tripped_at = current_time
            
            # Update daily count
            if self._last_trip_date != current_time.date():
                self._last_trip_date = current_time.date()
                self._trip_count = 1
            else:
                self._trip_count += 1
            
            logger.warning(f"Circuit breaker '{self.name}' tripped: {reason}")
            return True
    
    def reset(self) -> bool:
        """Reset the circuit breaker"""
        with self._lock:
            if not self._is_tripped:
                return False
            
            self._is_tripped = False
            self._tripped_at = None
            
            logger.info(f"Circuit breaker '{self.name}' reset")
            return True
    
    def check(self) -> bool:
        """Check circuit breaker status and auto-reset if needed"""
        with self._lock:
            if not self._is_tripped:
                return True
            
            if self.auto_reset and self._tripped_at:
                elapsed = (datetime.utcnow() - self._tripped_at).total_seconds() / 60
                if elapsed >= self.cooldown_minutes:
                    self.reset()
                    return True
            
            return False
    
    def is_tripped(self) -> bool:
        """Check if circuit breaker is tripped"""
        with self._lock:
            return self._is_tripped
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status"""
        with self._lock:
            return {
                "name": self.name,
                "is_tripped": self._is_tripped,
                "tripped_at": self._tripped_at.isoformat() if self._tripped_at else None,
                "trip_count_today": self._trip_count,
                "cooldown_minutes": self.cooldown_minutes,
                "auto_reset": self.auto_reset,
            }


class CircuitBreakerPanel:
    """Panel of circuit breakers for different risk conditions"""
    
    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.RLock()
    
    def register(self, name: str, cooldown_minutes: int = 15, auto_reset: bool = True):
        """Register a new circuit breaker"""
        with self._lock:
            self._breakers[name] = CircuitBreaker(name, cooldown_minutes, auto_reset)
    
    def trip(self, name: str, reason: str) -> bool:
        """Trip a circuit breaker"""
        with self._lock:
            if name not in self._breakers:
                logger.error(f"Circuit breaker '{name}' not found")
                return False
            return self._breakers[name].trip(reason)
    
    def reset(self, name: str) -> bool:
        """Reset a circuit breaker"""
        with self._lock:
            if name not in self._breakers:
                return False
            return self._breakers[name].reset()
    
    def check(self, name: str) -> bool:
        """Check a circuit breaker status"""
        with self._lock:
            if name not in self._breakers:
                return True
            return self._breakers[name].check()
    
    def check_all(self) -> Dict[str, bool]:
        """Check all circuit breakers"""
        with self._lock:
            return {name: breaker.check() for name, breaker in self._breakers.items()}
    
    def can_trade(self) -> bool:
        """Check if all circuit breakers allow trading"""
        with self._lock:
            return all(b.check() for b in self._breakers.values())
    
    def get_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers"""
        with self._lock:
            return {name: breaker.get_status() for name, breaker in self._breakers.items()}
