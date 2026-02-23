"""
Drawdown Control Module for TradeOS Risk Engine

Implements comprehensive drawdown protection:
- Current drawdown calculator
- Max drawdown circuit breaker
- Daily/weekly loss limits
- Equity curve monitoring
- Automatic trading halt on limits

STRICT ENFORCEMENT: No trade bypasses drawdown controls.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Callable, Any
from collections import deque
import threading

from .models.risk_profile import (
    DrawdownState,
    PortfolioState,
    RiskLevel,
)
from .config.risk_limits import get_risk_limits, SubscriptionTier, CIRCUIT_BREAKER_COOLDOWNS


logger = logging.getLogger(__name__)


class DrawdownController:
    """
    Drawdown monitoring and control system.
    Implements circuit breakers and automatic trading halts.
    """
    
    def __init__(
        self,
        tier: SubscriptionTier = SubscriptionTier.PRO,
        callback_on_halt: Optional[Callable[[str], None]] = None,
        callback_on_resume: Optional[Callable[[], None]] = None,
    ):
        self.risk_limits = get_risk_limits(tier)
        self.state = DrawdownState()
        self.callback_on_halt = callback_on_halt
        self.callback_on_resume = callback_on_resume
        
        # Historical tracking
        self._equity_history: deque = deque(maxlen=1000)
        self._daily_pnl_history: deque = deque(maxlen=30)
        self._weekly_pnl_history: deque = deque(maxlen=52)
        
        # Tracking state
        self._day_start_equity: Optional[Decimal] = None
        self._week_start_equity: Optional[Decimal] = None
        self._last_update: Optional[datetime] = None
        
        # Consecutive violation tracking
        self._consecutive_drawdown_breaches = 0
        self._consecutive_daily_loss_breaches = 0
        self._consecutive_weekly_loss_breaches = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info("DrawdownController initialized")
    
    def update(self, portfolio_state: PortfolioState) -> Dict[str, Any]:
        """
        Update drawdown state with current portfolio data.
        Returns status and any triggered alerts.
        
        Args:
            portfolio_state: Current portfolio state
            
        Returns:
            Dictionary with status, alerts, and triggered actions
        """
        with self._lock:
            result = {
                "updated": True,
                "alerts": [],
                "actions": [],
                "trading_halted": False,
                "risk_level": RiskLevel.LOW,
            }
            
            current_equity = portfolio_state.total_equity
            current_time = datetime.utcnow()
            
            # Initialize if first update
            if self.state.peak_equity == 0:
                self.state.peak_equity = current_equity
                self._day_start_equity = current_equity
                self._week_start_equity = current_equity
            
            # Update peak equity
            if current_equity > self.state.peak_equity:
                self.state.peak_equity = current_equity
                # End drawdown if we made new highs
                if self.state.in_drawdown:
                    self.state.in_drawdown = False
                    self.state.drawdown_start = None
                    logger.info(f"Drawdown ended. New peak: {current_equity}")
            
            # Calculate current drawdown
            if self.state.peak_equity > 0:
                self.state.current_drawdown_pct = (
                    (self.state.peak_equity - current_equity) / self.state.peak_equity
                )
            else:
                self.state.current_drawdown_pct = Decimal("0")
            
            # Update max drawdown
            if self.state.current_drawdown_pct > self.state.max_drawdown_pct:
                self.state.max_drawdown_pct = self.state.current_drawdown_pct
                self.state.max_drawdown_start = self.state.drawdown_start
                self.state.max_drawdown_end = current_time
            
            # Check if in drawdown
            if self.state.current_drawdown_pct > 0 and not self.state.in_drawdown:
                self.state.in_drawdown = True
                self.state.drawdown_start = current_time
                self.state.trough_equity = current_equity
            
            # Update trough
            if current_equity < self.state.trough_equity:
                self.state.trough_equity = current_equity
            
            # Calculate daily and weekly P&L
            self._update_periodic_pnl(current_equity, current_time)
            
            # Store equity history
            self._equity_history.append({
                "timestamp": current_time,
                "equity": current_equity,
                "drawdown_pct": self.state.current_drawdown_pct,
            })
            
            # Check all limits and circuit breakers
            checks = [
                self._check_drawdown_limits(result),
                self._check_daily_loss_limits(result),
                self._check_weekly_loss_limits(result),
                self._check_circuit_breakers(result),
            ]
            
            # Determine risk level
            result["risk_level"] = self._calculate_risk_level()
            
            # Update state
            self._last_update = current_time
            self.state.trading_halted = result["trading_halted"]
            
            return result
    
    def _update_periodic_pnl(self, current_equity: Decimal, current_time: datetime):
        """Update daily and weekly P&L calculations"""
        # Check if new day
        if self._last_update and self._day_start_equity:
            if current_time.date() != self._last_update.date():
                # New day - store previous day's P&L and reset
                daily_pnl = current_equity - self._day_start_equity
                daily_pnl_pct = daily_pnl / self._day_start_equity if self._day_start_equity > 0 else Decimal("0")
                self._daily_pnl_history.append({
                    "date": self._last_update.date(),
                    "pnl": daily_pnl,
                    "pnl_pct": daily_pnl_pct,
                })
                self._day_start_equity = current_equity
                self._consecutive_daily_loss_breaches = 0
        
        # Check if new week
        if self._last_update and self._week_start_equity:
            current_week = current_time.isocalendar()[1]
            last_week = self._last_update.isocalendar()[1]
            if current_week != last_week:
                # New week - store previous week's P&L and reset
                weekly_pnl = current_equity - self._week_start_equity
                weekly_pnl_pct = weekly_pnl / self._week_start_equity if self._week_start_equity > 0 else Decimal("0")
                self._weekly_pnl_history.append({
                    "week": last_week,
                    "year": self._last_update.year,
                    "pnl": weekly_pnl,
                    "pnl_pct": weekly_pnl_pct,
                })
                self._week_start_equity = current_equity
                self._consecutive_weekly_loss_breaches = 0
        
        # Calculate current daily P&L
        if self._day_start_equity and self._day_start_equity > 0:
            daily_pnl = current_equity - self._day_start_equity
            self.state.daily_loss_pct = -daily_pnl / self._day_start_equity if daily_pnl < 0 else Decimal("0")
        
        # Calculate current weekly P&L
        if self._week_start_equity and self._week_start_equity > 0:
            weekly_pnl = current_equity - self._week_start_equity
            self.state.weekly_loss_pct = -weekly_pnl / self._week_start_equity if weekly_pnl < 0 else Decimal("0")
    
    def _check_drawdown_limits(self, result: Dict[str, Any]) -> bool:
        """Check drawdown limits and generate alerts"""
        current_dd = self.state.current_drawdown_pct
        max_dd = self.risk_limits.max_drawdown_pct
        circuit_dd = self.risk_limits.circuit_breaker_drawdown_pct
        
        # Alert at 80% of max drawdown
        alert_threshold = max_dd * Decimal("0.80")
        
        if current_dd >= circuit_dd:
            # Circuit breaker triggered
            self._consecutive_drawdown_breaches += 1
            msg = f"CRITICAL: Circuit breaker drawdown reached: {current_dd:.2%} (limit: {circuit_dd:.2%})"
            result["alerts"].append(msg)
            result["actions"].append("TRADING_HALT")
            result["trading_halted"] = True
            result["risk_level"] = RiskLevel.EMERGENCY
            self._halt_trading(msg)
            logger.critical(msg)
            return False
        
        elif current_dd >= max_dd:
            # Max drawdown exceeded
            self._consecutive_drawdown_breaches += 1
            msg = f"HIGH: Max drawdown exceeded: {current_dd:.2%} (limit: {max_dd:.2%})"
            result["alerts"].append(msg)
            result["actions"].append("REDUCE_EXPOSURE")
            result["risk_level"] = RiskLevel.HIGH
            logger.warning(msg)
            return False
        
        elif current_dd >= alert_threshold:
            # Approaching limit
            msg = f"MEDIUM: Approaching max drawdown: {current_dd:.2%} (alert: {alert_threshold:.2%})"
            result["alerts"].append(msg)
            result["risk_level"] = RiskLevel.MEDIUM
            logger.info(msg)
        
        else:
            # Reset consecutive breaches
            self._consecutive_drawdown_breaches = 0
        
        return True
    
    def _check_daily_loss_limits(self, result: Dict[str, Any]) -> bool:
        """Check daily loss limits"""
        daily_loss = self.state.daily_loss_pct
        max_daily = self.risk_limits.max_daily_loss_pct
        kill_switch = self.risk_limits.kill_switch_daily_loss_pct
        
        # Alert at 75% of limit
        alert_threshold = max_daily * Decimal("0.75")
        
        if daily_loss >= kill_switch:
            # Kill switch triggered
            self._consecutive_daily_loss_breaches += 1
            msg = f"CRITICAL: Daily loss kill switch triggered: {daily_loss:.2%} (limit: {kill_switch:.2%})"
            result["alerts"].append(msg)
            result["actions"].append("KILL_SWITCH")
            result["trading_halted"] = True
            result["risk_level"] = RiskLevel.EMERGENCY
            self._halt_trading(msg)
            logger.critical(msg)
            return False
        
        elif daily_loss >= max_daily:
            # Daily limit exceeded
            self._consecutive_daily_loss_breaches += 1
            msg = f"HIGH: Daily loss limit exceeded: {daily_loss:.2%} (limit: {max_daily:.2%})"
            result["alerts"].append(msg)
            result["actions"].append("HALT_TRADING")
            result["trading_halted"] = True
            result["risk_level"] = RiskLevel.HIGH
            self._halt_trading(msg)
            logger.warning(msg)
            return False
        
        elif daily_loss >= alert_threshold:
            # Approaching limit
            msg = f"MEDIUM: Approaching daily loss limit: {daily_loss:.2%} (alert: {alert_threshold:.2%})"
            result["alerts"].append(msg)
            result["risk_level"] = RiskLevel.MEDIUM
            logger.info(msg)
        
        else:
            self._consecutive_daily_loss_breaches = 0
        
        return True
    
    def _check_weekly_loss_limits(self, result: Dict[str, Any]) -> bool:
        """Check weekly loss limits"""
        weekly_loss = self.state.weekly_loss_pct
        max_weekly = self.risk_limits.max_weekly_loss_pct
        
        # Alert at 80% of limit
        alert_threshold = max_weekly * Decimal("0.80")
        
        if weekly_loss >= max_weekly:
            # Weekly limit exceeded
            self._consecutive_weekly_loss_breaches += 1
            msg = f"HIGH: Weekly loss limit exceeded: {weekly_loss:.2%} (limit: {max_weekly:.2%})"
            result["alerts"].append(msg)
            result["actions"].append("REDUCE_EXPOSURE")
            result["risk_level"] = RiskLevel.HIGH
            logger.warning(msg)
            return False
        
        elif weekly_loss >= alert_threshold:
            # Approaching limit
            msg = f"MEDIUM: Approaching weekly loss limit: {weekly_loss:.2%} (alert: {alert_threshold:.2%})"
            result["alerts"].append(msg)
            result["risk_level"] = RiskLevel.MEDIUM
            logger.info(msg)
        
        else:
            self._consecutive_weekly_loss_breaches = 0
        
        return True
    
    def _check_circuit_breakers(self, result: Dict[str, Any]) -> bool:
        """Check all circuit breaker conditions"""
        # Check if trading is halted and if we can resume
        if self.state.trading_halted:
            if self.state.halt_until and datetime.utcnow() >= self.state.halt_until:
                # Cooldown period expired
                if self.risk_limits.auto_resume_after_halt:
                    self._resume_trading()
                    result["actions"].append("AUTO_RESUME")
                    result["trading_halted"] = False
                    logger.info("Trading auto-resumed after cooldown")
                else:
                    result["alerts"].append("Trading halt cooldown expired, manual resume required")
            else:
                result["trading_halted"] = True
        
        return True
    
    def _calculate_risk_level(self) -> RiskLevel:
        """Calculate overall risk level based on drawdown state"""
        # Emergency conditions
        if (self.state.current_drawdown_pct >= self.risk_limits.circuit_breaker_drawdown_pct or
            self.state.daily_loss_pct >= self.risk_limits.kill_switch_daily_loss_pct):
            return RiskLevel.EMERGENCY
        
        # Critical conditions
        if (self.state.current_drawdown_pct >= self.risk_limits.max_drawdown_pct or
            self.state.daily_loss_pct >= self.risk_limits.max_daily_loss_pct or
            self.state.weekly_loss_pct >= self.risk_limits.max_weekly_loss_pct):
            return RiskLevel.CRITICAL
        
        # High risk conditions
        if (self.state.current_drawdown_pct >= self.risk_limits.max_drawdown_pct * Decimal("0.80") or
            self.state.daily_loss_pct >= self.risk_limits.max_daily_loss_pct * Decimal("0.80")):
            return RiskLevel.HIGH
        
        # Medium risk conditions
        if (self.state.current_drawdown_pct >= self.risk_limits.max_drawdown_pct * Decimal("0.50") or
            self.state.daily_loss_pct >= self.risk_limits.max_daily_loss_pct * Decimal("0.50")):
            return RiskLevel.MEDIUM
        
        return RiskLevel.LOW
    
    def _halt_trading(self, reason: str):
        """Halt trading with reason"""
        self.state.trading_halted = True
        self.state.halt_reason = reason
        cooldown = self.risk_limits.trading_halt_cooldown_minutes
        self.state.halt_until = datetime.utcnow() + timedelta(minutes=cooldown)
        
        if self.callback_on_halt:
            try:
                self.callback_on_halt(reason)
            except Exception as e:
                logger.error(f"Error in halt callback: {e}")
    
    def _resume_trading(self):
        """Resume trading"""
        self.state.trading_halted = False
        self.state.halt_reason = None
        self.state.halt_until = None
        
        if self.callback_on_resume:
            try:
                self.callback_on_resume()
            except Exception as e:
                logger.error(f"Error in resume callback: {e}")
    
    def manual_resume(self) -> bool:
        """Manually resume trading after halt"""
        with self._lock:
            if not self.state.trading_halted:
                logger.warning("Trading not halted, no action taken")
                return False
            
            self._resume_trading()
            logger.info("Trading manually resumed")
            return True
    
    def can_trade(self) -> bool:
        """Check if trading is currently allowed"""
        with self._lock:
            return not self.state.trading_halted
    
    def get_drawdown_report(self) -> Dict[str, Any]:
        """Get comprehensive drawdown report"""
        with self._lock:
            return {
                "current_drawdown_pct": float(self.state.current_drawdown_pct),
                "max_drawdown_pct": float(self.state.max_drawdown_pct),
                "in_drawdown": self.state.in_drawdown,
                "drawdown_start": self.state.drawdown_start.isoformat() if self.state.drawdown_start else None,
                "daily_loss_pct": float(self.state.daily_loss_pct),
                "weekly_loss_pct": float(self.state.weekly_loss_pct),
                "trading_halted": self.state.trading_halted,
                "halt_reason": self.state.halt_reason,
                "halt_until": self.state.halt_until.isoformat() if self.state.halt_until else None,
                "peak_equity": float(self.state.peak_equity),
                "trough_equity": float(self.state.trough_equity),
                "risk_level": self._calculate_risk_level().name,
                "limits": {
                    "max_drawdown_pct": float(self.risk_limits.max_drawdown_pct),
                    "circuit_breaker_drawdown_pct": float(self.risk_limits.circuit_breaker_drawdown_pct),
                    "max_daily_loss_pct": float(self.risk_limits.max_daily_loss_pct),
                    "max_weekly_loss_pct": float(self.risk_limits.max_weekly_loss_pct),
                }
            }
    
    def get_equity_curve(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get equity curve data for visualization"""
        with self._lock:
            return list(self._equity_history)[-days * 24:]  # Approximate hourly data
    
    def reset(self, new_peak: Optional[Decimal] = None):
        """Reset drawdown tracking (use with caution)"""
        with self._lock:
            self.state = DrawdownState()
            if new_peak:
                self.state.peak_equity = new_peak
            self._equity_history.clear()
            self._daily_pnl_history.clear()
            self._weekly_pnl_history.clear()
            self._consecutive_drawdown_breaches = 0
            self._consecutive_daily_loss_breaches = 0
            self._consecutive_weekly_loss_breaches = 0
            logger.warning("Drawdown state reset")
