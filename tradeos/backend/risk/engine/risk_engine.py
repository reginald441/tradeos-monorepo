"""
Risk Engine - Main Coordinator for TradeOS Capital Protection Core

The Risk Engine orchestrates all risk management components:
- Position Sizing
- Drawdown Control
- Exposure Management
- Correlation Monitoring
- Kill Switch
- VaR Calculation
- Risk Validation

STRICT ENFORCEMENT: All trades flow through this engine.
"""

import logging
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Callable, Any, Set
from enum import Enum, auto
import threading

from ..models.risk_profile import (
    TradeRequest,
    ValidationResult,
    PortfolioState,
    Position,
    RiskReport,
    RiskLevel,
    PositionSizingParams,
    PositionSizingMethod,
    SizingRecommendation,
    SubscriptionTier,
    VaRMethod,
)
from ..config.risk_limits import get_risk_limits, SubscriptionTier as ConfigTier
from ..position_sizing import PositionSizer
from ..drawdown_control import DrawdownController
from ..exposure_manager import ExposureManager
from ..correlation_monitor import CorrelationMonitor
from ..kill_switch import KillSwitch, KillSwitchTrigger, CircuitBreakerPanel
from ..var_calculator import VaRCalculator
from ..risk_validator import RiskValidator, ValidationRule


logger = logging.getLogger(__name__)


class RiskEngineState(Enum):
    """Risk engine operational states"""
    INITIALIZING = auto()
    ACTIVE = auto()
    PAUSED = auto()
    EMERGENCY = auto()
    SHUTDOWN = auto()


class RiskEngine:
    """
    Main Risk Engine - Capital Protection Core
    
    Coordinates all risk management components and provides
    a unified interface for trade validation and risk monitoring.
    
    Usage:
        >>> engine = RiskEngine(tier=SubscriptionTier.PRO)
        >>> engine.initialize()
        >>> 
        >>> # Validate a trade
        >>> result = engine.validate_trade(trade_request, portfolio_state)
        >>> if result.is_valid:
        ...     execute_trade(trade_request)
        >>> 
        >>> # Get risk report
        >>> report = engine.generate_risk_report()
    """
    
    def __init__(
        self,
        tier: SubscriptionTier = SubscriptionTier.PRO,
        position_sizing_method: PositionSizingMethod = PositionSizingMethod.RISK_PER_TRADE,
        var_method: VaRMethod = VaRMethod.HISTORICAL,
        auto_update_interval_seconds: int = 60,
        enable_kill_switch: bool = True,
    ):
        """
        Initialize the Risk Engine.
        
        Args:
            tier: Subscription tier for risk limits
            position_sizing_method: Default position sizing method
            var_method: Default VaR calculation method
            auto_update_interval_seconds: Interval for auto-updating risk metrics
            enable_kill_switch: Whether to enable kill switch
        """
        self.tier = tier
        self.risk_limits = get_risk_limits(tier)
        self.state = RiskEngineState.INITIALIZING
        self.auto_update_interval = timedelta(seconds=auto_update_interval_seconds)
        
        # Position sizing parameters
        self.position_sizing_params = PositionSizingParams(
            method=position_sizing_method
        )
        
        # Risk components (initialized in initialize())
        self.position_sizer: Optional[PositionSizer] = None
        self.drawdown_controller: Optional[DrawdownController] = None
        self.exposure_manager: Optional[ExposureManager] = None
        self.correlation_monitor: Optional[CorrelationMonitor] = None
        self.kill_switch: Optional[KillSwitch] = None
        self.var_calculator: Optional[VaRCalculator] = None
        self.validator: Optional[RiskValidator] = None
        self.circuit_breakers: Optional[CircuitBreakerPanel] = None
        
        # Callbacks
        self._on_risk_alert: Optional[Callable[[str, Dict], None]] = None
        self._on_trading_halt: Optional[Callable[[str], None]] = None
        self._on_trading_resume: Optional[Callable[[], None]] = None
        
        # Auto-update
        self._auto_update_task: Optional[asyncio.Task] = None
        self._last_update: Optional[datetime] = None
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info(f"RiskEngine initialized for tier: {tier.value}")
    
    def initialize(self):
        """Initialize all risk components"""
        with self._lock:
            logger.info("Initializing Risk Engine components...")
            
            # Initialize Position Sizer
            self.position_sizer = PositionSizer(
                params=self.position_sizing_params,
                tier=self.tier,
            )
            
            # Initialize Drawdown Controller
            self.drawdown_controller = DrawdownController(
                tier=self.tier,
                callback_on_halt=self._on_drawdown_halt,
                callback_on_resume=self._on_drawdown_resume,
            )
            
            # Initialize Exposure Manager
            self.exposure_manager = ExposureManager(
                tier=self.tier,
                callback_on_limit_breach=self._on_exposure_breach,
            )
            
            # Initialize Correlation Monitor
            self.correlation_monitor = CorrelationMonitor(
                tier=self.tier,
            )
            
            # Initialize Kill Switch
            self.kill_switch = KillSwitch(
                tier=self.tier,
                on_kill=self._on_kill_switch,
                on_recover=self._on_kill_switch_recover,
            )
            
            # Initialize VaR Calculator
            self.var_calculator = VaRCalculator(
                tier=self.tier,
                default_method=VaRMethod.HISTORICAL,
            )
            
            # Initialize Risk Validator
            self.validator = RiskValidator(
                tier=self.tier,
                kill_switch=self.kill_switch,
                drawdown_controller=self.drawdown_controller,
                exposure_manager=self.exposure_manager,
                correlation_monitor=self.correlation_monitor,
                var_calculator=self.var_calculator,
                position_sizer=self.position_sizer,
                on_validation_failure=self._on_validation_failure,
            )
            
            # Initialize Circuit Breakers
            self.circuit_breakers = CircuitBreakerPanel()
            self._register_default_circuit_breakers()
            
            self.state = RiskEngineState.ACTIVE
            logger.info("Risk Engine initialized successfully")
    
    def _register_default_circuit_breakers(self):
        """Register default circuit breakers"""
        self.circuit_breakers.register("daily_loss", cooldown_minutes=30, auto_reset=True)
        self.circuit_breakers.register("drawdown", cooldown_minutes=60, auto_reset=True)
        self.circuit_breakers.register("var_breach", cooldown_minutes=15, auto_reset=True)
        self.circuit_breakers.register("exposure", cooldown_minutes=10, auto_reset=True)
    
    # ========================================================================
    # CORE TRADE VALIDATION
    # ========================================================================
    
    def validate_trade(
        self,
        trade: TradeRequest,
        portfolio_state: PortfolioState,
    ) -> ValidationResult:
        """
        Validate a trade through ALL risk checks.
        
        THIS IS THE MAIN ENTRY POINT FOR TRADE VALIDATION.
        
        Args:
            trade: Trade request to validate
            portfolio_state: Current portfolio state
            
        Returns:
            ValidationResult with is_valid and rejection_reason
        """
        with self._lock:
            if self.state != RiskEngineState.ACTIVE:
                return ValidationResult(
                    is_valid=False,
                    rejection_reason=f"Risk Engine not active: {self.state.name}",
                    risk_level=RiskLevel.CRITICAL,
                )
            
            if not self.validator:
                return ValidationResult(
                    is_valid=False,
                    rejection_reason="Risk Validator not initialized",
                    risk_level=RiskLevel.CRITICAL,
                )
            
            # Check circuit breakers
            if not self.circuit_breakers.can_trade():
                tripped = [name for name, ok in self.circuit_breakers.check_all().items() if not ok]
                return ValidationResult(
                    is_valid=False,
                    rejection_reason=f"Circuit breakers tripped: {', '.join(tripped)}",
                    risk_level=RiskLevel.HIGH,
                )
            
            # Run full validation
            result = self.validator.validate_trade(trade, portfolio_state)
            
            # Update risk metrics after validation
            self._update_metrics(portfolio_state)
            
            return result
    
    def can_trade(self) -> bool:
        """Check if trading is currently allowed"""
        with self._lock:
            if self.state != RiskEngineState.ACTIVE:
                return False
            
            if self.kill_switch and self.kill_switch.is_active():
                return False
            
            if self.drawdown_controller and not self.drawdown_controller.can_trade():
                return False
            
            if self.circuit_breakers and not self.circuit_breakers.can_trade():
                return False
            
            return True
    
    # ========================================================================
    # POSITION SIZING
    # ========================================================================
    
    def calculate_position_size(
        self,
        symbol: str,
        entry_price: Decimal,
        portfolio_state: PortfolioState,
        stop_loss: Optional[Decimal] = None,
        atr: Optional[Decimal] = None,
        trade_history: Optional[List[Dict]] = None,
        price_history: Optional[List[Decimal]] = None,
    ) -> SizingRecommendation:
        """
        Calculate optimal position size.
        
        Args:
            symbol: Trading symbol
            entry_price: Planned entry price
            portfolio_state: Current portfolio state
            stop_loss: Stop loss price
            atr: Average True Range
            trade_history: Historical trades for Kelly/Optimal f
            price_history: Price history for volatility calculations
            
        Returns:
            SizingRecommendation with recommended quantity
        """
        with self._lock:
            if not self.position_sizer:
                return SizingRecommendation(
                    recommended_quantity=Decimal("0"),
                    recommended_value=Decimal("0"),
                    risk_amount=Decimal("0"),
                    risk_pct=Decimal("0"),
                    method=self.position_sizing_params.method,
                    confidence_score=Decimal("0"),
                    warnings=["Position sizer not initialized"],
                )
            
            return self.position_sizer.calculate_position_size(
                symbol=symbol,
                entry_price=entry_price,
                portfolio_state=portfolio_state,
                stop_loss=stop_loss,
                atr=atr,
                trade_history=trade_history,
                price_history=price_history,
            )
    
    def set_position_sizing_method(self, method: PositionSizingMethod):
        """Change the position sizing method"""
        with self._lock:
            self.position_sizing_params.method = method
            if self.position_sizer:
                self.position_sizer.params.method = method
            logger.info(f"Position sizing method changed to: {method.value}")
    
    # ========================================================================
    # PORTFOLIO STATE UPDATES
    # ========================================================================
    
    def update_portfolio_state(self, portfolio_state: PortfolioState) -> Dict[str, Any]:
        """
        Update all risk components with current portfolio state.
        
        Args:
            portfolio_state: Current portfolio state
            
        Returns:
            Dictionary with updates from all components
        """
        with self._lock:
            updates = {
                "timestamp": datetime.utcnow().isoformat(),
                "drawdown": None,
                "exposure": None,
                "correlation": None,
                "var": None,
                "alerts": [],
            }
            
            # Update drawdown controller
            if self.drawdown_controller:
                updates["drawdown"] = self.drawdown_controller.update(portfolio_state)
                updates["alerts"].extend(updates["drawdown"].get("alerts", []))
            
            # Update exposure manager
            if self.exposure_manager:
                updates["exposure"] = self.exposure_manager.update_portfolio_state(portfolio_state)
                updates["alerts"].extend(updates["exposure"].get("alerts", []))
            
            # Update correlation monitor
            if self.correlation_monitor:
                updates["correlation"] = self.correlation_monitor.update_portfolio_state(portfolio_state)
                updates["alerts"].extend(updates["correlation"].get("alerts", []))
            
            # Update VaR calculator
            if self.var_calculator:
                var_result = self.var_calculator.calculate_var(portfolio_state)
                updates["var"] = {
                    "var_pct": float(var_result.var_pct),
                    "cvar_pct": float(var_result.cvar_pct),
                }
            
            # Check for kill switch triggers
            self._check_kill_switch_triggers(portfolio_state, updates)
            
            self._last_update = datetime.utcnow()
            
            return updates
    
    def update_price(self, symbol: str, price: Decimal, timestamp: Optional[datetime] = None):
        """Update price for correlation and VaR calculations"""
        with self._lock:
            if self.correlation_monitor:
                self.correlation_monitor.update_price(symbol, price, timestamp)
    
    def update_returns(self, symbol: str, return_value: float, timestamp: Optional[datetime] = None):
        """Update returns for VaR calculations"""
        with self._lock:
            if self.var_calculator:
                self.var_calculator.update_returns(symbol, return_value, timestamp)
    
    def _update_metrics(self, portfolio_state: PortfolioState):
        """Update risk metrics (called after validation)"""
        # This is a lightweight update - full update done periodically
        pass
    
    def _check_kill_switch_triggers(self, portfolio_state: PortfolioState, updates: Dict):
        """Check if kill switch should be triggered"""
        if not self.kill_switch:
            return
        
        # Get current risk metrics
        daily_loss = Decimal("0")
        drawdown = Decimal("0")
        var_pct = Decimal("0")
        
        if self.drawdown_controller:
            daily_loss = self.drawdown_controller.state.daily_loss_pct
            drawdown = self.drawdown_controller.state.current_drawdown_pct
        
        if self.var_calculator and self.var_calculator._last_var:
            var_pct = self.var_calculator._last_var.var_pct
        
        violations = updates.get("exposure", {}).get("violations", [])
        
        # Check auto-trigger conditions
        trigger = self.kill_switch.check_auto_trigger(
            daily_loss_pct=daily_loss,
            drawdown_pct=drawdown,
            var_pct=var_pct,
            violations=violations,
        )
        
        if trigger:
            self.kill_switch.trigger(
                reason=trigger,
                details={
                    "daily_loss": float(daily_loss),
                    "drawdown": float(drawdown),
                    "var_pct": float(var_pct),
                }
            )
    
    # ========================================================================
    # RISK REPORTS
    # ========================================================================
    
    def generate_risk_report(self, portfolio_state: PortfolioState) -> RiskReport:
        """
        Generate comprehensive risk report.
        
        Args:
            portfolio_state: Current portfolio state
            
        Returns:
            RiskReport with all risk metrics
        """
        with self._lock:
            report = RiskReport(
                timestamp=datetime.utcnow(),
                portfolio_state=portfolio_state,
            )
            
            # Add drawdown state
            if self.drawdown_controller:
                report.drawdown_state = self.drawdown_controller.state
            
            # Add kill switch state
            if self.kill_switch:
                report.kill_switch_state = self.kill_switch.state
            
            # Add VaR
            if self.var_calculator:
                report.var_result = self.var_calculator.calculate_var(portfolio_state)
            
            # Add exposure data
            if self.exposure_manager:
                report.exposure_by_asset = self.exposure_manager._asset_exposure.copy()
                report.exposure_by_sector = self.exposure_manager._sector_exposure.copy()
            
            # Add correlation matrix
            if self.correlation_monitor:
                report.correlation_matrix = self.correlation_monitor._correlation_matrix
                report.portfolio_heat = self.correlation_monitor._portfolio_heat
            
            # Calculate risk level
            report.risk_level = self._calculate_overall_risk_level(report)
            
            # Generate alerts and recommendations
            report.alerts = self._generate_alerts(report)
            report.recommendations = self._generate_recommendations(report)
            
            return report
    
    def _calculate_overall_risk_level(self, report: RiskReport) -> RiskLevel:
        """Calculate overall portfolio risk level"""
        risk_levels = []
        
        if report.drawdown_state:
            if report.drawdown_state.current_drawdown_pct >= self.risk_limits.max_drawdown_pct:
                risk_levels.append(RiskLevel.HIGH)
            elif report.drawdown_state.current_drawdown_pct >= self.risk_limits.max_drawdown_pct * Decimal("0.5"):
                risk_levels.append(RiskLevel.MEDIUM)
        
        if report.var_result:
            if report.var_result.var_pct >= self.risk_limits.max_daily_var_pct:
                risk_levels.append(RiskLevel.HIGH)
            elif report.var_result.var_pct >= self.risk_limits.max_daily_var_pct * Decimal("0.8"):
                risk_levels.append(RiskLevel.MEDIUM)
        
        if report.portfolio_heat:
            if report.portfolio_heat >= self.risk_limits.max_portfolio_heat:
                risk_levels.append(RiskLevel.HIGH)
            elif report.portfolio_heat >= self.risk_limits.max_portfolio_heat * Decimal("0.8"):
                risk_levels.append(RiskLevel.MEDIUM)
        
        if RiskLevel.HIGH in risk_levels:
            return RiskLevel.HIGH
        elif RiskLevel.MEDIUM in risk_levels:
            return RiskLevel.MEDIUM
        
        return RiskLevel.LOW
    
    def _generate_alerts(self, report: RiskReport) -> List[str]:
        """Generate risk alerts"""
        alerts = []
        
        if report.drawdown_state:
            if report.drawdown_state.current_drawdown_pct >= self.risk_limits.max_drawdown_pct:
                alerts.append(f"CRITICAL: Max drawdown exceeded: {report.drawdown_state.current_drawdown_pct:.2%}")
        
        if report.var_result:
            if report.var_result.var_pct >= self.risk_limits.max_daily_var_pct:
                alerts.append(f"CRITICAL: VaR limit exceeded: {report.var_result.var_pct:.2%}")
        
        return alerts
    
    def _generate_recommendations(self, report: RiskReport) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        
        if report.drawdown_state:
            if report.drawdown_state.current_drawdown_pct >= self.risk_limits.max_drawdown_pct * Decimal("0.8"):
                recommendations.append("Consider reducing position sizes")
                recommendations.append("Review and tighten stop losses")
        
        if report.portfolio_heat:
            if report.portfolio_heat >= self.risk_limits.max_portfolio_heat * Decimal("0.8"):
                recommendations.append("Portfolio concentration is high - consider diversification")
        
        return recommendations
    
    # ========================================================================
    # KILL SWITCH CONTROLS
    # ========================================================================
    
    def trigger_kill_switch(
        self,
        reason: str,
        user_id: Optional[str] = None,
    ) -> bool:
        """Manually trigger the kill switch"""
        with self._lock:
            if not self.kill_switch:
                logger.error("Kill switch not initialized")
                return False
            
            return self.kill_switch.manual_trigger(reason, user_id)
    
    def release_kill_switch(self, user_id: Optional[str] = None, force: bool = False) -> bool:
        """Release the kill switch"""
        with self._lock:
            if not self.kill_switch:
                return False
            
            return self.kill_switch.release(user_id, force)
    
    def get_kill_switch_status(self) -> Optional[Dict[str, Any]]:
        """Get kill switch status"""
        if not self.kill_switch:
            return None
        return self.kill_switch.get_status()
    
    # ========================================================================
    # CALLBACK HANDLERS
    # ========================================================================
    
    def _on_drawdown_halt(self, reason: str):
        """Handler for drawdown trading halt"""
        logger.critical(f"Trading halted due to drawdown: {reason}")
        if self._on_trading_halt:
            self._on_trading_halt(reason)
    
    def _on_drawdown_resume(self):
        """Handler for drawdown trading resume"""
        logger.info("Trading resumed after drawdown halt")
        if self._on_trading_resume:
            self._on_trading_resume()
    
    def _on_exposure_breach(self, limit_type: str, current_value: Decimal):
        """Handler for exposure limit breach"""
        logger.warning(f"Exposure limit breached: {limit_type} = {current_value}")
        if self._on_risk_alert:
            self._on_risk_alert("exposure_breach", {
                "limit_type": limit_type,
                "current_value": float(current_value),
            })
    
    def _on_kill_switch(self, trigger: KillSwitchTrigger, reason: str):
        """Handler for kill switch activation"""
        logger.critical(f"Kill switch activated: {trigger.name} - {reason}")
        self.state = RiskEngineState.EMERGENCY
        if self._on_trading_halt:
            self._on_trading_halt(reason)
    
    def _on_kill_switch_recover(self):
        """Handler for kill switch recovery"""
        logger.info("Kill switch recovered")
        self.state = RiskEngineState.ACTIVE
        if self._on_trading_resume:
            self._on_trading_resume()
    
    def _on_validation_failure(self, trade: Optional[TradeRequest], reason: str):
        """Handler for validation failure"""
        logger.warning(f"Trade validation failed: {reason}")
    
    # ========================================================================
    # CALLBACK REGISTRATION
    # ========================================================================
    
    def on_risk_alert(self, callback: Callable[[str, Dict], None]):
        """Register risk alert callback"""
        self._on_risk_alert = callback
    
    def on_trading_halt(self, callback: Callable[[str], None]):
        """Register trading halt callback"""
        self._on_trading_halt = callback
    
    def on_trading_resume(self, callback: Callable[[], None]):
        """Register trading resume callback"""
        self._on_trading_resume = callback
    
    # ========================================================================
    # STATE MANAGEMENT
    # ========================================================================
    
    def pause(self):
        """Pause the risk engine"""
        with self._lock:
            self.state = RiskEngineState.PAUSED
            logger.info("Risk Engine paused")
    
    def resume(self):
        """Resume the risk engine"""
        with self._lock:
            if self.state == RiskEngineState.PAUSED:
                self.state = RiskEngineState.ACTIVE
                logger.info("Risk Engine resumed")
    
    def shutdown(self):
        """Shutdown the risk engine"""
        with self._lock:
            self.state = RiskEngineState.SHUTDOWN
            logger.info("Risk Engine shutdown")
    
    def reset(self):
        """Reset all risk components (use with extreme caution)"""
        with self._lock:
            logger.warning("Resetting Risk Engine - USE WITH CAUTION")
            
            if self.drawdown_controller:
                self.drawdown_controller.reset()
            
            if self.kill_switch:
                self.kill_switch.reset()
            
            if self.correlation_monitor:
                self.correlation_monitor.reset()
            
            if self.var_calculator:
                self.var_calculator.reset()
            
            if self.validator:
                self.validator.reset_stats()
            
            self.state = RiskEngineState.ACTIVE
            logger.info("Risk Engine reset complete")
    
    # ========================================================================
    # STATISTICS
    # ========================================================================
    
    def get_validation_stats(self) -> Optional[Dict[str, Any]]:
        """Get validation statistics"""
        if not self.validator:
            return None
        return self.validator.get_validation_stats()
    
    def get_status(self) -> Dict[str, Any]:
        """Get risk engine status"""
        with self._lock:
            return {
                "state": self.state.name,
                "tier": self.tier.value,
                "can_trade": self.can_trade(),
                "last_update": self._last_update.isoformat() if self._last_update else None,
                "kill_switch_active": self.kill_switch.is_active() if self.kill_switch else False,
                "circuit_breakers": self.circuit_breakers.get_status() if self.circuit_breakers else {},
                "validation_stats": self.get_validation_stats(),
            }


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_risk_engine(
    tier: str = "pro",
    position_sizing_method: str = "risk_per_trade",
    **kwargs
) -> RiskEngine:
    """
    Factory function to create a RiskEngine instance.
    
    Args:
        tier: Subscription tier (free, basic, pro, enterprise, institutional)
        position_sizing_method: Position sizing method
        **kwargs: Additional arguments for RiskEngine
        
    Returns:
        Configured RiskEngine instance
        
    Example:
        >>> engine = create_risk_engine(tier="pro")
        >>> engine.initialize()
    """
    tier_enum = ConfigTier(tier.lower())
    
    method_map = {
        "fixed_fractional": PositionSizingMethod.FIXED_FRACTIONAL,
        "kelly_criterion": PositionSizingMethod.KELLY_CRITERION,
        "volatility_based": PositionSizingMethod.VOLATILITY_BASED,
        "risk_per_trade": PositionSizingMethod.RISK_PER_TRADE,
        "optimal_f": PositionSizingMethod.OPTIMAL_F,
    }
    method = method_map.get(position_sizing_method.lower(), PositionSizingMethod.RISK_PER_TRADE)
    
    engine = RiskEngine(
        tier=tier_enum,
        position_sizing_method=method,
        **kwargs
    )
    
    return engine
