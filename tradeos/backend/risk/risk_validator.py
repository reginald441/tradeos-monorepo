"""
Risk Validator Module for TradeOS Risk Engine

Pre-trade validation that checks ALL risk rules.
NO TRADE BYPASSES THIS VALIDATION.

This is the gatekeeper - every trade must pass through here.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Any, Callable
from enum import Enum, auto
import threading

from .models.risk_profile import (
    TradeRequest,
    ValidationResult,
    PortfolioState,
    Position,
    RiskLevel,
    PositionSizingParams,
)
from .config.risk_limits import get_risk_limits, SubscriptionTier
from .position_sizing import PositionSizer
from .drawdown_control import DrawdownController
from .exposure_manager import ExposureManager
from .correlation_monitor import CorrelationMonitor
from .kill_switch import KillSwitch, KillSwitchTrigger
from .var_calculator import VaRCalculator


logger = logging.getLogger(__name__)


class ValidationRule(Enum):
    """Validation rule types"""
    KILL_SWITCH = auto()
    DRAWDOWN = auto()
    DAILY_LOSS = auto()
    POSITION_SIZE = auto()
    RISK_PER_TRADE = auto()
    TOTAL_EXPOSURE = auto()
    ASSET_EXPOSURE = auto()
    SECTOR_EXPOSURE = auto()
    LEVERAGE = auto()
    MARGIN = auto()
    BUYING_POWER = auto()
    CORRELATION = auto()
    PORTFOLIO_HEAT = auto()
    VAR = auto()
    MIN_POSITION_SIZE = auto()


class RiskValidator:
    """
    Central risk validator - THE GATEKEEPER.
    
    Every trade MUST pass through validate_trade() before execution.
    No exceptions. No bypasses. Period.
    """
    
    def __init__(
        self,
        tier: SubscriptionTier = SubscriptionTier.PRO,
        kill_switch: Optional[KillSwitch] = None,
        drawdown_controller: Optional[DrawdownController] = None,
        exposure_manager: Optional[ExposureManager] = None,
        correlation_monitor: Optional[CorrelationMonitor] = None,
        var_calculator: Optional[VaRCalculator] = None,
        position_sizer: Optional[PositionSizer] = None,
        on_validation_failure: Optional[Callable[[TradeRequest, str], None]] = None,
    ):
        self.risk_limits = get_risk_limits(tier)
        
        # Risk components
        self.kill_switch = kill_switch
        self.drawdown_controller = drawdown_controller
        self.exposure_manager = exposure_manager
        self.correlation_monitor = correlation_monitor
        self.var_calculator = var_calculator
        self.position_sizer = position_sizer
        
        # Callback for validation failures
        self.on_validation_failure = on_validation_failure
        
        # Validation statistics
        self._validation_stats = {
            "total_validations": 0,
            "passed": 0,
            "rejected": 0,
            "rejection_reasons": {},
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info("RiskValidator initialized - GATEKEEPER ACTIVE")
    
    def validate_trade(
        self,
        trade: TradeRequest,
        portfolio_state: PortfolioState,
        skip_rules: Optional[List[ValidationRule]] = None,
    ) -> ValidationResult:
        """
        Validate a trade against ALL risk rules.
        
        THIS IS THE MAIN GATE. NO TRADE EXECUTES WITHOUT PASSING HERE.
        
        Args:
            trade: Trade request to validate
            portfolio_state: Current portfolio state
            skip_rules: Optional list of rules to skip (use with extreme caution)
            
        Returns:
            ValidationResult with is_valid and rejection_reason
            
        Example:
            >>> validator = RiskValidator()
            >>> result = validator.validate_trade(trade_request, portfolio)
            >>> if result.is_valid:
            ...     execute_trade(trade)
            ... else:
            ...     log_rejection(result.rejection_reason)
        """
        with self._lock:
            self._validation_stats["total_validations"] += 1
            
            skip_rules = skip_rules or []
            failure_details = []
            warnings = []
            
            # =========================================================================
            # CRITICAL CHECKS - These can NEVER be skipped
            # =========================================================================
            
            # Check 1: Kill Switch - IMMEDIATE REJECTION if active
            if self.kill_switch and self.kill_switch.is_active():
                reason = f"KILL SWITCH ACTIVE: {self.kill_switch.state.reason}"
                self._record_rejection("kill_switch")
                return self._create_failure_result(reason, RiskLevel.EMERGENCY)
            
            # Check 2: Drawdown Controller - Trading halted
            if self.drawdown_controller and not self.drawdown_controller.can_trade():
                state = self.drawdown_controller.state
                reason = f"TRADING HALTED: {state.halt_reason}"
                self._record_rejection("trading_halted")
                return self._create_failure_result(reason, RiskLevel.CRITICAL)
            
            # =========================================================================
            # POSITION & RISK CHECKS
            # =========================================================================
            
            # Check 3: Position size limits
            if ValidationRule.POSITION_SIZE not in skip_rules:
                result = self._validate_position_size(trade, portfolio_state)
                if not result.is_valid:
                    return result
                warnings.extend(result.warnings)
            
            # Check 4: Risk per trade
            if ValidationRule.RISK_PER_TRADE not in skip_rules:
                result = self._validate_risk_per_trade(trade, portfolio_state)
                if not result.is_valid:
                    return result
                warnings.extend(result.warnings)
            
            # Check 5: Minimum position size
            if ValidationRule.MIN_POSITION_SIZE not in skip_rules:
                result = self._validate_min_position_size(trade)
                if not result.is_valid:
                    return result
            
            # =========================================================================
            # EXPOSURE CHECKS
            # =========================================================================
            
            # Check 6: Total exposure
            if ValidationRule.TOTAL_EXPOSURE not in skip_rules:
                result = self._validate_total_exposure(trade, portfolio_state)
                if not result.is_valid:
                    return result
            
            # Check 7: Single asset exposure
            if ValidationRule.ASSET_EXPOSURE not in skip_rules:
                result = self._validate_asset_exposure(trade, portfolio_state)
                if not result.is_valid:
                    return result
            
            # Check 8: Sector exposure
            if ValidationRule.SECTOR_EXPOSURE not in skip_rules:
                result = self._validate_sector_exposure(trade, portfolio_state)
                if not result.is_valid:
                    return result
            
            # Check 9: Leverage
            if ValidationRule.LEVERAGE not in skip_rules:
                result = self._validate_leverage(trade, portfolio_state)
                if not result.is_valid:
                    return result
            
            # Check 10: Margin usage
            if ValidationRule.MARGIN not in skip_rules:
                result = self._validate_margin(trade, portfolio_state)
                if not result.is_valid:
                    return result
            
            # Check 11: Buying power
            if ValidationRule.BUYING_POWER not in skip_rules:
                result = self._validate_buying_power(trade, portfolio_state)
                if not result.is_valid:
                    return result
            
            # =========================================================================
            # DRAWDOWN & LOSS CHECKS
            # =========================================================================
            
            # Check 12: Current drawdown
            if ValidationRule.DRAWDOWN not in skip_rules and self.drawdown_controller:
                state = self.drawdown_controller.state
                if state.current_drawdown_pct >= self.risk_limits.max_drawdown_pct:
                    reason = f"Max drawdown exceeded: {state.current_drawdown_pct:.2%}"
                    self._record_rejection("drawdown")
                    return self._create_failure_result(reason, RiskLevel.HIGH)
            
            # Check 13: Daily loss
            if ValidationRule.DAILY_LOSS not in skip_rules and self.drawdown_controller:
                state = self.drawdown_controller.state
                if state.daily_loss_pct >= self.risk_limits.max_daily_loss_pct:
                    reason = f"Daily loss limit exceeded: {state.daily_loss_pct:.2%}"
                    self._record_rejection("daily_loss")
                    return self._create_failure_result(reason, RiskLevel.HIGH)
            
            # =========================================================================
            # CORRELATION CHECKS
            # =========================================================================
            
            # Check 14: Correlation limits
            if ValidationRule.CORRELATION not in skip_rules and self.correlation_monitor:
                result = self.correlation_monitor.validate_trade(
                    trade.symbol, trade.quantity, portfolio_state
                )
                if not result.is_valid:
                    self._record_rejection("correlation")
                    return result
                warnings.extend(result.warnings)
            
            # Check 15: Portfolio heat
            if ValidationRule.PORTFOLIO_HEAT not in skip_rules and self.correlation_monitor:
                heat = self.correlation_monitor._portfolio_heat
                if heat > self.risk_limits.max_portfolio_heat:
                    reason = f"Portfolio heat limit exceeded: {heat:.2%}"
                    self._record_rejection("portfolio_heat")
                    return self._create_failure_result(reason, RiskLevel.HIGH)
            
            # =========================================================================
            # VaR CHECKS
            # =========================================================================
            
            # Check 16: VaR limits
            if ValidationRule.VAR not in skip_rules and self.var_calculator:
                result = self.var_calculator.validate_trade(
                    trade.symbol, trade.quantity, trade.price or Decimal("0"), portfolio_state
                )
                if not result.is_valid:
                    self._record_rejection("var")
                    return result
                warnings.extend(result.warnings)
            
            # =========================================================================
            # ALL CHECKS PASSED
            # =========================================================================
            
            self._validation_stats["passed"] += 1
            
            # Determine risk level from warnings
            risk_level = RiskLevel.LOW
            if warnings:
                risk_level = RiskLevel.MEDIUM
            
            logger.info(f"Trade validated: {trade.symbol} {trade.quantity} @ {trade.price}")
            
            return ValidationResult(
                is_valid=True,
                risk_level=risk_level,
                warnings=warnings,
                details={
                    "symbol": trade.symbol,
                    "quantity": str(trade.quantity),
                    "price": str(trade.price) if trade.price else None,
                    "validation_time": datetime.utcnow().isoformat(),
                }
            )
    
    def _validate_position_size(
        self,
        trade: TradeRequest,
        portfolio_state: PortfolioState,
    ) -> ValidationResult:
        """Validate position size limits"""
        equity = portfolio_state.total_equity
        if equity == 0:
            return self._create_failure_result("Portfolio equity is zero", RiskLevel.CRITICAL)
        
        trade_value = trade.quantity * (trade.price or Decimal("0"))
        position_pct = trade_value / equity
        
        limit = self.risk_limits.max_position_size_pct
        
        if position_pct > limit:
            reason = f"Position size {position_pct:.2%} exceeds limit {limit:.2%}"
            self._record_rejection("position_size")
            return self._create_failure_result(reason, RiskLevel.HIGH)
        
        return ValidationResult(is_valid=True, risk_level=RiskLevel.LOW)
    
    def _validate_risk_per_trade(
        self,
        trade: TradeRequest,
        portfolio_state: PortfolioState,
    ) -> ValidationResult:
        """Validate risk per trade limits"""
        equity = portfolio_state.total_equity
        if equity == 0:
            return self._create_failure_result("Portfolio equity is zero", RiskLevel.CRITICAL)
        
        # Calculate risk amount
        if trade.stop_loss and trade.price:
            risk_per_unit = abs(trade.price - trade.stop_loss)
            risk_amount = trade.quantity * risk_per_unit
            risk_pct = risk_amount / equity
        else:
            # No stop loss - assume full position at risk
            trade_value = trade.quantity * (trade.price or Decimal("0"))
            risk_pct = trade_value / equity
        
        limit = self.risk_limits.max_risk_per_trade_pct
        
        if risk_pct > limit:
            reason = f"Risk per trade {risk_pct:.2%} exceeds limit {limit:.2%}"
            self._record_rejection("risk_per_trade")
            return self._create_failure_result(reason, RiskLevel.HIGH)
        
        return ValidationResult(is_valid=True, risk_level=RiskLevel.LOW)
    
    def _validate_min_position_size(self, trade: TradeRequest) -> ValidationResult:
        """Validate minimum position size"""
        trade_value = trade.quantity * (trade.price or Decimal("0"))
        
        if trade_value < self.risk_limits.min_position_size:
            reason = f"Position value {trade_value} below minimum {self.risk_limits.min_position_size}"
            self._record_rejection("min_position_size")
            return self._create_failure_result(reason, RiskLevel.LOW)
        
        return ValidationResult(is_valid=True, risk_level=RiskLevel.LOW)
    
    def _validate_total_exposure(
        self,
        trade: TradeRequest,
        portfolio_state: PortfolioState,
    ) -> ValidationResult:
        """Validate total exposure limits"""
        equity = portfolio_state.total_equity
        if equity == 0:
            return self._create_failure_result("Portfolio equity is zero", RiskLevel.CRITICAL)
        
        current_exposure = portfolio_state.total_exposure
        trade_value = trade.quantity * (trade.price or Decimal("0"))
        new_exposure = current_exposure + trade_value
        new_exposure_pct = new_exposure / equity
        
        limit = self.risk_limits.max_total_exposure_pct
        
        if new_exposure_pct > limit:
            reason = f"Total exposure {new_exposure_pct:.2%} would exceed limit {limit:.2%}"
            self._record_rejection("total_exposure")
            return self._create_failure_result(reason, RiskLevel.HIGH)
        
        return ValidationResult(is_valid=True, risk_level=RiskLevel.LOW)
    
    def _validate_asset_exposure(
        self,
        trade: TradeRequest,
        portfolio_state: PortfolioState,
    ) -> ValidationResult:
        """Validate single asset exposure limits"""
        equity = portfolio_state.total_equity
        if equity == 0:
            return self._create_failure_result("Portfolio equity is zero", RiskLevel.CRITICAL)
        
        # Get current exposure for this asset
        current_exposure = Decimal("0")
        if trade.symbol in portfolio_state.positions:
            current_exposure = portfolio_state.positions[trade.symbol].market_value
        
        trade_value = trade.quantity * (trade.price or Decimal("0"))
        new_exposure = current_exposure + trade_value
        new_exposure_pct = new_exposure / equity
        
        limit = self.risk_limits.max_single_asset_exposure_pct
        
        if new_exposure_pct > limit:
            reason = f"Asset exposure for {trade.symbol} {new_exposure_pct:.2%} would exceed limit {limit:.2%}"
            self._record_rejection("asset_exposure")
            return self._create_failure_result(reason, RiskLevel.HIGH)
        
        return ValidationResult(is_valid=True, risk_level=RiskLevel.LOW)
    
    def _validate_sector_exposure(
        self,
        trade: TradeRequest,
        portfolio_state: PortfolioState,
    ) -> ValidationResult:
        """Validate sector exposure limits"""
        # This requires exposure manager with sector data
        if not self.exposure_manager:
            return ValidationResult(is_valid=True, risk_level=RiskLevel.LOW)
        
        # Get sector for symbol
        metadata = self.exposure_manager._symbol_metadata.get(trade.symbol, {})
        sector = metadata.get("sector")
        
        if not sector:
            return ValidationResult(is_valid=True, risk_level=RiskLevel.LOW)
        
        # Validate through exposure manager
        return self.exposure_manager.validate_trade(
            trade.symbol, trade.quantity, trade.price or Decimal("0"), portfolio_state
        )
    
    def _validate_leverage(
        self,
        trade: TradeRequest,
        portfolio_state: PortfolioState,
    ) -> ValidationResult:
        """Validate leverage limits"""
        equity = portfolio_state.total_equity
        if equity == 0:
            return self._create_failure_result("Portfolio equity is zero", RiskLevel.CRITICAL)
        
        current_exposure = portfolio_state.total_exposure
        trade_value = trade.quantity * (trade.price or Decimal("0"))
        new_exposure = current_exposure + trade_value
        new_leverage = new_exposure / equity
        
        limit = self.risk_limits.max_leverage
        
        if new_leverage > limit:
            reason = f"Leverage {new_leverage:.2f}x would exceed limit {limit}x"
            self._record_rejection("leverage")
            return self._create_failure_result(reason, RiskLevel.HIGH)
        
        return ValidationResult(is_valid=True, risk_level=RiskLevel.LOW)
    
    def _validate_margin(
        self,
        trade: TradeRequest,
        portfolio_state: PortfolioState,
    ) -> ValidationResult:
        """Validate margin usage limits"""
        # Estimate new margin usage
        trade_value = trade.quantity * (trade.price or Decimal("0"))
        
        # Assume 50% margin requirement for stocks
        margin_required = trade_value * Decimal("0.5")
        new_margin_used = portfolio_state.margin_used + margin_required
        
        total_margin = new_margin_used + portfolio_state.margin_available
        if total_margin > 0:
            new_margin_usage = new_margin_used / total_margin
        else:
            new_margin_usage = Decimal("0")
        
        limit = self.risk_limits.max_margin_usage_pct
        
        if new_margin_usage > limit:
            reason = f"Margin usage {new_margin_usage:.2%} would exceed limit {limit:.2%}"
            self._record_rejection("margin")
            return self._create_failure_result(reason, RiskLevel.HIGH)
        
        return ValidationResult(is_valid=True, risk_level=RiskLevel.LOW)
    
    def _validate_buying_power(
        self,
        trade: TradeRequest,
        portfolio_state: PortfolioState,
    ) -> ValidationResult:
        """Validate sufficient buying power"""
        trade_value = trade.quantity * (trade.price or Decimal("0"))
        
        if trade_value > portfolio_state.buying_power:
            reason = f"Insufficient buying power: {portfolio_state.buying_power} < {trade_value}"
            self._record_rejection("buying_power")
            return self._create_failure_result(reason, RiskLevel.HIGH)
        
        return ValidationResult(is_valid=True, risk_level=RiskLevel.LOW)
    
    def _create_failure_result(
        self,
        reason: str,
        risk_level: RiskLevel,
        details: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """Create a failure validation result"""
        self._validation_stats["rejected"] += 1
        
        if self.on_validation_failure:
            try:
                self.on_validation_failure(None, reason)
            except Exception as e:
                logger.error(f"Error in validation failure callback: {e}")
        
        logger.warning(f"Trade validation FAILED: {reason}")
        
        return ValidationResult(
            is_valid=False,
            rejection_reason=reason,
            risk_level=risk_level,
            details=details or {},
        )
    
    def _record_rejection(self, reason_type: str):
        """Record rejection reason for statistics"""
        current = self._validation_stats["rejection_reasons"].get(reason_type, 0)
        self._validation_stats["rejection_reasons"][reason_type] = current + 1
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        with self._lock:
            total = self._validation_stats["total_validations"]
            passed = self._validation_stats["passed"]
            rejected = self._validation_stats["rejected"]
            
            return {
                "total_validations": total,
                "passed": passed,
                "rejected": rejected,
                "pass_rate": passed / total if total > 0 else 0,
                "rejection_reasons": dict(self._validation_stats["rejection_reasons"]),
            }
    
    def reset_stats(self):
        """Reset validation statistics"""
        with self._lock:
            self._validation_stats = {
                "total_validations": 0,
                "passed": 0,
                "rejected": 0,
                "rejection_reasons": {},
            }


# ============================================================================
# CONVENIENCE FUNCTION - THE MAIN ENTRY POINT
# ============================================================================

def validate_trade(
    trade: TradeRequest,
    portfolio_state: PortfolioState,
    validator: RiskValidator,
) -> ValidationResult:
    """
    Convenience function to validate a trade.
    
    This is the RECOMMENDED way to validate trades.
    
    Args:
        trade: Trade request to validate
        portfolio_state: Current portfolio state
        validator: RiskValidator instance
        
    Returns:
        ValidationResult with is_valid and rejection_reason
        
    Example:
        >>> from tradeos.backend.risk.risk_validator import validate_trade
        >>> result = validate_trade(trade, portfolio, validator)
        >>> if result.is_valid:
        ...     execute_trade(trade)
    """
    return validator.validate_trade(trade, portfolio_state)
