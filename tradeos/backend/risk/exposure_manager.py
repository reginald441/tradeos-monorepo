"""
Exposure Manager Module for TradeOS Risk Engine

Manages portfolio exposure across multiple dimensions:
- Total exposure calculator
- Per-asset exposure limits
- Sector/regional exposure tracking
- Leverage monitoring
- Margin usage tracking

STRICT ENFORCEMENT: No trade exceeds exposure limits.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Set, Callable, Any
from collections import defaultdict
import threading

from .models.risk_profile import (
    PortfolioState,
    Position,
    ExposureByAsset,
    ExposureBySector,
    RiskLevel,
    ValidationResult,
)
from .config.risk_limits import get_risk_limits, SubscriptionTier, ALERT_THRESHOLDS


logger = logging.getLogger(__name__)


class ExposureManager:
    """
    Portfolio exposure management system.
    Tracks and enforces exposure limits across all dimensions.
    """
    
    def __init__(
        self,
        tier: SubscriptionTier = SubscriptionTier.PRO,
        callback_on_limit_breach: Optional[Callable[[str, Decimal], None]] = None,
    ):
        self.risk_limits = get_risk_limits(tier)
        self.callback_on_limit_breach = callback_on_limit_breach
        
        # Exposure tracking
        self._asset_exposure: Dict[str, ExposureByAsset] = {}
        self._sector_exposure: Dict[str, ExposureBySector] = {}
        self._regional_exposure: Dict[str, Decimal] = defaultdict(Decimal)
        
        # Symbol metadata (sector, region, etc.)
        self._symbol_metadata: Dict[str, Dict[str, str]] = {}
        
        # Consecutive violations tracking
        self._consecutive_violations: Dict[str, int] = defaultdict(int)
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info("ExposureManager initialized")
    
    def register_symbol_metadata(
        self,
        symbol: str,
        sector: Optional[str] = None,
        region: Optional[str] = None,
        asset_class: Optional[str] = None,
        **kwargs
    ):
        """Register metadata for a symbol"""
        with self._lock:
            self._symbol_metadata[symbol] = {
                "sector": sector or "unknown",
                "region": region or "unknown",
                "asset_class": asset_class or "unknown",
                **kwargs
            }
    
    def update_portfolio_state(self, portfolio_state: PortfolioState) -> Dict[str, Any]:
        """
        Update exposure tracking with current portfolio state.
        
        Args:
            portfolio_state: Current portfolio state
            
        Returns:
            Dictionary with exposure summary and alerts
        """
        with self._lock:
            result = {
                "updated": True,
                "alerts": [],
                "violations": [],
                "exposure_summary": {},
                "risk_level": RiskLevel.LOW,
            }
            
            equity = portfolio_state.total_equity
            if equity == 0:
                logger.warning("Portfolio equity is zero, cannot calculate exposure")
                return result
            
            # Reset exposure tracking
            self._asset_exposure.clear()
            self._sector_exposure.clear()
            self._regional_exposure.clear()
            
            # Calculate exposures
            for symbol, position in portfolio_state.positions.items():
                self._update_asset_exposure(symbol, position, equity)
                self._update_sector_exposure(symbol, position, equity)
                self._update_regional_exposure(symbol, position, equity)
            
            # Check all limits
            self._check_total_exposure(result, portfolio_state)
            self._check_leverage(result, portfolio_state)
            self._check_margin_usage(result, portfolio_state)
            self._check_asset_exposure_limits(result)
            self._check_sector_exposure_limits(result)
            
            # Build exposure summary
            result["exposure_summary"] = self._build_exposure_summary(portfolio_state)
            result["risk_level"] = self._calculate_risk_level(result)
            
            return result
    
    def _update_asset_exposure(
        self,
        symbol: str,
        position: Position,
        equity: Decimal
    ):
        """Update per-asset exposure"""
        market_value = position.market_value
        exposure_pct = market_value / equity
        
        self._asset_exposure[symbol] = ExposureByAsset(
            symbol=symbol,
            market_value=market_value,
            exposure_pct=exposure_pct,
            position_count=1,
            unrealized_pnl=position.unrealized_pnl,
        )
    
    def _update_sector_exposure(
        self,
        symbol: str,
        position: Position,
        equity: Decimal
    ):
        """Update sector exposure"""
        metadata = self._symbol_metadata.get(symbol, {})
        sector = metadata.get("sector", "unknown")
        
        market_value = position.market_value
        
        if sector not in self._sector_exposure:
            self._sector_exposure[sector] = ExposureBySector(
                sector=sector,
                market_value=Decimal("0"),
                exposure_pct=Decimal("0"),
                symbols=[],
            )
        
        self._sector_exposure[sector].market_value += market_value
        self._sector_exposure[sector].symbols.append(symbol)
    
    def _update_regional_exposure(
        self,
        symbol: str,
        position: Position,
        equity: Decimal
    ):
        """Update regional exposure"""
        metadata = self._symbol_metadata.get(symbol, {})
        region = metadata.get("region", "unknown")
        
        self._regional_exposure[region] += position.market_value
    
    def _check_total_exposure(self, result: Dict[str, Any], portfolio_state: PortfolioState):
        """Check total portfolio exposure limits"""
        equity = portfolio_state.total_equity
        if equity == 0:
            return
        
        total_exposure = portfolio_state.total_exposure
        total_exposure_pct = total_exposure / equity
        
        limit = self.risk_limits.max_total_exposure_pct
        alert_threshold = limit * ALERT_THRESHOLDS["exposure"]
        
        if total_exposure_pct > limit:
            self._consecutive_violations["total_exposure"] += 1
            msg = f"Total exposure limit exceeded: {total_exposure_pct:.2%} (limit: {limit:.2%})"
            result["violations"].append({
                "type": "total_exposure",
                "current": float(total_exposure_pct),
                "limit": float(limit),
                "message": msg,
            })
            result["alerts"].append(msg)
            logger.warning(msg)
            
            if self.callback_on_limit_breach:
                self.callback_on_limit_breach("total_exposure", total_exposure_pct)
        
        elif total_exposure_pct > alert_threshold:
            msg = f"Approaching total exposure limit: {total_exposure_pct:.2%} (alert: {alert_threshold:.2%})"
            result["alerts"].append(msg)
            logger.info(msg)
        
        else:
            self._consecutive_violations["total_exposure"] = 0
    
    def _check_leverage(self, result: Dict[str, Any], portfolio_state: PortfolioState):
        """Check leverage limits"""
        leverage = portfolio_state.leverage
        limit = self.risk_limits.max_leverage
        alert_threshold = limit * ALERT_THRESHOLDS["leverage"]
        
        if leverage > limit:
            self._consecutive_violations["leverage"] += 1
            msg = f"Leverage limit exceeded: {leverage:.2f}x (limit: {limit}x)"
            result["violations"].append({
                "type": "leverage",
                "current": float(leverage),
                "limit": float(limit),
                "message": msg,
            })
            result["alerts"].append(msg)
            logger.warning(msg)
            
            if self.callback_on_limit_breach:
                self.callback_on_limit_breach("leverage", leverage)
        
        elif leverage > alert_threshold:
            msg = f"Approaching leverage limit: {leverage:.2f}x (alert: {alert_threshold:.2f}x)"
            result["alerts"].append(msg)
            logger.info(msg)
        
        else:
            self._consecutive_violations["leverage"] = 0
    
    def _check_margin_usage(self, result: Dict[str, Any], portfolio_state: PortfolioState):
        """Check margin usage limits"""
        margin_usage = portfolio_state.margin_usage_pct
        limit = self.risk_limits.max_margin_usage_pct
        alert_threshold = limit * ALERT_THRESHOLDS["margin"]
        
        if margin_usage > limit:
            self._consecutive_violations["margin"] += 1
            msg = f"Margin usage limit exceeded: {margin_usage:.2%} (limit: {limit:.2%})"
            result["violations"].append({
                "type": "margin_usage",
                "current": float(margin_usage),
                "limit": float(limit),
                "message": msg,
            })
            result["alerts"].append(msg)
            logger.warning(msg)
            
            if self.callback_on_limit_breach:
                self.callback_on_limit_breach("margin_usage", margin_usage)
        
        elif margin_usage > alert_threshold:
            msg = f"Approaching margin usage limit: {margin_usage:.2%} (alert: {alert_threshold:.2%})"
            result["alerts"].append(msg)
            logger.info(msg)
        
        else:
            self._consecutive_violations["margin"] = 0
    
    def _check_asset_exposure_limits(self, result: Dict[str, Any]):
        """Check per-asset exposure limits"""
        limit = self.risk_limits.max_single_asset_exposure_pct
        alert_threshold = limit * ALERT_THRESHOLDS["position_size"]
        
        for symbol, exposure in self._asset_exposure.items():
            if exposure.exposure_pct > limit:
                self._consecutive_violations[f"asset_{symbol}"] += 1
                msg = f"Asset exposure limit exceeded for {symbol}: {exposure.exposure_pct:.2%} (limit: {limit:.2%})"
                result["violations"].append({
                    "type": "asset_exposure",
                    "symbol": symbol,
                    "current": float(exposure.exposure_pct),
                    "limit": float(limit),
                    "message": msg,
                })
                result["alerts"].append(msg)
                logger.warning(msg)
            
            elif exposure.exposure_pct > alert_threshold:
                msg = f"Approaching exposure limit for {symbol}: {exposure.exposure_pct:.2%}"
                result["alerts"].append(msg)
                logger.info(msg)
            
            else:
                self._consecutive_violations[f"asset_{symbol}"] = 0
    
    def _check_sector_exposure_limits(self, result: Dict[str, Any]):
        """Check sector exposure limits"""
        equity = sum(exp.market_value for exp in self._asset_exposure.values())
        if equity == 0:
            return
        
        limit = self.risk_limits.max_sector_exposure_pct
        alert_threshold = limit * ALERT_THRESHOLDS["exposure"]
        
        for sector, exposure in self._sector_exposure.items():
            exposure_pct = exposure.market_value / equity
            exposure.exposure_pct = exposure_pct
            
            if exposure_pct > limit:
                self._consecutive_violations[f"sector_{sector}"] += 1
                msg = f"Sector exposure limit exceeded for {sector}: {exposure_pct:.2%} (limit: {limit:.2%})"
                result["violations"].append({
                    "type": "sector_exposure",
                    "sector": sector,
                    "current": float(exposure_pct),
                    "limit": float(limit),
                    "message": msg,
                })
                result["alerts"].append(msg)
                logger.warning(msg)
            
            elif exposure_pct > alert_threshold:
                msg = f"Approaching sector exposure limit for {sector}: {exposure_pct:.2%}"
                result["alerts"].append(msg)
                logger.info(msg)
            
            else:
                self._consecutive_violations[f"sector_{sector}"] = 0
    
    def _build_exposure_summary(self, portfolio_state: PortfolioState) -> Dict[str, Any]:
        """Build comprehensive exposure summary"""
        equity = portfolio_state.total_equity
        
        return {
            "total_exposure": float(portfolio_state.total_exposure),
            "total_exposure_pct": float(portfolio_state.total_exposure_pct) if equity > 0 else 0,
            "leverage": float(portfolio_state.leverage),
            "margin_usage_pct": float(portfolio_state.margin_usage_pct),
            "buying_power": float(portfolio_state.buying_power),
            "by_asset": {
                symbol: {
                    "market_value": float(exp.market_value),
                    "exposure_pct": float(exp.exposure_pct),
                    "unrealized_pnl": float(exp.unrealized_pnl),
                }
                for symbol, exp in self._asset_exposure.items()
            },
            "by_sector": {
                sector: {
                    "market_value": float(exp.market_value),
                    "exposure_pct": float(exp.exposure_pct),
                    "symbols": exp.symbols,
                }
                for sector, exp in self._sector_exposure.items()
            },
            "by_region": {
                region: float(value)
                for region, value in self._regional_exposure.items()
            },
            "limits": {
                "max_total_exposure_pct": float(self.risk_limits.max_total_exposure_pct),
                "max_single_asset_exposure_pct": float(self.risk_limits.max_single_asset_exposure_pct),
                "max_sector_exposure_pct": float(self.risk_limits.max_sector_exposure_pct),
                "max_leverage": float(self.risk_limits.max_leverage),
                "max_margin_usage_pct": float(self.risk_limits.max_margin_usage_pct),
            }
        }
    
    def _calculate_risk_level(self, result: Dict[str, Any]) -> RiskLevel:
        """Calculate overall risk level from violations"""
        violations = result.get("violations", [])
        
        if not violations:
            return RiskLevel.LOW
        
        # Count critical violations
        critical_types = {"total_exposure", "leverage", "margin_usage"}
        critical_count = sum(1 for v in violations if v["type"] in critical_types)
        
        if critical_count > 0:
            return RiskLevel.CRITICAL
        
        if len(violations) > 2:
            return RiskLevel.HIGH
        
        if len(violations) > 0:
            return RiskLevel.MEDIUM
        
        return RiskLevel.LOW
    
    def validate_trade(
        self,
        symbol: str,
        quantity: Decimal,
        price: Decimal,
        portfolio_state: PortfolioState,
    ) -> ValidationResult:
        """
        Validate a potential trade against exposure limits.
        
        Args:
            symbol: Trading symbol
            quantity: Trade quantity
            price: Trade price
            portfolio_state: Current portfolio state
            
        Returns:
            ValidationResult with pass/fail and details
        """
        with self._lock:
            equity = portfolio_state.total_equity
            if equity == 0:
                return ValidationResult(
                    is_valid=False,
                    rejection_reason="Portfolio equity is zero",
                    risk_level=RiskLevel.CRITICAL,
                )
            
            trade_value = abs(quantity) * price
            
            # Check 1: Total exposure
            current_exposure = portfolio_state.total_exposure
            new_exposure = current_exposure + trade_value
            new_exposure_pct = new_exposure / equity
            
            if new_exposure_pct > self.risk_limits.max_total_exposure_pct:
                return ValidationResult(
                    is_valid=False,
                    rejection_reason=f"Trade would exceed total exposure limit: {new_exposure_pct:.2%}",
                    risk_level=RiskLevel.HIGH,
                    details={
                        "limit_type": "total_exposure",
                        "current": float(current_exposure / equity),
                        "after_trade": float(new_exposure_pct),
                        "limit": float(self.risk_limits.max_total_exposure_pct),
                    }
                )
            
            # Check 2: Single asset exposure
            current_asset_exposure = self._get_asset_exposure(symbol, portfolio_state)
            new_asset_exposure = current_asset_exposure + trade_value
            new_asset_exposure_pct = new_asset_exposure / equity
            
            if new_asset_exposure_pct > self.risk_limits.max_single_asset_exposure_pct:
                return ValidationResult(
                    is_valid=False,
                    rejection_reason=f"Trade would exceed single asset exposure limit for {symbol}: {new_asset_exposure_pct:.2%}",
                    risk_level=RiskLevel.HIGH,
                    details={
                        "limit_type": "single_asset_exposure",
                        "symbol": symbol,
                        "current": float(current_asset_exposure / equity),
                        "after_trade": float(new_asset_exposure_pct),
                        "limit": float(self.risk_limits.max_single_asset_exposure_pct),
                    }
                )
            
            # Check 3: Leverage
            new_leverage = new_exposure / equity
            if new_leverage > self.risk_limits.max_leverage:
                return ValidationResult(
                    is_valid=False,
                    rejection_reason=f"Trade would exceed leverage limit: {new_leverage:.2f}x",
                    risk_level=RiskLevel.HIGH,
                    details={
                        "limit_type": "leverage",
                        "current": float(portfolio_state.leverage),
                        "after_trade": float(new_leverage),
                        "limit": float(self.risk_limits.max_leverage),
                    }
                )
            
            # Check 4: Buying power
            if trade_value > portfolio_state.buying_power:
                return ValidationResult(
                    is_valid=False,
                    rejection_reason=f"Insufficient buying power: {portfolio_state.buying_power} < {trade_value}",
                    risk_level=RiskLevel.HIGH,
                    details={
                        "limit_type": "buying_power",
                        "required": float(trade_value),
                        "available": float(portfolio_state.buying_power),
                    }
                )
            
            # Check 5: Sector exposure
            metadata = self._symbol_metadata.get(symbol, {})
            sector = metadata.get("sector", "unknown")
            
            if sector != "unknown":
                current_sector_exposure = self._get_sector_exposure(sector, portfolio_state)
                new_sector_exposure = current_sector_exposure + trade_value
                new_sector_exposure_pct = new_sector_exposure / equity
                
                if new_sector_exposure_pct > self.risk_limits.max_sector_exposure_pct:
                    return ValidationResult(
                        is_valid=False,
                        rejection_reason=f"Trade would exceed sector exposure limit for {sector}: {new_sector_exposure_pct:.2%}",
                        risk_level=RiskLevel.HIGH,
                        details={
                            "limit_type": "sector_exposure",
                            "sector": sector,
                            "current": float(current_sector_exposure / equity),
                            "after_trade": float(new_sector_exposure_pct),
                            "limit": float(self.risk_limits.max_sector_exposure_pct),
                        }
                    )
            
            return ValidationResult(
                is_valid=True,
                risk_level=RiskLevel.LOW,
                details={
                    "new_total_exposure_pct": float(new_exposure_pct),
                    "new_asset_exposure_pct": float(new_asset_exposure_pct),
                    "new_leverage": float(new_leverage),
                }
            )
    
    def _get_asset_exposure(self, symbol: str, portfolio_state: PortfolioState) -> Decimal:
        """Get current exposure for a symbol"""
        if symbol in portfolio_state.positions:
            return portfolio_state.positions[symbol].market_value
        return Decimal("0")
    
    def _get_sector_exposure(self, sector: str, portfolio_state: PortfolioState) -> Decimal:
        """Get current exposure for a sector"""
        total = Decimal("0")
        for symbol, position in portfolio_state.positions.items():
            metadata = self._symbol_metadata.get(symbol, {})
            if metadata.get("sector") == sector:
                total += position.market_value
        return total
    
    def get_available_exposure(self, portfolio_state: PortfolioState) -> Dict[str, Decimal]:
        """Get available exposure before hitting limits"""
        with self._lock:
            equity = portfolio_state.total_equity
            if equity == 0:
                return {
                    "total": Decimal("0"),
                    "per_asset": Decimal("0"),
                    "per_sector": Decimal("0"),
                }
            
            current_exposure = portfolio_state.total_exposure
            max_total = equity * self.risk_limits.max_total_exposure_pct
            
            return {
                "total": max_total - current_exposure,
                "per_asset": equity * self.risk_limits.max_single_asset_exposure_pct,
                "per_sector": equity * self.risk_limits.max_sector_exposure_pct,
            }
    
    def get_exposure_report(self) -> Dict[str, Any]:
        """Get comprehensive exposure report"""
        with self._lock:
            return {
                "by_asset": {
                    symbol: {
                        "market_value": float(exp.market_value),
                        "exposure_pct": float(exp.exposure_pct),
                        "unrealized_pnl": float(exp.unrealized_pnl),
                    }
                    for symbol, exp in self._asset_exposure.items()
                },
                "by_sector": {
                    sector: {
                        "market_value": float(exp.market_value),
                        "exposure_pct": float(exp.exposure_pct),
                        "symbols": exp.symbols,
                    }
                    for sector, exp in self._sector_exposure.items()
                },
                "by_region": dict(self._regional_exposure),
                "consecutive_violations": dict(self._consecutive_violations),
            }
