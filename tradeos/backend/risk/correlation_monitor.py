"""
Correlation Monitor Module for TradeOS Risk Engine

Tracks cross-asset correlations and portfolio heat:
- Real-time correlation matrix
- Correlation-based position adjustment
- Portfolio heat calculator
- Concentration risk detection

STRICT ENFORCEMENT: High correlation positions are flagged and adjusted.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Set, Any
from collections import deque, defaultdict
import threading
import numpy as np
from scipy import stats

from .models.risk_profile import (
    PortfolioState,
    Position,
    CorrelationMatrix,
    RiskLevel,
    ValidationResult,
)
from .config.risk_limits import get_risk_limits, SubscriptionTier, ALERT_THRESHOLDS


logger = logging.getLogger(__name__)


class CorrelationMonitor:
    """
    Cross-asset correlation monitoring system.
    Calculates portfolio heat and detects concentration risk.
    """
    
    def __init__(
        self,
        tier: SubscriptionTier = SubscriptionTier.PRO,
        correlation_lookback: int = 30,
        correlation_min_periods: int = 10,
        update_interval_minutes: int = 5,
    ):
        self.risk_limits = get_risk_limits(tier)
        self.correlation_lookback = correlation_lookback
        self.correlation_min_periods = correlation_min_periods
        self.update_interval = timedelta(minutes=update_interval_minutes)
        
        # Price history for correlation calculation
        self._price_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=correlation_lookback * 2)
        )
        
        # Returns history
        self._returns_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=correlation_lookback)
        )
        
        # Correlation matrix cache
        self._correlation_matrix: Optional[CorrelationMatrix] = None
        self._last_update: Optional[datetime] = None
        
        # Portfolio heat tracking
        self._portfolio_heat: Decimal = Decimal("0")
        self._heat_breakdown: Dict[str, Decimal] = {}
        
        # High correlation pairs
        self._high_correlation_pairs: Set[Tuple[str, str]] = set()
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info("CorrelationMonitor initialized")
    
    def update_price(self, symbol: str, price: Decimal, timestamp: Optional[datetime] = None):
        """
        Update price for a symbol.
        
        Args:
            symbol: Trading symbol
            price: Current price
            timestamp: Price timestamp (defaults to now)
        """
        with self._lock:
            timestamp = timestamp or datetime.utcnow()
            
            # Store price
            self._price_history[symbol].append({
                "timestamp": timestamp,
                "price": float(price),
            })
            
            # Calculate return if we have previous price
            if len(self._price_history[symbol]) >= 2:
                prev_price = self._price_history[symbol][-2]["price"]
                curr_price = float(price)
                if prev_price > 0:
                    ret = (curr_price - prev_price) / prev_price
                    self._returns_history[symbol].append(ret)
    
    def update_portfolio_state(self, portfolio_state: PortfolioState) -> Dict[str, Any]:
        """
        Update correlation tracking with current portfolio state.
        
        Args:
            portfolio_state: Current portfolio state
            
        Returns:
            Dictionary with correlation summary and alerts
        """
        with self._lock:
            result = {
                "updated": True,
                "alerts": [],
                "correlation_matrix": None,
                "portfolio_heat": 0.0,
                "risk_level": RiskLevel.LOW,
            }
            
            current_time = datetime.utcnow()
            
            # Check if we need to update correlation matrix
            should_update = (
                self._last_update is None or
                (current_time - self._last_update) >= self.update_interval
            )
            
            if should_update:
                self._update_correlation_matrix(portfolio_state)
                self._last_update = current_time
            
            # Calculate portfolio heat
            self._calculate_portfolio_heat(portfolio_state)
            
            # Check correlation limits
            self._check_correlation_limits(result)
            
            # Check portfolio heat limits
            self._check_portfolio_heat_limits(result)
            
            # Build result
            if self._correlation_matrix:
                result["correlation_matrix"] = {
                    "symbols": self._correlation_matrix.symbols,
                    "matrix": self._correlation_matrix.correlation_matrix.tolist(),
                    "timestamp": self._correlation_matrix.timestamp.isoformat(),
                }
            
            result["portfolio_heat"] = float(self._portfolio_heat)
            result["heat_breakdown"] = {
                k: float(v) for k, v in self._heat_breakdown.items()
            }
            result["high_correlation_pairs"] = list(self._high_correlation_pairs)
            result["risk_level"] = self._calculate_risk_level(result)
            
            return result
    
    def _update_correlation_matrix(self, portfolio_state: PortfolioState):
        """Update the correlation matrix from returns history"""
        symbols = list(portfolio_state.positions.keys())
        
        if len(symbols) < 2:
            self._correlation_matrix = None
            return
        
        # Build returns matrix
        returns_data = []
        valid_symbols = []
        
        for symbol in symbols:
            if len(self._returns_history[symbol]) >= self.correlation_min_periods:
                returns = list(self._returns_history[symbol])
                returns_data.append(returns)
                valid_symbols.append(symbol)
        
        if len(valid_symbols) < 2:
            self._correlation_matrix = None
            return
        
        # Align returns to same length
        min_len = min(len(r) for r in returns_data)
        returns_matrix = np.array([r[-min_len:] for r in returns_data])
        
        # Calculate correlation matrix
        correlation_matrix = np.corrcoef(returns_matrix)
        
        # Handle NaN values
        correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
        
        self._correlation_matrix = CorrelationMatrix(
            symbols=valid_symbols,
            correlation_matrix=correlation_matrix,
            timestamp=datetime.utcnow(),
        )
        
        # Update high correlation pairs
        self._update_high_correlation_pairs()
    
    def _update_high_correlation_pairs(self):
        """Update set of high correlation pairs"""
        self._high_correlation_pairs.clear()
        
        if self._correlation_matrix is None:
            return
        
        threshold = float(self.risk_limits.max_correlation_exposure)
        symbols = self._correlation_matrix.symbols
        matrix = self._correlation_matrix.correlation_matrix
        
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                corr = abs(matrix[i, j])
                if corr >= threshold:
                    self._high_correlation_pairs.add((symbols[i], symbols[j]))
    
    def _calculate_portfolio_heat(self, portfolio_state: PortfolioState):
        """
        Calculate portfolio heat - a measure of concentration risk.
        Heat = sum of (weight_i × weight_j × correlation_ij) for all pairs
        """
        equity = portfolio_state.total_equity
        if equity == 0 or self._correlation_matrix is None:
            self._portfolio_heat = Decimal("0")
            return
        
        symbols = self._correlation_matrix.symbols
        matrix = self._correlation_matrix.correlation_matrix
        
        # Calculate weights
        weights = {}
        for symbol in symbols:
            if symbol in portfolio_state.positions:
                position_value = portfolio_state.positions[symbol].market_value
                weights[symbol] = float(position_value / equity)
            else:
                weights[symbol] = 0.0
        
        # Calculate portfolio heat
        heat = 0.0
        heat_breakdown = defaultdict(float)
        
        for i, sym_i in enumerate(symbols):
            for j, sym_j in enumerate(symbols):
                if i != j:
                    pair_heat = weights[sym_i] * weights[sym_j] * abs(matrix[i, j])
                    heat += pair_heat
                    heat_breakdown[sym_i] += Decimal(str(pair_heat))
        
        self._portfolio_heat = Decimal(str(heat))
        self._heat_breakdown = dict(heat_breakdown)
    
    def _check_correlation_limits(self, result: Dict[str, Any]):
        """Check correlation exposure limits"""
        if self._correlation_matrix is None:
            return
        
        threshold = float(self.risk_limits.max_correlation_exposure)
        alert_threshold = threshold * float(ALERT_THRESHOLDS["correlation"])
        
        symbols = self._correlation_matrix.symbols
        matrix = self._correlation_matrix.correlation_matrix
        
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                corr = abs(matrix[i, j])
                
                if corr >= threshold:
                    msg = f"High correlation detected: {symbols[i]} - {symbols[j]} ({corr:.2%})"
                    result["alerts"].append({
                        "type": "high_correlation",
                        "symbols": [symbols[i], symbols[j]],
                        "correlation": corr,
                        "message": msg,
                    })
                    logger.warning(msg)
                
                elif corr >= alert_threshold:
                    msg = f"Elevated correlation: {symbols[i]} - {symbols[j]} ({corr:.2%})"
                    result["alerts"].append({
                        "type": "elevated_correlation",
                        "symbols": [symbols[i], symbols[j]],
                        "correlation": corr,
                        "message": msg,
                    })
                    logger.info(msg)
    
    def _check_portfolio_heat_limits(self, result: Dict[str, Any]):
        """Check portfolio heat limits"""
        heat = self._portfolio_heat
        limit = self.risk_limits.max_portfolio_heat
        alert_threshold = limit * ALERT_THRESHOLDS["correlation"]
        
        if heat > limit:
            msg = f"Portfolio heat limit exceeded: {heat:.2%} (limit: {limit:.2%})"
            result["alerts"].append({
                "type": "portfolio_heat",
                "current": float(heat),
                "limit": float(limit),
                "message": msg,
            })
            logger.warning(msg)
        
        elif heat > alert_threshold:
            msg = f"Approaching portfolio heat limit: {heat:.2%} (alert: {alert_threshold:.2%})"
            result["alerts"].append({
                "type": "portfolio_heat_warning",
                "current": float(heat),
                "threshold": float(alert_threshold),
                "message": msg,
            })
            logger.info(msg)
    
    def _calculate_risk_level(self, result: Dict[str, Any]) -> RiskLevel:
        """Calculate overall risk level from correlation metrics"""
        alerts = result.get("alerts", [])
        
        if not alerts:
            return RiskLevel.LOW
        
        high_corr_count = sum(1 for a in alerts if a.get("type") == "high_correlation")
        heat_exceeded = any(a.get("type") == "portfolio_heat" for a in alerts)
        
        if heat_exceeded or high_corr_count >= 3:
            return RiskLevel.HIGH
        
        if high_corr_count >= 1:
            return RiskLevel.MEDIUM
        
        return RiskLevel.LOW
    
    def get_correlation(self, symbol1: str, symbol2: str) -> Optional[float]:
        """Get correlation between two symbols"""
        with self._lock:
            if self._correlation_matrix is None:
                return None
            return self._correlation_matrix.get_correlation(symbol1, symbol2)
    
    def get_correlation_adjusted_position_size(
        self,
        symbol: str,
        base_size: Decimal,
        portfolio_state: PortfolioState,
    ) -> Decimal:
        """
        Adjust position size based on correlations with existing positions.
        Reduces size for highly correlated assets.
        
        Args:
            symbol: Symbol to trade
            base_size: Base position size
            portfolio_state: Current portfolio state
            
        Returns:
            Adjusted position size
        """
        with self._lock:
            if self._correlation_matrix is None:
                return base_size
            
            if symbol not in self._correlation_matrix.symbols:
                return base_size
            
            # Find max correlation with existing positions
            max_corr = 0.0
            
            for existing_symbol in portfolio_state.positions.keys():
                if existing_symbol == symbol:
                    continue
                
                corr = self.get_correlation(symbol, existing_symbol)
                if corr is not None:
                    max_corr = max(max_corr, abs(corr))
            
            # Adjust size based on correlation
            # Higher correlation = smaller position
            if max_corr >= float(self.risk_limits.max_correlation_exposure):
                # Highly correlated - reduce to 50%
                adjustment = Decimal("0.50")
            elif max_corr >= 0.5:
                # Moderately correlated - reduce proportionally
                adjustment = Decimal("1.0") - Decimal(str(max_corr))
            else:
                # Low correlation - no adjustment
                adjustment = Decimal("1.0")
            
            adjusted_size = base_size * adjustment
            
            logger.debug(
                f"Correlation adjustment for {symbol}: "
                f"base={base_size}, adjusted={adjusted_size}, "
                f"max_corr={max_corr:.2%}, adjustment={adjustment}"
            )
            
            return adjusted_size
    
    def validate_trade(
        self,
        symbol: str,
        quantity: Decimal,
        portfolio_state: PortfolioState,
    ) -> ValidationResult:
        """
        Validate a potential trade against correlation limits.
        
        Args:
            symbol: Trading symbol
            quantity: Trade quantity
            portfolio_state: Current portfolio state
            
        Returns:
            ValidationResult with pass/fail and details
        """
        with self._lock:
            if self._correlation_matrix is None:
                return ValidationResult(
                    is_valid=True,
                    risk_level=RiskLevel.LOW,
                    details={"message": "No correlation data available"},
                )
            
            # Check if symbol has high correlation with existing positions
            high_corr_symbols = []
            
            for existing_symbol in portfolio_state.positions.keys():
                if existing_symbol == symbol:
                    continue
                
                corr = self.get_correlation(symbol, existing_symbol)
                if corr is not None and corr >= float(self.risk_limits.max_correlation_exposure):
                    high_corr_symbols.append({
                        "symbol": existing_symbol,
                        "correlation": corr,
                    })
            
            if len(high_corr_symbols) >= 3:
                # Too many highly correlated positions
                return ValidationResult(
                    is_valid=False,
                    rejection_reason=f"Trade rejected: {symbol} has high correlation with {len(high_corr_symbols)} existing positions",
                    risk_level=RiskLevel.HIGH,
                    warnings=[
                        f"High correlation with: {', '.join(s['symbol'] for s in high_corr_symbols[:3])}"
                    ],
                    details={
                        "high_correlations": high_corr_symbols,
                    }
                )
            
            # Check portfolio heat after trade
            equity = portfolio_state.total_equity
            if equity > 0:
                # Estimate new heat
                new_position_value = abs(quantity) * portfolio_state.positions.get(
                    symbol, Position(symbol, Decimal("0"), Decimal("0"), Decimal("0"), "long")
                ).current_price
                
                # Simple heat estimation
                estimated_heat_increase = Decimal("0")
                for existing_symbol, position in portfolio_state.positions.items():
                    if existing_symbol == symbol:
                        continue
                    
                    corr = self.get_correlation(symbol, existing_symbol)
                    if corr is not None:
                        existing_weight = float(position.market_value / equity)
                        new_weight = float(new_position_value / equity)
                        estimated_heat_increase += Decimal(str(existing_weight * new_weight * corr))
                
                estimated_heat = self._portfolio_heat + estimated_heat_increase
                
                if estimated_heat > self.risk_limits.max_portfolio_heat:
                    return ValidationResult(
                        is_valid=False,
                        rejection_reason=f"Trade would exceed portfolio heat limit: {estimated_heat:.2%}",
                        risk_level=RiskLevel.HIGH,
                        details={
                            "current_heat": float(self._portfolio_heat),
                            "estimated_heat": float(estimated_heat),
                            "limit": float(self.risk_limits.max_portfolio_heat),
                        }
                    )
            
            warnings = []
            if high_corr_symbols:
                warnings.append(
                    f"Note: {symbol} has elevated correlation with: "
                    f"{', '.join(s['symbol'] for s in high_corr_symbols[:2])}"
                )
            
            return ValidationResult(
                is_valid=True,
                risk_level=RiskLevel.MEDIUM if high_corr_symbols else RiskLevel.LOW,
                warnings=warnings,
                details={
                    "high_correlations": high_corr_symbols,
                }
            )
    
    def get_correlation_report(self) -> Dict[str, Any]:
        """Get comprehensive correlation report"""
        with self._lock:
            return {
                "correlation_matrix": {
                    "symbols": self._correlation_matrix.symbols if self._correlation_matrix else [],
                    "matrix": self._correlation_matrix.correlation_matrix.tolist() if self._correlation_matrix else [],
                    "last_updated": self._last_update.isoformat() if self._last_update else None,
                },
                "portfolio_heat": float(self._portfolio_heat),
                "heat_breakdown": {k: float(v) for k, v in self._heat_breakdown.items()},
                "high_correlation_pairs": list(self._high_correlation_pairs),
                "limits": {
                    "max_correlation_exposure": float(self.risk_limits.max_correlation_exposure),
                    "max_portfolio_heat": float(self.risk_limits.max_portfolio_heat),
                },
            }
    
    def reset(self):
        """Reset correlation data"""
        with self._lock:
            self._price_history.clear()
            self._returns_history.clear()
            self._correlation_matrix = None
            self._last_update = None
            self._portfolio_heat = Decimal("0")
            self._heat_breakdown.clear()
            self._high_correlation_pairs.clear()
            logger.info("CorrelationMonitor reset")
