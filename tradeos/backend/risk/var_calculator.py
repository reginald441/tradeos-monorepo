"""
Value at Risk (VaR) Calculator for TradeOS Risk Engine

Implements multiple VaR calculation methods:
- Historical VaR
- Parametric VaR
- Monte Carlo VaR
- CVaR (Conditional VaR / Expected Shortfall)

STRICT ENFORCEMENT: Trades rejected if VaR limits exceeded.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import deque
import threading
import numpy as np
from scipy import stats

from .models.risk_profile import (
    VaRResult,
    VaRMethod,
    PortfolioState,
    Position,
    RiskLevel,
    ValidationResult,
)
from .config.risk_limits import get_risk_limits, SubscriptionTier


logger = logging.getLogger(__name__)


class VaRCalculator:
    """
    Value at Risk calculator supporting multiple methods.
    """
    
    def __init__(
        self,
        tier: SubscriptionTier = SubscriptionTier.PRO,
        default_method: VaRMethod = VaRMethod.HISTORICAL,
        lookback_days: int = 252,
        holding_period_days: int = 1,
    ):
        self.risk_limits = get_risk_limits(tier)
        self.default_method = default_method
        self.lookback_days = lookback_days
        self.holding_period_days = holding_period_days
        
        # Returns history
        self._returns_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=lookback_days)
        )
        
        # Portfolio returns history
        self._portfolio_returns: deque = deque(maxlen=lookback_days)
        
        # Last calculation cache
        self._last_var: Optional[VaRResult] = None
        self._last_calculation_time: Optional[datetime] = None
        
        # Monte Carlo settings
        self._monte_carlo_simulations = 10000
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info("VaRCalculator initialized")
    
    def update_returns(
        self,
        symbol: str,
        return_value: float,
        timestamp: Optional[datetime] = None,
    ):
        """
        Update returns for a symbol.
        
        Args:
            symbol: Trading symbol
            return_value: Daily return (as decimal, e.g., 0.01 for 1%)
            timestamp: Return timestamp
        """
        with self._lock:
            self._returns_history[symbol].append({
                "timestamp": timestamp or datetime.utcnow(),
                "return": return_value,
            })
    
    def update_portfolio_return(
        self,
        return_value: float,
        timestamp: Optional[datetime] = None,
    ):
        """Update portfolio-level returns"""
        with self._lock:
            self._portfolio_returns.append({
                "timestamp": timestamp or datetime.utcnow(),
                "return": return_value,
            })
    
    def calculate_var(
        self,
        portfolio_state: PortfolioState,
        method: Optional[VaRMethod] = None,
        confidence_level: Optional[Decimal] = None,
        holding_period_days: Optional[int] = None,
    ) -> VaRResult:
        """
        Calculate Value at Risk for the portfolio.
        
        Args:
            portfolio_state: Current portfolio state
            method: VaR calculation method
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            holding_period_days: Holding period in days
            
        Returns:
            VaRResult with calculated VaR and CVaR
        """
        with self._lock:
            method = method or self.default_method
            confidence = float(confidence_level or self.risk_limits.var_confidence_level)
            holding_days = holding_period_days or self.holding_period_days
            
            portfolio_value = float(portfolio_state.total_equity)
            
            if method == VaRMethod.HISTORICAL:
                return self._calculate_historical_var(
                    portfolio_value, confidence, holding_days
                )
            elif method == VaRMethod.PARAMETRIC:
                return self._calculate_parametric_var(
                    portfolio_value, confidence, holding_days
                )
            elif method == VaRMethod.MONTE_CARLO:
                return self._calculate_monte_carlo_var(
                    portfolio_value, confidence, holding_days
                )
            else:
                return self._calculate_historical_var(
                    portfolio_value, confidence, holding_days
                )
    
    def _calculate_historical_var(
        self,
        portfolio_value: float,
        confidence_level: float,
        holding_period_days: int,
    ) -> VaRResult:
        """
        Calculate Historical VaR using actual returns.
        
        VaR = Portfolio Value × |Percentile of Historical Returns|
        """
        if len(self._portfolio_returns) < 30:
            logger.warning("Insufficient returns data for Historical VaR")
            return self._create_zero_var_result(VaRMethod.HISTORICAL, confidence_level, holding_period_days)
        
        returns = np.array([r["return"] for r in self._portfolio_returns])
        
        # Calculate VaR at the specified confidence level
        var_percentile = 1 - confidence_level
        var_return = np.percentile(returns, var_percentile * 100)
        
        # Scale to holding period
        var_return_scaled = var_return * np.sqrt(holding_period_days)
        
        # Calculate VaR value
        var_value = portfolio_value * abs(var_return_scaled)
        var_pct = abs(var_return_scaled)
        
        # Calculate CVaR (Expected Shortfall)
        cvar_returns = returns[returns <= var_return]
        if len(cvar_returns) > 0:
            cvar_return = np.mean(cvar_returns) * np.sqrt(holding_period_days)
            cvar_value = portfolio_value * abs(cvar_return)
            cvar_pct = abs(cvar_return)
        else:
            cvar_value = var_value
            cvar_pct = var_pct
        
        result = VaRResult(
            var_value=Decimal(str(var_value)),
            var_pct=Decimal(str(var_pct)),
            cvar_value=Decimal(str(cvar_value)),
            cvar_pct=Decimal(str(cvar_pct)),
            confidence_level=Decimal(str(confidence_level)),
            method=VaRMethod.HISTORICAL,
            holding_period_days=holding_period_days,
        )
        
        self._last_var = result
        self._last_calculation_time = datetime.utcnow()
        
        return result
    
    def _calculate_parametric_var(
        self,
        portfolio_value: float,
        confidence_level: float,
        holding_period_days: int,
    ) -> VaRResult:
        """
        Calculate Parametric (Variance-Covariance) VaR.
        
        VaR = Portfolio Value × (μ - z × σ) × √t
        where z is the z-score for the confidence level
        """
        if len(self._portfolio_returns) < 30:
            logger.warning("Insufficient returns data for Parametric VaR")
            return self._create_zero_var_result(VaRMethod.PARAMETRIC, confidence_level, holding_period_days)
        
        returns = np.array([r["return"] for r in self._portfolio_returns])
        
        # Calculate mean and standard deviation
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        # Get z-score for confidence level
        z_score = stats.norm.ppf(1 - confidence_level)
        
        # Calculate VaR
        var_return = (mean_return - z_score * std_return) * np.sqrt(holding_period_days)
        var_value = portfolio_value * abs(var_return)
        var_pct = abs(var_return)
        
        # Calculate CVaR
        # CVaR = μ - σ × φ(z) / (1 - confidence)
        # where φ(z) is the standard normal PDF
        phi_z = stats.norm.pdf(z_score)
        cvar_return = (mean_return - std_return * phi_z / (1 - confidence_level)) * np.sqrt(holding_period_days)
        cvar_value = portfolio_value * abs(cvar_return)
        cvar_pct = abs(cvar_return)
        
        result = VaRResult(
            var_value=Decimal(str(var_value)),
            var_pct=Decimal(str(var_pct)),
            cvar_value=Decimal(str(cvar_value)),
            cvar_pct=Decimal(str(cvar_pct)),
            confidence_level=Decimal(str(confidence_level)),
            method=VaRMethod.PARAMETRIC,
            holding_period_days=holding_period_days,
        )
        
        self._last_var = result
        self._last_calculation_time = datetime.utcnow()
        
        return result
    
    def _calculate_monte_carlo_var(
        self,
        portfolio_value: float,
        confidence_level: float,
        holding_period_days: int,
    ) -> VaRResult:
        """
        Calculate Monte Carlo VaR.
        
        Simulates many possible portfolio paths and calculates
        VaR from the distribution of outcomes.
        """
        if len(self._portfolio_returns) < 30:
            logger.warning("Insufficient returns data for Monte Carlo VaR")
            return self._create_zero_var_result(VaRMethod.MONTE_CARLO, confidence_level, holding_period_days)
        
        returns = np.array([r["return"] for r in self._portfolio_returns])
        
        # Calculate parameters from historical returns
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        # Generate random returns
        np.random.seed(42)  # For reproducibility
        simulated_returns = np.random.normal(
            mean_return,
            std_return,
            self._monte_carlo_simulations * holding_period_days
        )
        
        # Reshape and compound returns
        simulated_returns = simulated_returns.reshape(self._monte_carlo_simulations, holding_period_days)
        compounded_returns = np.prod(1 + simulated_returns, axis=1) - 1
        
        # Calculate VaR
        var_percentile = 1 - confidence_level
        var_return = np.percentile(compounded_returns, var_percentile * 100)
        var_value = portfolio_value * abs(var_return)
        var_pct = abs(var_return)
        
        # Calculate CVaR
        cvar_returns = compounded_returns[compounded_returns <= var_return]
        if len(cvar_returns) > 0:
            cvar_return = np.mean(cvar_returns)
            cvar_value = portfolio_value * abs(cvar_return)
            cvar_pct = abs(cvar_return)
        else:
            cvar_value = var_value
            cvar_pct = var_pct
        
        result = VaRResult(
            var_value=Decimal(str(var_value)),
            var_pct=Decimal(str(var_pct)),
            cvar_value=Decimal(str(cvar_value)),
            cvar_pct=Decimal(str(cvar_pct)),
            confidence_level=Decimal(str(confidence_level)),
            method=VaRMethod.MONTE_CARLO,
            holding_period_days=holding_period_days,
        )
        
        self._last_var = result
        self._last_calculation_time = datetime.utcnow()
        
        return result
    
    def _create_zero_var_result(
        self,
        method: VaRMethod,
        confidence_level: float,
        holding_period_days: int,
    ) -> VaRResult:
        """Create a zero VaR result when data is insufficient"""
        return VaRResult(
            var_value=Decimal("0"),
            var_pct=Decimal("0"),
            cvar_value=Decimal("0"),
            cvar_pct=Decimal("0"),
            confidence_level=Decimal(str(confidence_level)),
            method=method,
            holding_period_days=holding_period_days,
        )
    
    def calculate_component_var(
        self,
        symbol: str,
        portfolio_state: PortfolioState,
    ) -> Optional[VaRResult]:
        """
        Calculate Component VaR for a specific position.
        Shows contribution of a position to total portfolio VaR.
        """
        with self._lock:
            if symbol not in portfolio_state.positions:
                return None
            
            position = portfolio_state.positions[symbol]
            position_value = float(position.market_value)
            
            # Get position returns
            if len(self._returns_history[symbol]) < 30:
                return None
            
            returns = np.array([r["return"] for r in self._returns_history[symbol]])
            
            # Calculate position VaR
            confidence = float(self.risk_limits.var_confidence_level)
            var_percentile = 1 - confidence
            var_return = np.percentile(returns, var_percentile * 100)
            var_value = position_value * abs(var_return)
            var_pct = abs(var_return)
            
            # Calculate CVaR
            cvar_returns = returns[returns <= var_return]
            if len(cvar_returns) > 0:
                cvar_return = np.mean(cvar_returns)
                cvar_value = position_value * abs(cvar_return)
                cvar_pct = abs(cvar_return)
            else:
                cvar_value = var_value
                cvar_pct = var_pct
            
            return VaRResult(
                var_value=Decimal(str(var_value)),
                var_pct=Decimal(str(var_pct)),
                cvar_value=Decimal(str(cvar_value)),
                cvar_pct=Decimal(str(cvar_pct)),
                confidence_level=self.risk_limits.var_confidence_level,
                method=VaRMethod.HISTORICAL,
                holding_period_days=1,
            )
    
    def validate_trade(
        self,
        symbol: str,
        quantity: Decimal,
        price: Decimal,
        portfolio_state: PortfolioState,
    ) -> ValidationResult:
        """
        Validate a potential trade against VaR limits.
        
        Args:
            symbol: Trading symbol
            quantity: Trade quantity
            price: Trade price
            portfolio_state: Current portfolio state
            
        Returns:
            ValidationResult with pass/fail and details
        """
        with self._lock:
            # Calculate current VaR
            current_var = self.calculate_var(portfolio_state)
            
            equity = portfolio_state.total_equity
            if equity == 0:
                return ValidationResult(
                    is_valid=False,
                    rejection_reason="Portfolio equity is zero",
                    risk_level=RiskLevel.CRITICAL,
                )
            
            current_var_pct = current_var.var_pct
            limit = self.risk_limits.max_daily_var_pct
            
            # Check if current VaR already exceeds limit
            if current_var_pct > limit:
                return ValidationResult(
                    is_valid=False,
                    rejection_reason=f"Current VaR {current_var_pct:.2%} exceeds limit {limit:.2%}",
                    risk_level=RiskLevel.HIGH,
                    details={
                        "current_var_pct": float(current_var_pct),
                        "limit": float(limit),
                    }
                )
            
            # Estimate VaR impact of trade (simplified)
            trade_value = abs(quantity) * price
            trade_weight = trade_value / equity
            
            # Get symbol volatility
            if len(self._returns_history[symbol]) >= 30:
                returns = np.array([r["return"] for r in self._returns_history[symbol]])
                symbol_volatility = np.std(returns, ddof=1)
            else:
                # Use portfolio volatility as proxy
                if len(self._portfolio_returns) >= 30:
                    returns = np.array([r["return"] for r in self._portfolio_returns])
                    symbol_volatility = np.std(returns, ddof=1)
                else:
                    symbol_volatility = 0.02  # Default 2% daily volatility
            
            # Estimate marginal VaR increase
            z_score = stats.norm.ppf(1 - float(self.risk_limits.var_confidence_level))
            marginal_var_increase = trade_weight * symbol_volatility * abs(z_score)
            estimated_var_pct = current_var_pct + Decimal(str(marginal_var_increase))
            
            if estimated_var_pct > limit:
                return ValidationResult(
                    is_valid=False,
                    rejection_reason=f"Trade would exceed VaR limit: {estimated_var_pct:.2%}",
                    risk_level=RiskLevel.HIGH,
                    details={
                        "current_var_pct": float(current_var_pct),
                        "estimated_var_pct": float(estimated_var_pct),
                        "limit": float(limit),
                        "marginal_increase": float(marginal_var_increase),
                    }
                )
            
            # Alert if approaching limit
            alert_threshold = limit * Decimal("0.90")
            warnings = []
            if estimated_var_pct > alert_threshold:
                warnings.append(
                    f"Trade brings VaR close to limit: {estimated_var_pct:.2%} "
                    f"(limit: {limit:.2%})"
                )
            
            return ValidationResult(
                is_valid=True,
                risk_level=RiskLevel.MEDIUM if warnings else RiskLevel.LOW,
                warnings=warnings,
                details={
                    "current_var_pct": float(current_var_pct),
                    "estimated_var_pct": float(estimated_var_pct),
                    "limit": float(limit),
                }
            )
    
    def get_var_report(self) -> Dict[str, Any]:
        """Get comprehensive VaR report"""
        with self._lock:
            report = {
                "last_calculation": self._last_calculation_time.isoformat() if self._last_calculation_time else None,
                "settings": {
                    "default_method": self.default_method.value,
                    "lookback_days": self.lookback_days,
                    "holding_period_days": self.holding_period_days,
                    "confidence_level": float(self.risk_limits.var_confidence_level),
                },
                "limits": {
                    "max_daily_var_pct": float(self.risk_limits.max_daily_var_pct),
                },
            }
            
            if self._last_var:
                report["last_var"] = {
                    "var_value": float(self._last_var.var_value),
                    "var_pct": float(self._last_var.var_pct),
                    "cvar_value": float(self._last_var.cvar_value),
                    "cvar_pct": float(self._last_var.cvar_pct),
                    "method": self._last_var.method.value,
                    "confidence_level": float(self._last_var.confidence_level),
                    "holding_period_days": self._last_var.holding_period_days,
                }
            
            return report
    
    def compare_methods(
        self,
        portfolio_state: PortfolioState,
    ) -> Dict[VaRMethod, VaRResult]:
        """Compare VaR calculated by all methods"""
        with self._lock:
            results = {}
            for method in VaRMethod:
                try:
                    result = self.calculate_var(portfolio_state, method=method)
                    results[method] = result
                except Exception as e:
                    logger.error(f"Error calculating {method.value} VaR: {e}")
            return results
    
    def reset(self):
        """Reset VaR calculator state"""
        with self._lock:
            self._returns_history.clear()
            self._portfolio_returns.clear()
            self._last_var = None
            self._last_calculation_time = None
            logger.info("VaRCalculator reset")


def calculate_bootstrap_var(
    returns: np.ndarray,
    confidence_level: float = 0.95,
    num_bootstrap_samples: int = 1000,
) -> Tuple[float, float]:
    """
    Calculate VaR using bootstrap method.
    
    Args:
        returns: Array of historical returns
        confidence_level: Confidence level for VaR
        num_bootstrap_samples: Number of bootstrap samples
        
    Returns:
        Tuple of (VaR, CVaR)
    """
    var_samples = []
    cvar_samples = []
    
    for _ in range(num_bootstrap_samples):
        # Sample with replacement
        sample = np.random.choice(returns, size=len(returns), replace=True)
        
        # Calculate VaR for this sample
        var_percentile = 1 - confidence_level
        var = np.percentile(sample, var_percentile * 100)
        var_samples.append(var)
        
        # Calculate CVaR for this sample
        cvar_returns = sample[sample <= var]
        if len(cvar_returns) > 0:
            cvar_samples.append(np.mean(cvar_returns))
        else:
            cvar_samples.append(var)
    
    # Return median of bootstrap samples
    return np.median(var_samples), np.median(cvar_samples)


def calculate_stressed_var(
    returns: np.ndarray,
    stress_scenarios: Dict[str, float],
    confidence_level: float = 0.99,
) -> Dict[str, float]:
    """
    Calculate Stressed VaR by applying stress scenarios.
    
    Args:
        returns: Array of historical returns
        stress_scenarios: Dict of scenario name -> stress multiplier
        confidence_level: Confidence level for VaR
        
    Returns:
        Dict of scenario name -> stressed VaR
    """
    results = {}
    
    # Baseline VaR
    var_percentile = 1 - confidence_level
    baseline_var = np.percentile(returns, var_percentile * 100)
    results["baseline"] = abs(baseline_var)
    
    # Stressed VaR for each scenario
    for scenario_name, multiplier in stress_scenarios.items():
        stressed_returns = returns * multiplier
        stressed_var = np.percentile(stressed_returns, var_percentile * 100)
        results[scenario_name] = abs(stressed_var)
    
    return results
