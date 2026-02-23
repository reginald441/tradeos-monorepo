"""
Position Sizing Module for TradeOS Risk Engine

Implements multiple position sizing methods:
- Fixed fractional sizing
- Kelly Criterion
- Volatility-based sizing (ATR)
- Risk-per-trade sizing
- Optimal f calculation

All methods enforce strict capital protection.
"""

import logging
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import numpy as np

from .models.risk_profile import (
    PositionSizingParams,
    PositionSizingMethod,
    SizingRecommendation,
    PortfolioState,
    Position,
)
from .config.risk_limits import get_risk_limits, SubscriptionTier


logger = logging.getLogger(__name__)


class PositionSizer:
    """
    Position sizing calculator with multiple methods.
    All sizing is constrained by risk limits.
    """
    
    def __init__(
        self,
        params: Optional[PositionSizingParams] = None,
        tier: SubscriptionTier = SubscriptionTier.PRO
    ):
        self.params = params or PositionSizingParams()
        self.risk_limits = get_risk_limits(tier)
        self._kelly_cache: Dict[str, Dict[str, Decimal]] = {}
        self._optimal_f_cache: Dict[str, Decimal] = {}
        
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
        Calculate optimal position size using configured method.
        
        Args:
            symbol: Trading symbol
            entry_price: Planned entry price
            portfolio_state: Current portfolio state
            stop_loss: Stop loss price (required for risk-based methods)
            atr: Average True Range (required for volatility-based method)
            trade_history: Historical trades for Kelly/Optimal f
            price_history: Price history for volatility calculations
            
        Returns:
            SizingRecommendation with recommended quantity and risk metrics
        """
        method = self.params.method
        
        if method == PositionSizingMethod.FIXED_FRACTIONAL:
            return self._fixed_fractional_sizing(
                symbol, entry_price, portfolio_state
            )
        elif method == PositionSizingMethod.KELLY_CRITERION:
            return self._kelly_sizing(
                symbol, entry_price, portfolio_state, trade_history
            )
        elif method == PositionSizingMethod.VOLATILITY_BASED:
            return self._volatility_sizing(
                symbol, entry_price, portfolio_state, atr
            )
        elif method == PositionSizingMethod.RISK_PER_TRADE:
            return self._risk_per_trade_sizing(
                symbol, entry_price, portfolio_state, stop_loss
            )
        elif method == PositionSizingMethod.OPTIMAL_F:
            return self._optimal_f_sizing(
                symbol, entry_price, portfolio_state, price_history
            )
        else:
            # Default to risk-per-trade
            return self._risk_per_trade_sizing(
                symbol, entry_price, portfolio_state, stop_loss
            )
    
    def _fixed_fractional_sizing(
        self,
        symbol: str,
        entry_price: Decimal,
        portfolio_state: PortfolioState,
    ) -> SizingRecommendation:
        """
        Fixed fractional position sizing.
        Position size = Account Equity × Fixed Fraction / Entry Price
        """
        equity = portfolio_state.total_equity
        fraction = self.params.fixed_fraction_pct
        
        # Calculate position value
        position_value = equity * fraction
        
        # Calculate quantity
        if entry_price > 0:
            quantity = position_value / entry_price
        else:
            quantity = Decimal("0")
        
        # Apply constraints
        quantity, constraints_applied = self._apply_constraints(
            quantity, entry_price, position_value, portfolio_state, symbol
        )
        
        # Recalculate after constraints
        position_value = quantity * entry_price
        risk_amount = position_value * fraction
        risk_pct = (risk_amount / equity) if equity > 0 else Decimal("0")
        
        warnings = []
        if constraints_applied:
            warnings.append("Position size constrained by risk limits")
        
        return SizingRecommendation(
            recommended_quantity=quantity.quantize(Decimal("0.0001"), rounding=ROUND_DOWN),
            recommended_value=position_value,
            risk_amount=risk_amount,
            risk_pct=risk_pct,
            method=PositionSizingMethod.FIXED_FRACTIONAL,
            confidence_score=Decimal("0.85"),
            warnings=warnings,
            details={
                "fixed_fraction_pct": fraction,
                "equity": equity,
                "constraints_applied": constraints_applied,
            }
        )
    
    def _kelly_sizing(
        self,
        symbol: str,
        entry_price: Decimal,
        portfolio_state: PortfolioState,
        trade_history: Optional[List[Dict]] = None,
    ) -> SizingRecommendation:
        """
        Kelly Criterion position sizing.
        f* = (p × b - q) / b
        where: p = win rate, q = loss rate (1-p), b = win/loss ratio
        
        Uses fractional Kelly for safety (default 25% of full Kelly).
        """
        equity = portfolio_state.total_equity
        
        # Calculate Kelly parameters from history or use defaults
        if trade_history and len(trade_history) >= 10:
            win_rate, avg_win_loss_ratio = self._calculate_kelly_params(trade_history)
        else:
            # Use provided defaults or conservative estimates
            win_rate = self.params.win_rate or Decimal("0.50")
            avg_win_loss_ratio = self.params.avg_win_loss_ratio or Decimal("1.5")
        
        # Calculate Kelly fraction
        loss_rate = Decimal("1") - win_rate
        
        if avg_win_loss_ratio > 0:
            kelly_fraction = (win_rate * avg_win_loss_ratio - loss_rate) / avg_win_loss_ratio
        else:
            kelly_fraction = Decimal("0")
        
        # Apply safety fraction (default 25% Kelly)
        safe_kelly = max(Decimal("0"), kelly_fraction * self.params.kelly_fraction)
        
        # Cap at max risk per trade
        max_risk = self.risk_limits.max_risk_per_trade_pct
        safe_kelly = min(safe_kelly, max_risk)
        
        # Calculate position
        position_value = equity * safe_kelly
        
        if entry_price > 0:
            quantity = position_value / entry_price
        else:
            quantity = Decimal("0")
        
        # Apply constraints
        quantity, constraints_applied = self._apply_constraints(
            quantity, entry_price, position_value, portfolio_state, symbol
        )
        
        # Recalculate
        position_value = quantity * entry_price
        risk_amount = position_value * safe_kelly
        risk_pct = (risk_amount / equity) if equity > 0 else Decimal("0")
        
        warnings = []
        if kelly_fraction > max_risk:
            warnings.append(f"Full Kelly ({kelly_fraction:.2%}) exceeds max risk, capped at {max_risk:.2%}")
        if constraints_applied:
            warnings.append("Position size constrained by risk limits")
        
        return SizingRecommendation(
            recommended_quantity=quantity.quantize(Decimal("0.0001"), rounding=ROUND_DOWN),
            recommended_value=position_value,
            risk_amount=risk_amount,
            risk_pct=risk_pct,
            method=PositionSizingMethod.KELLY_CRITERION,
            confidence_score=Decimal("0.75") if trade_history else Decimal("0.50"),
            warnings=warnings,
            details={
                "full_kelly": kelly_fraction,
                "fractional_kelly": safe_kelly,
                "kelly_fraction_used": self.params.kelly_fraction,
                "win_rate": win_rate,
                "win_loss_ratio": avg_win_loss_ratio,
                "constraints_applied": constraints_applied,
            }
        )
    
    def _volatility_sizing(
        self,
        symbol: str,
        entry_price: Decimal,
        portfolio_state: PortfolioState,
        atr: Optional[Decimal],
    ) -> SizingRecommendation:
        """
        Volatility-based position sizing using ATR.
        Position size = (Account × Risk%) / (ATR × Multiplier)
        """
        equity = portfolio_state.total_equity
        
        if atr is None or atr <= 0:
            logger.warning(f"ATR not provided for {symbol}, falling back to fixed fractional")
            return self._fixed_fractional_sizing(symbol, entry_price, portfolio_state)
        
        # Calculate risk amount
        risk_pct = self.params.risk_per_trade_pct
        risk_amount = equity * risk_pct
        
        # Calculate position size based on ATR
        atr_risk = atr * self.params.atr_multiplier
        
        if atr_risk > 0:
            quantity = risk_amount / atr_risk
        else:
            quantity = Decimal("0")
        
        position_value = quantity * entry_price
        
        # Apply constraints
        quantity, constraints_applied = self._apply_constraints(
            quantity, entry_price, position_value, portfolio_state, symbol
        )
        
        # Recalculate
        position_value = quantity * entry_price
        risk_amount = quantity * atr_risk
        risk_pct = (risk_amount / equity) if equity > 0 else Decimal("0")
        
        warnings = []
        if constraints_applied:
            warnings.append("Position size constrained by risk limits")
        
        return SizingRecommendation(
            recommended_quantity=quantity.quantize(Decimal("0.0001"), rounding=ROUND_DOWN),
            recommended_value=position_value,
            risk_amount=risk_amount,
            risk_pct=risk_pct,
            method=PositionSizingMethod.VOLATILITY_BASED,
            confidence_score=Decimal("0.80"),
            warnings=warnings,
            details={
                "atr": atr,
                "atr_multiplier": self.params.atr_multiplier,
                "atr_risk": atr_risk,
                "constraints_applied": constraints_applied,
            }
        )
    
    def _risk_per_trade_sizing(
        self,
        symbol: str,
        entry_price: Decimal,
        portfolio_state: PortfolioState,
        stop_loss: Optional[Decimal],
    ) -> SizingRecommendation:
        """
        Risk-per-trade position sizing.
        Position size = (Account × Risk%) / |Entry - Stop Loss|
        """
        equity = portfolio_state.total_equity
        risk_pct = self.params.risk_per_trade_pct
        risk_amount = equity * risk_pct
        
        # Calculate risk per unit
        if stop_loss is not None and stop_loss > 0:
            risk_per_unit = abs(entry_price - stop_loss)
        else:
            # Default to 1% risk if no stop loss
            risk_per_unit = entry_price * Decimal("0.01")
            logger.warning(f"No stop loss provided for {symbol}, using 1% default")
        
        if risk_per_unit > 0:
            quantity = risk_amount / risk_per_unit
        else:
            quantity = Decimal("0")
        
        position_value = quantity * entry_price
        
        # Apply constraints
        quantity, constraints_applied = self._apply_constraints(
            quantity, entry_price, position_value, portfolio_state, symbol
        )
        
        # Recalculate
        position_value = quantity * entry_price
        risk_amount = quantity * risk_per_unit
        risk_pct = (risk_amount / equity) if equity > 0 else Decimal("0")
        
        warnings = []
        if stop_loss is None:
            warnings.append("No stop loss provided, using default 1% risk")
        if constraints_applied:
            warnings.append("Position size constrained by risk limits")
        
        return SizingRecommendation(
            recommended_quantity=quantity.quantize(Decimal("0.0001"), rounding=ROUND_DOWN),
            recommended_value=position_value,
            risk_amount=risk_amount,
            risk_pct=risk_pct,
            method=PositionSizingMethod.RISK_PER_TRADE,
            confidence_score=Decimal("0.90"),
            warnings=warnings,
            details={
                "stop_loss": stop_loss,
                "risk_per_unit": risk_per_unit,
                "target_risk_pct": self.params.risk_per_trade_pct,
                "constraints_applied": constraints_applied,
            }
        )
    
    def _optimal_f_sizing(
        self,
        symbol: str,
        entry_price: Decimal,
        portfolio_state: PortfolioState,
        price_history: Optional[List[Decimal]],
    ) -> SizingRecommendation:
        """
        Optimal f position sizing (Ralph Vince).
        Finds the optimal fraction that maximizes geometric growth.
        Uses conservative fraction for safety.
        """
        equity = portfolio_state.total_equity
        
        if price_history is None or len(price_history) < self.params.optimal_f_lookback:
            logger.warning(f"Insufficient price history for {symbol}, falling back to fixed fractional")
            return self._fixed_fractional_sizing(symbol, entry_price, portfolio_state)
        
        # Calculate optimal f
        optimal_f = self._calculate_optimal_f(price_history)
        
        # Apply safety fraction
        safe_f = optimal_f * self.params.optimal_f_fraction
        
        # Cap at max risk per trade
        max_risk = self.risk_limits.max_risk_per_trade_pct
        safe_f = min(safe_f, max_risk)
        
        # Calculate position
        position_value = equity * safe_f
        
        if entry_price > 0:
            quantity = position_value / entry_price
        else:
            quantity = Decimal("0")
        
        # Apply constraints
        quantity, constraints_applied = self._apply_constraints(
            quantity, entry_price, position_value, portfolio_state, symbol
        )
        
        # Recalculate
        position_value = quantity * entry_price
        risk_amount = position_value * safe_f
        risk_pct = (risk_amount / equity) if equity > 0 else Decimal("0")
        
        warnings = []
        if optimal_f > max_risk:
            warnings.append(f"Optimal f ({optimal_f:.2%}) exceeds max risk, capped at {max_risk:.2%}")
        if constraints_applied:
            warnings.append("Position size constrained by risk limits")
        
        return SizingRecommendation(
            recommended_quantity=quantity.quantize(Decimal("0.0001"), rounding=ROUND_DOWN),
            recommended_value=position_value,
            risk_amount=risk_amount,
            risk_pct=risk_pct,
            method=PositionSizingMethod.OPTIMAL_F,
            confidence_score=Decimal("0.70"),
            warnings=warnings,
            details={
                "optimal_f": optimal_f,
                "safe_f": safe_f,
                "fraction_used": self.params.optimal_f_fraction,
                "lookback": self.params.optimal_f_lookback,
                "constraints_applied": constraints_applied,
            }
        )
    
    def _apply_constraints(
        self,
        quantity: Decimal,
        entry_price: Decimal,
        position_value: Decimal,
        portfolio_state: PortfolioState,
        symbol: str,
    ) -> Tuple[Decimal, bool]:
        """
        Apply all risk constraints to position size.
        Returns (constrained_quantity, was_constrained)
        """
        was_constrained = False
        equity = portfolio_state.total_equity
        
        # Constraint 1: Max position size percentage
        max_position_value = equity * self.risk_limits.max_position_size_pct
        if position_value > max_position_value:
            quantity = max_position_value / entry_price
            position_value = max_position_value
            was_constrained = True
            logger.debug(f"Position constrained by max_position_size_pct: {quantity}")
        
        # Constraint 2: Max single asset exposure
        current_exposure = self._get_current_exposure(portfolio_state, symbol)
        max_asset_exposure = equity * self.risk_limits.max_single_asset_exposure_pct
        available_exposure = max_asset_exposure - current_exposure
        
        if position_value > available_exposure and available_exposure > 0:
            quantity = available_exposure / entry_price
            position_value = available_exposure
            was_constrained = True
            logger.debug(f"Position constrained by max_single_asset_exposure: {quantity}")
        
        # Constraint 3: Max total exposure
        current_total_exposure = portfolio_state.total_exposure
        max_total_exposure = equity * self.risk_limits.max_total_exposure_pct
        available_total = max_total_exposure - current_total_exposure
        
        if position_value > available_total and available_total > 0:
            quantity = available_total / entry_price
            position_value = available_total
            was_constrained = True
            logger.debug(f"Position constrained by max_total_exposure: {quantity}")
        
        # Constraint 4: Minimum position size
        min_value = self.risk_limits.min_position_size
        if position_value < min_value and position_value > 0:
            quantity = Decimal("0")
            was_constrained = True
            logger.debug(f"Position below minimum size, rejected")
        
        # Constraint 5: Available buying power
        if position_value > portfolio_state.buying_power:
            quantity = portfolio_state.buying_power / entry_price
            position_value = portfolio_state.buying_power
            was_constrained = True
            logger.debug(f"Position constrained by buying_power: {quantity}")
        
        # Constraint 6: Leverage limit
        new_exposure = current_total_exposure + position_value
        if equity > 0:
            new_leverage = new_exposure / equity
            if new_leverage > self.risk_limits.max_leverage:
                max_position = (self.risk_limits.max_leverage * equity) - current_total_exposure
                if max_position > 0:
                    quantity = max_position / entry_price
                    position_value = max_position
                else:
                    quantity = Decimal("0")
                    position_value = Decimal("0")
                was_constrained = True
                logger.debug(f"Position constrained by max_leverage: {quantity}")
        
        return quantity.quantize(Decimal("0.0001"), rounding=ROUND_DOWN), was_constrained
    
    def _get_current_exposure(
        self,
        portfolio_state: PortfolioState,
        symbol: str
    ) -> Decimal:
        """Get current exposure for a symbol"""
        if symbol in portfolio_state.positions:
            return portfolio_state.positions[symbol].market_value
        return Decimal("0")
    
    def _calculate_kelly_params(
        self,
        trade_history: List[Dict]
    ) -> Tuple[Decimal, Decimal]:
        """Calculate win rate and win/loss ratio from trade history"""
        wins = []
        losses = []
        
        for trade in trade_history:
            pnl = Decimal(str(trade.get("pnl", 0)))
            if pnl > 0:
                wins.append(pnl)
            elif pnl < 0:
                losses.append(abs(pnl))
        
        total_trades = len(wins) + len(losses)
        if total_trades == 0:
            return Decimal("0.50"), Decimal("1.0")
        
        win_rate = Decimal(str(len(wins))) / Decimal(str(total_trades))
        
        avg_win = sum(wins) / len(wins) if wins else Decimal("0")
        avg_loss = sum(losses) / len(losses) if losses else Decimal("1")
        
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else Decimal("1")
        
        return win_rate, win_loss_ratio
    
    def _calculate_optimal_f(self, price_history: List[Decimal]) -> Decimal:
        """
        Calculate Optimal f using the method from Ralph Vince.
        Uses binary search to find the f that maximizes Terminal Wealth Relative.
        """
        if len(price_history) < 2:
            return Decimal("0.01")
        
        # Calculate returns
        returns = []
        for i in range(1, len(price_history)):
            if price_history[i-1] > 0:
                ret = (price_history[i] - price_history[i-1]) / price_history[i-1]
                returns.append(float(ret))
        
        if not returns:
            return Decimal("0.01")
        
        # Find optimal f using binary search
        # Maximize: TWR = Product(1 + f × (-trade/WCS))
        # where WCS = Worst Case Scenario (largest loss)
        
        worst_loss = min(returns)
        if worst_loss >= 0:
            return Decimal("0.01")  # No losses, use minimum
        
        def calculate_twr(f: float) -> float:
            twr = 1.0
            for ret in returns:
                hpr = 1 + f * (-ret / worst_loss)
                if hpr <= 0:
                    return 0.0
                twr *= hpr
            return twr
        
        # Binary search for optimal f
        low, high = 0.0, 1.0
        optimal_f = 0.0
        best_twr = 0.0
        
        for _ in range(50):  # 50 iterations for precision
            mid = (low + high) / 2
            twr = calculate_twr(mid)
            
            if twr > best_twr:
                best_twr = twr
                optimal_f = mid
            
            # Check nearby values
            twr_low = calculate_twr(mid - 0.01)
            twr_high = calculate_twr(mid + 0.01)
            
            if twr_low > twr:
                high = mid
            elif twr_high > twr:
                low = mid
            else:
                break
        
        # Convert to Decimal and cap at reasonable maximum
        result = Decimal(str(optimal_f))
        return min(result, Decimal("0.25"))  # Cap at 25%
    
    def compare_methods(
        self,
        symbol: str,
        entry_price: Decimal,
        portfolio_state: PortfolioState,
        stop_loss: Optional[Decimal] = None,
        atr: Optional[Decimal] = None,
        trade_history: Optional[List[Dict]] = None,
        price_history: Optional[List[Decimal]] = None,
    ) -> Dict[PositionSizingMethod, SizingRecommendation]:
        """
        Compare all position sizing methods for a trade.
        Useful for analysis and method selection.
        """
        results = {}
        original_method = self.params.method
        
        for method in PositionSizingMethod:
            self.params.method = method
            try:
                result = self.calculate_position_size(
                    symbol=symbol,
                    entry_price=entry_price,
                    portfolio_state=portfolio_state,
                    stop_loss=stop_loss,
                    atr=atr,
                    trade_history=trade_history,
                    price_history=price_history,
                )
                results[method] = result
            except Exception as e:
                logger.error(f"Error calculating {method.value}: {e}")
        
        # Restore original method
        self.params.method = original_method
        
        return results
